"""Inference and evaluation pipeline for Sasol Customer Value competition.

CLI entrypoint to train baseline or ensemble models, evaluate on local OOT validation,
and generate formatted, verified competition submissions for test.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sasol_customer_value.config import (
    DEFAULT_VAL_CUTOFF,
    ID_COL,
    PUBLIC_HISTORY_END_EXCLUSIVE,
    SUBMISSIONS_DIR,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.data.loader import load_test_customers, load_transactions
from sasol_customer_value.data.snapshots import build_features
from sasol_customer_value.models.baseline import BaselinePipeline, prepare_snapshot
from sasol_customer_value.models.ensemble import DEFAULT_OPP_BIASES, EnsemblePipeline
from sasol_customer_value.utils.helpers import set_seed, validate_submission_format


def get_default_monthly_cutoffs(val_cutoff: str) -> list[str]:
    """All monthly cutoffs from 2024-07-01 whose 3-month outcome window ends <= val_cutoff."""
    pool = [str(d.date()) for d in pd.date_range("2024-07-01", val_cutoff, freq="MS")]
    val_ts = pd.Timestamp(val_cutoff)
    return [
        c
        for c in pool
        if (val_ts.year * 12 + val_ts.month)
        - (pd.Timestamp(c).year * 12 + pd.Timestamp(c).month)
        >= 3
    ]


def train_and_evaluate(
    transactions_df: pd.DataFrame,
    pipeline_type: str = "ensemble",
    train_cutoffs: list[str] | None = None,
    val_cutoff: str = DEFAULT_VAL_CUTOFF,
) -> tuple[BaselinePipeline | EnsemblePipeline, dict[str, float]]:
    """Train pipeline on historical cutoffs and score on validation cutoff."""
    if train_cutoffs is None:
        train_cutoffs = get_default_monthly_cutoffs(val_cutoff)

    print(f"--> Building training snapshot(s) for: {train_cutoffs}")
    X_train_list, y_train_list = [], []
    for cutoff in train_cutoffs:
        X_cut, y_cut = prepare_snapshot(transactions_df, cutoff)
        X_train_list.append(X_cut)
        y_train_list.append(y_cut)

    X_train = pd.concat(X_train_list, axis=0, ignore_index=True)
    y_train = pd.concat(y_train_list, axis=0, ignore_index=True)
    print(
        f"    Training pool: {len(X_train):,} customer examples across {X_train.shape[1]} features"
    )

    if pipeline_type == "ensemble":
        pipeline: BaselinePipeline | EnsemblePipeline = EnsemblePipeline()
        print("--> Fitting Multi-Model Ensemble (LightGBM + CatBoost + XGBoost)...")
    else:
        pipeline = BaselinePipeline()
        print("--> Fitting Baseline LightGBM models...")

    pipeline.fit(X_train, y_train)

    print(f"--> Preparing validation cohort for cutoff: {val_cutoff}")
    X_val, y_val = prepare_snapshot(transactions_df, val_cutoff)
    metrics = pipeline.evaluate(X_val, y_val)

    print("=" * 60)
    print(
        f"LOCAL OOT VALIDATION RESULTS ({pipeline_type.upper()} | Cutoff: {val_cutoff})"
    )
    print("=" * 60)
    print(f"  Final Score (Competition Loss): {metrics['final_score']:.5f}")
    print(f"  CLV_fuel RMSE:                  {metrics['rmse_fuel']:.5f}")
    print(f"  CLV_nonfuel RMSE:               {metrics['rmse_nonfuel']:.5f}")
    print(f"  Opportunity Weighted F1:        {metrics['weighted_f1_opportunity']:.5f}")
    print("=" * 60)

    return pipeline, metrics


def generate_submission(
    transactions_df: pd.DataFrame,
    pipeline: BaselinePipeline | EnsemblePipeline,
    test_cutoff: str = PUBLIC_HISTORY_END_EXCLUSIVE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Extract features for test.csv customers as of test_cutoff and generate submission."""
    df_test = load_test_customers()
    test_ids = pd.Index(df_test[ID_COL].astype(str))
    print(
        f"--> Generating test features for {len(test_ids):,} customers as of {test_cutoff}..."
    )

    X_test = build_features(
        transactions_df,
        cutoff=test_cutoff,
        customer_ids=test_ids,
    )

    preds = pipeline.predict(X_test)

    # Enforce exact test customer ID order
    preds = preds.set_index(ID_COL).reindex(test_ids).reset_index()

    # Validate against competition schema
    print("--> Verifying submission compliance...")
    validate_submission_format(preds, expected_ids=list(test_ids))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(output_path, index=False)
        print(f"[SUCCESS] Submission saved to: {output_path}")

    return preds


def generate_multiseed_submission(
    transactions_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    seeds: list[int],
    test_cutoff: str = PUBLIC_HISTORY_END_EXCLUSIVE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Train ensemble across multiple seeds and average predictions for lower variance."""
    df_test = load_test_customers()
    test_ids = pd.Index(df_test[ID_COL].astype(str))
    print(
        f"--> Generating test features for {len(test_ids):,} customers as of {test_cutoff}..."
    )
    X_test = build_features(
        transactions_df,
        cutoff=test_cutoff,
        customer_ids=test_ids,
    )

    fuel_list: list[np.ndarray] = []
    nonfuel_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    classes: np.ndarray | None = None

    for s in seeds:
        print(f"    Fitting ensemble with seed {s}...")
        pipeline = EnsemblePipeline(random_seed=s)
        pipeline.fit(X_train, y_train)
        f_pred, nf_pred, p_mat = pipeline.predict_components(X_test)
        fuel_list.append(f_pred)
        nonfuel_list.append(nf_pred)
        probs_list.append(p_mat)
        if classes is None:
            classes = pipeline.classes_

    avg_fuel = np.mean(fuel_list, axis=0)
    avg_nonfuel = np.mean(nonfuel_list, axis=0)
    avg_probs = np.mean(probs_list, axis=0)

    # Calibrate Opportunity with DEFAULT_OPP_BIASES
    assert classes is not None
    bias_vec = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes], dtype=float)
    log_probs = np.log(np.clip(avg_probs, 1e-9, 1.0)) + bias_vec
    pred_opp = classes[log_probs.argmax(axis=1)].copy()

    # Inactivity consistency check
    if "days_since_last_visit" in X_test.columns:
        inactive_mask = (
            (avg_fuel < 0.10)
            & (avg_nonfuel < 0.10)
            & (X_test["days_since_last_visit"] > 60.0)
        )
        pred_opp[inactive_mask] = "Inactivity"

    preds = pd.DataFrame(
        {
            ID_COL: X_test.index.astype(str),
            TARGET_FUEL: avg_fuel,
            TARGET_NONFUEL: avg_nonfuel,
            TARGET_OPPORTUNITY: pred_opp,
        }
    )

    preds = preds.set_index(ID_COL).reindex(test_ids).reset_index()
    print("--> Verifying submission compliance...")
    validate_submission_format(preds, expected_ids=list(test_ids))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(output_path, index=False)
        print(f"[SUCCESS] Multi-seed submission saved to: {output_path}")

    return preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sasol model training & submission runner"
    )
    parser.add_argument(
        "--pipeline",
        choices=["baseline", "ensemble"],
        default="ensemble",
        help="Pipeline type: baseline LightGBM or multi-model ensemble",
    )
    parser.add_argument(
        "--train-cutoffs",
        nargs="+",
        default=None,
        help="Historical cutoffs for training (defaults to all monthly cutoffs ending <= val_cutoff)",
    )
    parser.add_argument(
        "--val-cutoff",
        default=DEFAULT_VAL_CUTOFF,
        help="Validation cutoff date",
    )
    parser.add_argument(
        "--cutoff",
        default=PUBLIC_HISTORY_END_EXCLUSIVE,
        help="Test cutoff date for submission generation",
    )
    parser.add_argument(
        "--output",
        default=str(SUBMISSIONS_DIR / "sub_model_best1.csv"),
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run validation evaluation only without generating submission",
    )
    parser.add_argument(
        "--refit-all",
        action="store_true",
        default=True,
        help="Refit pipeline on all available snapshots (including val) before final submission",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="List of random seeds for multi-seed averaging (e.g. 42 100 2024 777 999)",
    )

    args = parser.parse_args()
    set_seed()

    print("Loading full transaction table...")
    df_trans = load_transactions()

    pipeline, _ = train_and_evaluate(
        df_trans,
        pipeline_type=args.pipeline,
        train_cutoffs=args.train_cutoffs,
        val_cutoff=args.val_cutoff,
    )

    if not args.eval_only:
        full_cutoffs = get_default_monthly_cutoffs(args.cutoff)
        print(
            f"--> Refitting pipeline on all {len(full_cutoffs)} snapshots up to {args.val_cutoff} for final submission..."
        )
        Xs_full, ys_full = [], []
        for c in full_cutoffs:
            Xc, yc = prepare_snapshot(df_trans, c)
            Xs_full.append(Xc)
            ys_full.append(yc)
        X_full = pd.concat(Xs_full, ignore_index=True)
        y_full = pd.concat(ys_full, ignore_index=True)

        if args.seeds and len(args.seeds) > 1 and args.pipeline == "ensemble":
            print(f"--> Running multi-seed bagging with seeds: {args.seeds}...")
            generate_multiseed_submission(
                df_trans,
                X_full,
                y_full,
                seeds=args.seeds,
                test_cutoff=args.cutoff,
                output_path=Path(args.output),
            )
        else:
            if args.refit_all:
                pipeline.fit(X_full, y_full)
            generate_submission(
                df_trans,
                pipeline,
                test_cutoff=args.cutoff,
                output_path=Path(args.output),
            )


if __name__ == "__main__":
    main()
