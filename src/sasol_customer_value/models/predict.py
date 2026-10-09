"""Inference and evaluation pipeline for Sasol Customer Value competition.

CLI entrypoint to train baseline or ensemble models, evaluate on local OOT validation,
and generate formatted, verified competition submissions for test.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from sasol_customer_value.config import (
    DEFAULT_VAL_CUTOFF,
    ID_COL,
    PUBLIC_HISTORY_END_EXCLUSIVE,
    SUBMISSIONS_DIR,
)
from sasol_customer_value.data.loader import load_test_customers, load_transactions
from sasol_customer_value.features.rfm import extract_rfm_features
from sasol_customer_value.models.baseline import BaselinePipeline, prepare_snapshot
from sasol_customer_value.models.ensemble import EnsemblePipeline
from sasol_customer_value.utils.helpers import set_seed, validate_submission_format


def train_and_evaluate(
    transactions_df: pd.DataFrame,
    pipeline_type: str = "ensemble",
    train_cutoffs: list[str] | None = None,
    val_cutoff: str = DEFAULT_VAL_CUTOFF,
) -> tuple[BaselinePipeline | EnsemblePipeline, dict[str, float]]:
    """Train pipeline on historical cutoffs and score on validation cutoff."""
    if train_cutoffs is None:
        train_cutoffs = ["2024-12-01", "2025-03-01", "2025-06-01"]

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

    X_test = extract_rfm_features(
        transactions_df,
        cutoff_date=test_cutoff,
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
        default=["2024-12-01", "2025-03-01", "2025-06-01"],
        help="Historical cutoffs for training",
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
        default=str(SUBMISSIONS_DIR / "sub_ensemble_v2.csv"),
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run validation evaluation only without generating submission",
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
        generate_submission(
            df_trans,
            pipeline,
            test_cutoff=args.cutoff,
            output_path=Path(args.output),
        )


if __name__ == "__main__":
    main()
