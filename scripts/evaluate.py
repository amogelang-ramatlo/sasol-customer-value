"""Unified Out-Of-Time (OOT) Validation Benchmark.

Evaluates Target Specialists (F, N, O) across the 3 official embargoed folds:
- Fold 1: Val Cutoff 2025-03-01 (Train cutoffs <= 2024-12-01)
- Fold 2: Val Cutoff 2025-06-01 (Train cutoffs <= 2025-03-01)
- Fold 3: Val Cutoff 2025-09-01 (Train cutoffs <= 2025-06-01: Sep-Nov 2025)

Strictly enforces 3-month outcome embargo (zero temporal target leakage).
"""

import argparse
import time

import pandas as pd

from sasol_customer_value.config import (
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.models.fuel import FuelModel
from sasol_customer_value.models.metrics import compute_competition_metrics
from sasol_customer_value.models.nonfuel import NonfuelModel
from sasol_customer_value.models.opportunity import OpportunityModel
from sasol_customer_value.models.validation import (
    FoldResult,
    get_embargoed_folds,
    load_fold_data,
    summarize_validation_results,
)


def evaluate_pipeline(config_name: str) -> pd.DataFrame:
    """Run full 3-fold embargoed evaluation for the given configuration."""
    print("\n" + "=" * 80)
    print(f"RUNNING EMBARGOED EVALUATION FOR: {config_name.upper()}")
    print("=" * 80)

    fold_results: list[FoldResult] = []

    for train_cutoffs, val_cutoff in get_embargoed_folds():
        t0 = time.time()
        print(f"\n--> Fold: Val Cutoff = {val_cutoff}")
        print(f"    Train Cutoffs ({len(train_cutoffs)}): {train_cutoffs[0]} to {train_cutoffs[-1]}")

        X_train, y_train, X_val, y_val = load_fold_data(train_cutoffs, val_cutoff)
        n_train, n_val = len(X_train), len(X_val)

        y_f_train = y_train[TARGET_FUEL].to_numpy(dtype=float)
        y_nf_train = y_train[TARGET_NONFUEL].to_numpy(dtype=float)
        y_opp_train = y_train[TARGET_OPPORTUNITY].astype(str).to_numpy()

        # 1. Model F (Fuel)
        fuel_model = FuelModel(variant=config_name)
        fuel_model.fit(X_train, y_f_train)
        pred_f = fuel_model.predict(X_val)

        # 2. Model N (Non-fuel)
        nonfuel_model = NonfuelModel(variant=config_name)
        nonfuel_model.fit(X_train, y_nf_train)
        pred_nf = nonfuel_model.predict(X_val)

        # 3. Model O (Opportunity)
        opp_model = OpportunityModel(variant=config_name)
        opp_model.fit(X_train, y_opp_train)
        pred_opp = opp_model.predict(X_val)

        df_pred = pd.DataFrame(
            {
                TARGET_FUEL: pred_f,
                TARGET_NONFUEL: pred_nf,
                TARGET_OPPORTUNITY: pred_opp,
            }
        )

        metrics = compute_competition_metrics(y_val, df_pred)
        fold_res = FoldResult(
            val_cutoff=val_cutoff,
            n_train_rows=n_train,
            n_val_rows=n_val,
            rmse_fuel=metrics["rmse_fuel"],
            rmse_nonfuel=metrics["rmse_nonfuel"],
            f1_opportunity=metrics["weighted_f1_opportunity"],
            lb_score=metrics["lb_score"],
            y_pred=df_pred,
        )
        fold_results.append(fold_res)
        print(
            f"    Fold {val_cutoff} Result ({time.time() - t0:.1f}s): "
            f"Fuel RMSE={metrics['rmse_fuel']:.5f} | "
            f"Nonfuel RMSE={metrics['rmse_nonfuel']:.5f} | "
            f"Opp F1={metrics['weighted_f1_opportunity']:.5f} | "
            f"Composite Score={metrics['lb_score']:.5f}"
        )

    return summarize_validation_results(fold_results, model_name=config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models across embargoed OOT folds")
    parser.add_argument(
        "--config",
        choices=["best_zindi", "best_local", "all"],
        default="all",
        help="Model configuration to evaluate",
    )
    args = parser.parse_args()

    configs = ["best_zindi", "best_local"] if args.config == "all" else [args.config]
    summaries = []
    for cfg in configs:
        summaries.append(evaluate_pipeline(cfg))

    df_all = pd.concat(summaries, ignore_index=True)
    print("\n" + "=" * 90)
    print("MASTER EMBARGOED VALIDATION BENCHMARK SUMMARY")
    print("=" * 90)
    print(df_all[df_all["Val Cutoff"] == "MEAN"].to_string(index=False))
    print("=" * 90)
