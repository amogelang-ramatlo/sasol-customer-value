"""Production Training & Refitting CLI.

Trains Target Specialists (F, N, O) on all 15 historical monthly snapshots (65k rows)
and generates verified test predictions for the Dec 2025 - Feb 2026 test horizon.
"""

import argparse
import time

import numpy as np
import pandas as pd

from sasol_customer_value.config import (
    ID_COL,
    PUBLIC_HISTORY_END_EXCLUSIVE,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
    ModelVariant,
)
from sasol_customer_value.data.loader import load_test_customers, load_transactions
from sasol_customer_value.data.snapshots import build_features, cached_paths
from sasol_customer_value.models.fuel import FuelModel
from sasol_customer_value.models.nonfuel import NonfuelModel
from sasol_customer_value.models.opportunity import OpportunityModel


def train_production_models(
    config_name: ModelVariant, target: str = "all"
) -> dict[str, pd.Series]:
    """Train on all 15 historical snapshots and generate test predictions."""
    t0 = time.time()
    print("=" * 80)
    print(f"PRODUCTION TRAINING: {config_name.upper()} (Target: {target.upper()})")
    print("=" * 80)

    # 1. Load all 15 historical monthly snapshots
    print("--> 1. Loading all 15 historical training snapshots...")
    cutoffs = [
        str(d.date()) for d in pd.date_range("2024-07-01", "2025-09-01", freq="MS")
    ]
    X_full = pd.concat(
        [pd.read_parquet(cached_paths(c)[0]) for c in cutoffs], ignore_index=True
    )
    y_full = pd.concat(
        [pd.read_parquet(cached_paths(c)[1]) for c in cutoffs], ignore_index=True
    )
    print(f"       Loaded {len(X_full):,} rows across {X_full.shape[1]} features.")

    # 2. Extract test features
    print("--> 2. Extracting test features as of 2025-12-01...")
    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))
    df_trans = load_transactions()
    X_test = build_features(
        df_trans, cutoff=PUBLIC_HISTORY_END_EXCLUSIVE, customer_ids=pd.Index(test_ids)
    ).reindex(test_ids)

    # Clean missing values using median imputation for best_local (matching original single-seed build)
    if config_name == "best_local":
        med_full = X_full.median()
        X_full = X_full.fillna(med_full)
        X_test = X_test.fillna(med_full)

    predictions: dict[str, pd.Series] = {}

    # Target F (Fuel)
    if target in ["all", "F", "fuel"]:
        print("\n--> 3. Training Model F (Fuel Specialist)...")
        t_f = time.time()
        fuel_model = FuelModel(variant=config_name)
        fuel_model.fit(X_full, y_full[TARGET_FUEL].to_numpy(dtype=float))
        pred_f = fuel_model.predict(X_test)
        predictions[TARGET_FUEL] = pd.Series(pred_f, index=test_ids)
        print(
            f"       [DONE] Mean Fuel: {pred_f.mean():.4f} (std: {pred_f.std():.4f}, Elapsed: {time.time() - t_f:.1f}s)"
        )

    # Target N (Non-fuel)
    if target in ["all", "N", "nonfuel"]:
        print("\n--> 4. Training Model N (Non-fuel Specialist)...")
        t_nf = time.time()
        nonfuel_model = NonfuelModel(variant=config_name)
        nonfuel_model.fit(X_full, y_full[TARGET_NONFUEL].to_numpy(dtype=float))
        pred_nf = nonfuel_model.predict(X_test)
        predictions[TARGET_NONFUEL] = pd.Series(pred_nf, index=test_ids)
        print(
            f"       [DONE] Mean Non-fuel: {pred_nf.mean():.4f} (std: {pred_nf.std():.4f}, Elapsed: {time.time() - t_nf:.1f}s)"
        )

    # Target O (Opportunity)
    if target in ["all", "O", "opportunity"]:
        print("\n--> 5. Training Model O (Opportunity Specialist)...")
        t_o = time.time()
        opp_model = OpportunityModel(variant=config_name)
        opp_model.fit(X_full, y_full[TARGET_OPPORTUNITY].astype(str).to_numpy())
        pred_opp = opp_model.predict(X_test)
        predictions[TARGET_OPPORTUNITY] = pd.Series(pred_opp, index=test_ids)
        print(
            f"       [DONE] Classes: {len(np.unique(pred_opp))}, Top Class: {pd.Series(pred_opp).value_counts().index[0]} ({pd.Series(pred_opp).value_counts().iloc[0]}), Elapsed: {time.time() - t_o:.1f}s"
        )

    print(f"\n[SUCCESS] Production training completed in {time.time() - t0:.1f}s.")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train production models on full historical data"
    )
    parser.add_argument(
        "--config",
        choices=["best_zindi", "best_local", "champion_v2"],
        default="champion_v2",
        help="Configuration to train",
    )
    parser.add_argument(
        "--target",
        choices=["all", "F", "N", "O"],
        default="all",
        help="Target to train",
    )
    args = parser.parse_args()

    train_production_models(config_name=args.config, target=args.target)
