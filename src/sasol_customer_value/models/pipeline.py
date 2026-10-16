"""Production pipeline orchestrator for target specialists.

Trains Model F, Model N, and Model O across all historical monthly snapshots
and generates verified test predictions for the competition evaluation horizon.
"""

import time

import numpy as np
import pandas as pd

from sasol_customer_value.config import (
    HISTORICAL_SNAPSHOT_CUTOFFS,
    ID_COL,
    MATURE_SNAPSHOT_CUTOFFS,
    PUBLIC_HISTORY_END_EXCLUSIVE,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
    ModelVariant,
)
from sasol_customer_value.data.loader import load_test_customers, load_transactions
from sasol_customer_value.data.snapshots import (
    build_features,
    cached_paths,
    load_stacked_snapshots,
)
from sasol_customer_value.features.hurdles import compute_snapshot_weights
from sasol_customer_value.models.fuel import FuelModel
from sasol_customer_value.models.nonfuel import NonfuelModel
from sasol_customer_value.models.opportunity import OpportunityModel


def train_production_models(
    config_name: ModelVariant, target: str = "all"
) -> dict[str, pd.Series]:
    """Train specialists on all historical snapshots and generate test predictions.

    Args:
        config_name: Model configuration variant ('best_zindi', 'best_local', 'champion_v2', 'champion_v3', 'champion_v4', 'champion_v5').
        target: Target specialists to train ('all', 'F', 'N', 'O').

    Returns:
        Dictionary mapping target names to test prediction Series indexed by customer ID.
    """
    t0 = time.time()
    print("=" * 80)
    print(f"PRODUCTION TRAINING: {config_name.upper()} (Target: {target.upper()})")
    print("=" * 80)

    # 1. Load historical monthly snapshots using unified snapshot loader
    if config_name == "champion_v6":
        cutoffs = MATURE_SNAPSHOT_CUTOFFS
        print(
            f"--> 1. Loading {len(cutoffs)} mature historical training snapshots (2024-10 to 2025-09)..."
        )
    else:
        cutoffs = HISTORICAL_SNAPSHOT_CUTOFFS
        print(
            f"--> 1. Loading all {len(HISTORICAL_SNAPSHOT_CUTOFFS)} historical training snapshots..."
        )
    X_full, y_full = load_stacked_snapshots(cutoffs)
    print(f"       Loaded {len(X_full):,} rows across {X_full.shape[1]} features.")

    # 2. Extract test features
    print("--> 2. Extracting test features as of 2025-12-01...")
    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))
    df_trans = load_transactions()
    X_test = build_features(
        df_trans, cutoff=PUBLIC_HISTORY_END_EXCLUSIVE, customer_ids=pd.Index(test_ids)
    ).reindex(test_ids)

    # Clean missing values using median imputation for best_local (matching baseline local benchmark)
    if config_name == "best_local":
        med_full = X_full.median()
        X_full = X_full.fillna(med_full)
        X_test = X_test.fillna(med_full)

    predictions: dict[str, pd.Series] = {}

    # Target F (Fuel)
    if target in ["all", "F", "fuel"]:
        print("\n--> 3. Training Model F (Fuel Specialist)...")
        t_f = time.time()
        fuel_variant = (
            "best_zindi"
            if config_name in ("1", "champion_triple_crown")
            else config_name
        )
        fuel_model = FuelModel(variant=fuel_variant)
        if config_name == "champion_v3":
            snap_cutoffs = []
            for c in cutoffs:
                xp, _ = cached_paths(c)
                snap_cutoffs.extend([c] * len(pd.read_parquet(xp)))
            weights_f = compute_snapshot_weights(snap_cutoffs, target_month_day=(12, 1))
            fuel_model.fit(
                X_full,
                y_full[TARGET_FUEL].to_numpy(dtype=float),
                sample_weight=weights_f,
            )
        else:
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
        nonfuel_variant = (
            "champion_v5"
            if config_name in ("1", "champion_triple_crown")
            else config_name
        )
        nonfuel_model = NonfuelModel(variant=nonfuel_variant)
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
        opp_variant = (
            "champion_triple_crown"
            if config_name in ("1", "champion_triple_crown")
            else config_name
        )
        opp_model = OpportunityModel(variant=opp_variant)
        opp_model.fit(X_full, y_full[TARGET_OPPORTUNITY].astype(str).to_numpy())
        pred_opp = opp_model.predict(X_test)
        predictions[TARGET_OPPORTUNITY] = pd.Series(pred_opp, index=test_ids)
        print(
            f"       [DONE] Classes: {len(np.unique(pred_opp))}, Top Class: {pd.Series(pred_opp).value_counts().index[0]} ({pd.Series(pred_opp).value_counts().iloc[0]}), Elapsed: {time.time() - t_o:.1f}s"
        )

    print(f"\n[SUCCESS] Production training completed in {time.time() - t0:.1f}s.")
    return predictions
