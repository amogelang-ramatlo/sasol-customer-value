"""Standalone reproduction script for Version 10 (No-LightGBM Ensemble).

Architecture:
1. Continuous CLV_fuel: CatBoost (0.70) + XGBoost (0.30)
2. Continuous CLV_nonfuel: CatBoost (0.70) + XGBoost (0.30)
3. Opportunity: XGBClassifier(tree_method='hist') + Summer-Calibrated Log-Odds Biases
4. Bagging: 5 random seeds [42, 100, 2024, 777, 999] across all 15 monthly snapshots
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from sasol_customer_value.config import (
    ID_COL,
    PUBLIC_HISTORY_END_EXCLUSIVE,
    SUBMISSIONS_DIR,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.data.loader import load_test_customers, load_transactions
from sasol_customer_value.data.snapshots import build_features, cached_paths
from sasol_customer_value.models.ensemble import DEFAULT_OPP_BIASES
from sasol_customer_value.utils.helpers import validate_submission_format


def build_v10_submission(output_path: Path = SUBMISSIONS_DIR / "sub_v10_no_lightgbm.csv") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("--> Loading all 15 monthly snapshots for training...")
    cutoffs = [str(d.date()) for d in pd.date_range("2024-07-01", "2025-09-01", freq="MS")]
    X_full = pd.concat([pd.read_parquet(cached_paths(c)[0]) for c in cutoffs], ignore_index=True)
    y_full = pd.concat([pd.read_parquet(cached_paths(c)[1]) for c in cutoffs], ignore_index=True)
    print(f"    Training pool: {len(X_full):,} rows across {X_full.shape[1]} features")

    print(f"--> Extracting test features as of {PUBLIC_HISTORY_END_EXCLUSIVE}...")
    df_trans = load_transactions()
    df_test = load_test_customers()
    test_ids = pd.Index(df_test[ID_COL].astype(str))
    X_test = build_features(df_trans, cutoff=PUBLIC_HISTORY_END_EXCLUSIVE, customer_ids=test_ids)

    le = LabelEncoder()
    y_opp_enc = le.fit_transform(y_full[TARGET_OPPORTUNITY].astype(str))
    classes = le.classes_

    seeds = [42, 100, 2024, 777, 999]
    fuel_list, nonfuel_list, probs_list = [], [], []

    for s in seeds:
        print(f"    Fitting No-LightGBM models with seed {s}...")
        # 1. Fuel Regressors (CatBoost 0.70 + XGBoost 0.30)
        cb_f = CatBoostRegressor(random_seed=s, iterations=250, learning_rate=0.04, verbose=0).fit(
            X_full, y_full[TARGET_FUEL]
        )
        xgb_f = XGBRegressor(random_state=s, n_estimators=150, learning_rate=0.03, max_depth=5).fit(
            X_full, y_full[TARGET_FUEL]
        )
        p_cb_f = np.clip(cb_f.predict(X_test), 0.0, None)
        p_xgb_f = np.clip(xgb_f.predict(X_test), 0.0, None)
        fuel_list.append(0.70 * p_cb_f + 0.30 * p_xgb_f)

        # 2. Non-fuel Regressors (CatBoost 0.70 + XGBoost 0.30)
        cb_nf = CatBoostRegressor(random_seed=s, iterations=250, learning_rate=0.04, verbose=0).fit(
            X_full, y_full[TARGET_NONFUEL]
        )
        xgb_nf = XGBRegressor(random_state=s, n_estimators=150, learning_rate=0.03, max_depth=5).fit(
            X_full, y_full[TARGET_NONFUEL]
        )
        p_cb_nf = np.clip(cb_nf.predict(X_test), 0.0, None)
        p_xgb_nf = np.clip(xgb_nf.predict(X_test), 0.0, None)
        nonfuel_list.append(0.70 * p_cb_nf + 0.30 * p_xgb_nf)

        # 3. Opportunity Classifier (XGBClassifier)
        xgb_clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.06,
            max_depth=5,
            tree_method="hist",
            random_state=s,
        )
        xgb_clf.fit(X_full, y_opp_enc)
        probs_list.append(xgb_clf.predict_proba(X_test))

    avg_fuel = np.mean(fuel_list, axis=0)
    avg_nonfuel = np.mean(nonfuel_list, axis=0)
    avg_probs = np.mean(probs_list, axis=0)

    # Opportunity log-odds calibration
    bias_vec = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes], dtype=float)
    log_probs = np.log(np.clip(avg_probs, 1e-9, 1.0)) + bias_vec
    pred_opp = classes[log_probs.argmax(axis=1)].copy()

    if "days_since_last_visit" in X_test.columns:
        inactive_mask = (avg_fuel < 0.10) & (avg_nonfuel < 0.10) & (X_test["days_since_last_visit"] > 60.0)
        pred_opp[inactive_mask] = "Inactivity"

    out_df = pd.DataFrame(
        {
            ID_COL: X_test.index.astype(str),
            TARGET_FUEL: avg_fuel,
            TARGET_NONFUEL: avg_nonfuel,
            TARGET_OPPORTUNITY: pred_opp,
        }
    ).set_index(ID_COL).reindex(test_ids).reset_index()

    validate_submission_format(out_df, expected_ids=list(test_ids))
    out_df.to_csv(output_path, index=False)
    print(f"[REPRODUCED] Successfully saved: {output_path}")
    return output_path


if __name__ == "__main__":
    build_v10_submission()
