"""Script to build Version 12: 3-Way Tri-Model Opportunity Ensemble + Seasonal Regression.

Architecture:
1. Continuous Spend:
   - CLV_fuel: sub_v3_seasonal (0.59720 RMSE)
   - CLV_nonfuel: sub_v3_seasonal (0.72776 RMSE)
2. Opportunity Classification:
   - 3-Way Multi-Model Ensemble:
     * LightGBMClassifier (40% weight)
     * XGBClassifier(hist) (40% weight)
     * CatBoostClassifier (20% weight)
   - Bagged across 5 random seeds: [42, 100, 2024, 777, 999]
   - Trained on all 15 monthly historical snapshots (~65,722 rows across 225 features)
   - Calibrated with summer-optimized log-odds class bias vector
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from sasol_customer_value.config import (
    ID_COL,
    PUBLIC_HISTORY_END_EXCLUSIVE,
    SUBMISSIONS_DIR,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.data.loader import load_test_customers, load_transactions
from sasol_customer_value.data.snapshots import build_features, cached_paths
from sasol_customer_value.utils.helpers import validate_submission_format

# Optimal summer-calibrated bias vector discovered in 3-way blend
OPTIMAL_3WAY_BIASES: dict[str, float] = {
    "Existing-category growth: Beverages": 1.1,
    "Existing-category growth: Food & beverage": 0.8,
    "Existing-category growth: Fuel": 0.1,
    "Existing-category growth: Other": 0.6,
    "Inactivity": 0.1,
    "New category adoption: Beverages": 1.0,
    "New category adoption: Braai & Ice": 0.5,
    "New category adoption: Breads & rolls": 1.4,
    "New category adoption: Confectionary": 1.4,
    "New category adoption: Food & beverage": -0.1,
    "New category adoption: Groceries": 1.0,
    "New category adoption: Ice-cream": 1.4,
    "New category adoption: Lubricants": 0.9,
    "New category adoption: Other": 1.4,
    "New category adoption: Snacking": 1.0,
    "New category adoption: Tobacco": 0.8,
    "Stable": 0.0,
}

SEEDS = [42, 100, 2024, 777, 999]


def build_v12_submission(
    output_path: Path = SUBMISSIONS_DIR / "sub_v12_3way_ensemble_opp.csv",
) -> Path:
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("--> 1. Loading continuous spend from sub_v3_seasonal.csv...")
    v3_path = SUBMISSIONS_DIR / "sub_v3_seasonal.csv"
    if not v3_path.exists():
        raise FileNotFoundError(f"Missing {v3_path}")
    df_v3 = pd.read_csv(v3_path)
    df_v3["ID"] = df_v3["ID"].astype(str)

    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))
    df_v3 = df_v3.set_index("ID").reindex(test_ids).reset_index()

    print("--> 2. Loading all 15 monthly historical snapshots...")
    cutoffs = [
        str(d.date()) for d in pd.date_range("2024-07-01", "2025-09-01", freq="MS")
    ]
    X_train = pd.concat(
        [pd.read_parquet(cached_paths(c)[0]) for c in cutoffs], ignore_index=True
    )
    y_train = pd.concat(
        [pd.read_parquet(cached_paths(c)[1]) for c in cutoffs], ignore_index=True
    )
    print(
        f"       Training pool: {len(X_train):,} rows across {X_train.shape[1]} features"
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train[TARGET_OPPORTUNITY].astype(str))
    classes = le.classes_

    print(f"--> 3. Extracting test features as of {PUBLIC_HISTORY_END_EXCLUSIVE}...")
    df_trans = load_transactions()
    X_test = build_features(
        df_trans, cutoff=PUBLIC_HISTORY_END_EXCLUSIVE, customer_ids=pd.Index(test_ids)
    )

    print(f"--> 4. Training 3-Way Opportunity Ensemble across {len(SEEDS)} seeds...")
    seed_probs = []
    for s in SEEDS:
        t_seed = time.time()
        print(f"       Fitting Seed {s}...")
        # 1. LightGBM
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.03,
            colsample_bytree=0.75,
            subsample=0.8,
            subsample_freq=1,
            random_state=s,
            verbose=-1,
        ).fit(X_train, y_train_enc)
        p_lgb = lgb_clf.predict_proba(X_test)

        # 2. XGBoost (histogram mode)
        xgb_clf = XGBClassifier(
            n_estimators=160,
            learning_rate=0.03,
            max_depth=5,
            colsample_bytree=0.75,
            subsample=0.8,
            tree_method="hist",
            random_state=s,
        ).fit(X_train, y_train_enc)
        p_xgb = xgb_clf.predict_proba(X_test)

        # 3. CatBoost (depth=4)
        cb_clf = CatBoostClassifier(
            iterations=120,
            depth=4,
            learning_rate=0.06,
            random_seed=s,
            verbose=0,
        ).fit(X_train, y_train_enc)
        p_cb = cb_clf.predict_proba(X_test)

        p_blend = 0.40 * p_lgb + 0.40 * p_xgb + 0.20 * p_cb
        seed_probs.append(p_blend)
        print(f"       Seed {s} finished in {time.time() - t_seed:.1f}s")

    avg_probs = np.mean(seed_probs, axis=0)

    print("--> 5. Applying summer-calibrated log-odds class biases...")
    bias_vec = np.array([OPTIMAL_3WAY_BIASES.get(c, 0.0) for c in classes], dtype=float)
    log_probs = np.log(np.clip(avg_probs, 1e-9, 1.0)) + bias_vec
    pred_opp = classes[log_probs.argmax(axis=1)].copy()

    # Inactivity consistency check
    if "days_since_last_visit" in X_test.columns:
        fuel_low = df_v3["CLV_fuel"].to_numpy() < 0.10
        nonfuel_low = df_v3["CLV_nonfuel"].to_numpy() < 0.10
        recency_high = X_test["days_since_last_visit"].to_numpy() > 60.0
        inactive_mask = fuel_low & nonfuel_low & recency_high
        pred_opp[inactive_mask] = "Inactivity"

    sub_v12 = pd.DataFrame(
        {
            ID_COL: test_ids,
            "CLV_fuel": df_v3["CLV_fuel"],
            "CLV_nonfuel": df_v3["CLV_nonfuel"],
            "Opportunity": pred_opp,
        }
    )

    print("--> 6. Verifying submission format compliance...")
    validate_submission_format(sub_v12, expected_ids=test_ids)
    sub_v12.to_csv(output_path, index=False)
    print(
        f"[SUCCESS] sub_v12 saved to: {output_path} (Total runtime: {time.time() - t0:.1f}s)"
    )

    print("\nOpportunity Distribution in sub_v12:")
    print(sub_v12["Opportunity"].value_counts())
    return output_path


if __name__ == "__main__":
    build_v12_submission()
