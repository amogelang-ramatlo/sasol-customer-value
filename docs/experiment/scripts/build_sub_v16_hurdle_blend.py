"""Flagship Submission v16: Two-Stage Hurdle Blend + Out-Of-Fold Calibrated Consensus Classifier.

Winning Synthesis:
1. CLV_fuel:
   Trained on all 15 historical snapshots using the optimal convex blend:
   z_fuel = 0.45 * z_direct + 0.55 * z_hurdle
   (Confirmed to beat baseline Direct regressor by -0.00080 RMSE on embargoed validation).
2. CLV_nonfuel:
   Trained on all 15 historical snapshots using the optimal convex blend:
   z_nonfuel = 0.35 * z_direct + 0.65 * z_hurdle
   (Confirmed to beat baseline Direct regressor by -0.00058 RMSE on embargoed validation).
3. Opportunity:
   Consensus Committee (v14/v13 core) with optimal constrained macro-group decision boundaries:
   b_inact = +0.1016, b_fuel = +0.1485, b_store_growth = +0.8611, b_adopt = +0.5305, b_stable = 0.0.

Full Dataset Utilization:
Trained on ALL 15 historical monthly snapshots (65,722 rows) including all validation periods.
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

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
from sasol_customer_value.models.hurdle import BaggedHurdleRegressor
from sasol_customer_value.utils.helpers import validate_submission_format

OPTIMAL_OOF_BIASES: dict[str, float] = {
    "Existing-category growth: Beverages": 0.8611,
    "Existing-category growth: Food & beverage": 0.8611,
    "Existing-category growth: Fuel": 0.1485,
    "Existing-category growth: Other": 0.8611,
    "Inactivity": 0.1016,
    "New category adoption: Beverages": 0.5305,
    "New category adoption: Braai & Ice": 0.5305,
    "New category adoption: Breads & rolls": 0.5305,
    "New category adoption: Confectionary": 0.5305,
    "New category adoption: Food & beverage": 0.5305,
    "New category adoption: Groceries": 0.5305,
    "New category adoption: Ice-cream": 0.5305,
    "New category adoption: Lubricants": 0.5305,
    "New category adoption: Other": 0.5305,
    "New category adoption: Snacking": 0.5305,
    "New category adoption: Tobacco": 0.5305,
    "Stable": 0.0000,
}


from pathlib import Path


def build_sub_v16(output_dir: Path = SUBMISSIONS_DIR) -> Path:
    t0 = time.time()
    print("=" * 80)
    print("BUILDING SUBMISSION V16 (HURDLE BLEND + OOF CALIBRATED CLASSIFIER)")
    print("=" * 80)

    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))
    output_path = output_dir / "sub_v16_hurdle_blend.csv"

    # 1. Load full training data across ALL 15 historical monthly snapshots
    print(
        "--> 1. Loading all 15 historical snapshots (including validation periods)..."
    )
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

    # 2. Extract test features as of 2025-12-01
    print("--> 2. Extracting test features as of 2025-12-01...")
    df_trans = load_transactions()
    X_test = build_features(
        df_trans, cutoff=PUBLIC_HISTORY_END_EXCLUSIVE, customer_ids=pd.Index(test_ids)
    )
    X_test = X_test.reindex(test_ids)

    # 3. Train Fuel Models (Direct + Hurdle Blend)
    print("--> 3. Training Fuel Regressors (45% Direct + 55% Hurdle)...")
    seeds = (42, 100, 2024, 777, 999)
    f_direct_preds = []
    for s in seeds:
        m_cb_f = CatBoostRegressor(
            iterations=250,
            learning_rate=0.04,
            random_seed=s,
            verbose=0,
            thread_count=-1,
        )
        m_cb_f.fit(X_train, y_train[TARGET_FUEL])
        f_direct_preds.append(np.clip(m_cb_f.predict(X_test), 0.0, None))
    pred_fuel_direct = np.mean(f_direct_preds, axis=0)

    hurdle_f = BaggedHurdleRegressor(seeds=seeds)
    hurdle_f.fit(X_train, y_train[TARGET_FUEL])
    pred_fuel_hurdle = hurdle_f.predict(X_test)

    pred_fuel_final = 0.45 * pred_fuel_direct + 0.55 * pred_fuel_hurdle

    # 4. Train Non-Fuel Models (35% Direct + 65% Hurdle Blend)
    print("--> 4. Training Non-Fuel Regressors (35% Direct + 65% Hurdle)...")
    nf_direct_preds = []
    for s in seeds:
        m_cb_nf = CatBoostRegressor(
            iterations=250,
            learning_rate=0.04,
            random_seed=s,
            verbose=0,
            thread_count=-1,
        )
        m_lgb_nf = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.03, random_state=s, n_jobs=-1, verbose=-1
        )
        m_cb_nf.fit(X_train, y_train[TARGET_NONFUEL])
        m_lgb_nf.fit(X_train, y_train[TARGET_NONFUEL])
        p_nf = 0.60 * np.clip(m_cb_nf.predict(X_test), 0.0, None) + 0.40 * np.clip(
            m_lgb_nf.predict(X_test), 0.0, None
        )
        nf_direct_preds.append(p_nf)
    pred_nonfuel_direct = np.mean(nf_direct_preds, axis=0)

    hurdle_nf = BaggedHurdleRegressor(seeds=seeds)
    hurdle_nf.fit(X_train, y_train[TARGET_NONFUEL])
    pred_nonfuel_hurdle = hurdle_nf.predict(X_test)

    pred_nonfuel_final = 0.35 * pred_nonfuel_direct + 0.65 * pred_nonfuel_hurdle

    # 5. Train Opportunity Classifier with OOF Optimal Biases & Consensus
    print("--> 5. Training Opportunity Committee Classifiers across 5 seeds...")
    opp_probs_list = []
    classes = None
    for s in seeds:
        clf = lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.04,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            random_state=s,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_train, y_train[TARGET_OPPORTUNITY].astype(str))
        if classes is None:
            classes = clf.classes_
        opp_probs_list.append(clf.predict_proba(X_test))

    avg_opp_probs = np.mean(opp_probs_list, axis=0)
    bias_vec = np.array([OPTIMAL_OOF_BIASES.get(c, 0.0) for c in classes], dtype=float)

    # Calibrated opportunity predictions from 5-seed averaged probabilities with optimal OOF biases
    log_probs_calib = np.log(np.clip(avg_opp_probs, 1e-9, 1.0)) + bias_vec
    pred_opp_final = classes[log_probs_calib.argmax(axis=1)]

    # 6. Verify and Save Submission
    sub_v16 = pd.DataFrame(
        {
            ID_COL: test_ids,
            "CLV_fuel": pred_fuel_final,
            "CLV_nonfuel": pred_nonfuel_final,
            "Opportunity": pred_opp_final,
        }
    )

    validate_submission_format(sub_v16, expected_ids=test_ids)
    sub_v16.to_csv(output_path, index=False)
    print(
        f"\n[SUCCESS] sub_v16 saved to: {output_path} (Total runtime: {time.time() - t0:.1f}s)"
    )

    print("\nOpportunity Class Distribution in sub_v16:")
    print(sub_v16["Opportunity"].value_counts())

    return output_path


if __name__ == "__main__":
    build_sub_v16()
