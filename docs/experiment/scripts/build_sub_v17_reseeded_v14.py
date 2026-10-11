"""Run exact sub_v14 architecture with a new distinct seed bank.

Architecture of v14:
1. Model A (corresponds to v4):
   - Single seed (originally 42 -> new seed 101)
   - Features: 202 features (first 202 columns of X)
   - Regressors: Blend weights (Fuel: 0.50 LGB / 0.30 CB / 0.20 XGB; Nonfuel: 0.50 LGB / 0.30 CB / 0.20 XGB)
   - Classifier: LightGBM (120 trees, lr=0.04) + DEFAULT_OPP_BIASES

2. Model B (corresponds to v8):
   - 5-seed bagged ensemble (originally [42, 100, 2024, 777, 999] -> new seeds [101, 202, 303, 404, 505])
   - Features: 218 features (first 218 columns of X)
   - Regressors: Blend weights (Fuel: 0.20 LGB / 0.55 CB / 0.25 XGB; Nonfuel: 0.40 LGB / 0.60 CB)
   - Classifier: 5-seed averaged LightGBM + DEFAULT_OPP_BIASES

3. Model C (corresponds to v9):
   - 5-seed bagged ensemble (new seeds [101, 202, 303, 404, 505])
   - Features: All 225 features
   - Regressors: Blend weights (Fuel: 0.20 LGB / 0.55 CB / 0.25 XGB; Nonfuel: 0.40 LGB / 0.60 CB)
   - Classifier: 5-seed averaged LightGBM + DEFAULT_OPP_BIASES

4. Target Synthesis (identical to v14/v13):
   - CLV_fuel: Model A
   - CLV_nonfuel: Model B
   - Opportunity: Majority committee consensus among (Model A, Model B, Model C), defaulting to Model B on 3-way tie.
"""

import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

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


def train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    seed: int,
    reg_weights_fuel: tuple[float, float, float],
    reg_weights_nf: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train single-seed regressors and classifier, returning test predictions."""
    y_f = y_train[TARGET_FUEL].to_numpy(dtype=float)
    y_nf = y_train[TARGET_NONFUEL].to_numpy(dtype=float)
    y_opp = y_train[TARGET_OPPORTUNITY].astype(str).to_numpy()

    # Fuel models
    lgb_f = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.03, random_state=seed, verbose=-1, n_jobs=-1
    ).fit(X_train, y_f)
    cb_f = CatBoostRegressor(
        iterations=250, learning_rate=0.04, random_seed=seed, verbose=0, thread_count=-1
    ).fit(X_train, y_f)
    xgb_f = XGBRegressor(
        n_estimators=150, learning_rate=0.03, max_depth=5, random_state=seed, n_jobs=-1
    ).fit(X_train, y_f)

    p_lgb_f = np.clip(lgb_f.predict(X_test), 0.0, None)
    p_cb_f = np.clip(cb_f.predict(X_test), 0.0, None)
    p_xgb_f = np.clip(xgb_f.predict(X_test), 0.0, None)
    pred_fuel = (
        reg_weights_fuel[0] * p_lgb_f
        + reg_weights_fuel[1] * p_cb_f
        + reg_weights_fuel[2] * p_xgb_f
    )

    # Nonfuel models
    lgb_nf = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.03, random_state=seed, verbose=-1, n_jobs=-1
    ).fit(X_train, y_nf)
    cb_nf = CatBoostRegressor(
        iterations=250, learning_rate=0.04, random_seed=seed, verbose=0, thread_count=-1
    ).fit(X_train, y_nf)

    p_lgb_nf = np.clip(lgb_nf.predict(X_test), 0.0, None)
    p_cb_nf = np.clip(cb_nf.predict(X_test), 0.0, None)
    pred_nonfuel = reg_weights_nf[0] * p_lgb_nf + reg_weights_nf[1] * p_cb_nf

    # Opportunity model
    clf = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.04,
        num_class=17,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    ).fit(X_train, y_opp)

    probs = clf.predict_proba(X_test)
    classes = clf.classes_

    return pred_fuel, pred_nonfuel, probs, classes


def run_v14_with_new_seeds(
    single_seed: int = 101,
    multi_seeds: list[int] | None = None,
    output_path: Path = SUBMISSIONS_DIR / "sub_v17_v14_reseeded.csv",
) -> Path:
    if multi_seeds is None:
        multi_seeds = [101, 202, 303, 404, 505]

    t0 = time.time()
    print("=" * 80)
    print("RERUNNING V14 ARCHITECTURE WITH NEW DISTINCT SEEDS")
    print(f"Single Seed (v4 counterpart): {single_seed}")
    print(f"Multi Seeds  (v8 & v9 counterpart): {multi_seeds}")
    print("=" * 80)

    # 1. Load data
    print("--> 1. Loading all 15 historical monthly snapshots...")
    cutoffs = [
        str(d.date()) for d in pd.date_range("2024-07-01", "2025-09-01", freq="MS")
    ]
    X_full = pd.concat(
        [pd.read_parquet(cached_paths(c)[0]) for c in cutoffs], ignore_index=True
    )
    y_full = pd.concat(
        [pd.read_parquet(cached_paths(c)[1]) for c in cutoffs], ignore_index=True
    )

    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))

    print("--> 2. Extracting test features as of 2025-12-01...")
    df_trans = load_transactions()
    X_test_full = build_features(
        df_trans, cutoff=PUBLIC_HISTORY_END_EXCLUSIVE, customer_ids=pd.Index(test_ids)
    ).reindex(test_ids)

    cols_202 = list(X_full.columns[:202])
    cols_218 = list(X_full.columns[:218])
    cols_225 = list(X_full.columns[:225])

    # ---------------- MODEL A (v4 Counterpart) ----------------
    print(f"\n--> 3. Training Model A (202 features, Seed={single_seed})...")
    # v4 weights: Fuel (0.50 LGB / 0.30 CB / 0.20 XGB), Nonfuel (0.50 LGB / 0.50 CB)
    f_pred_A, _nf_pred_A, probs_A, classes_A = train_single_model(
        X_full[cols_202],
        y_full,
        X_test_full[cols_202],
        seed=single_seed,
        reg_weights_fuel=(0.50, 0.30, 0.20),
        reg_weights_nf=(0.50, 0.50),
    )
    bias_vec_A = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes_A], dtype=float)
    opp_pred_A = classes_A[(np.log(np.clip(probs_A, 1e-9, 1.0)) + bias_vec_A).argmax(axis=1)]

    # ---------------- MODEL B (v8 Counterpart) ----------------
    print(f"\n--> 4. Training Model B (218 features, Seeds={multi_seeds})...")
    # v8 weights: Fuel (0.20 LGB / 0.55 CB / 0.25 XGB), Nonfuel (0.40 LGB / 0.60 CB)
    nf_list_B = []
    probs_list_B = []
    classes_B = None
    for s in multi_seeds:
        print(f"       Fitting seed {s}...")
        _, p_nf, probs, cls = train_single_model(
            X_full[cols_218],
            y_full,
            X_test_full[cols_218],
            seed=s,
            reg_weights_fuel=(0.20, 0.55, 0.25),
            reg_weights_nf=(0.40, 0.60),
        )
        nf_list_B.append(p_nf)
        probs_list_B.append(probs)
        if classes_B is None:
            classes_B = cls

    nf_pred_B = np.mean(nf_list_B, axis=0)
    avg_probs_B = np.mean(probs_list_B, axis=0)
    bias_vec_B = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes_B], dtype=float)
    opp_pred_B = classes_B[(np.log(np.clip(avg_probs_B, 1e-9, 1.0)) + bias_vec_B).argmax(axis=1)]

    # ---------------- MODEL C (v9 Counterpart) ----------------
    print(f"\n--> 5. Training Model C (225 features, Seeds={multi_seeds})...")
    probs_list_C = []
    classes_C = None
    for s in multi_seeds:
        print(f"       Fitting seed {s}...")
        _, _, probs, cls = train_single_model(
            X_full[cols_225],
            y_full,
            X_test_full[cols_225],
            seed=s,
            reg_weights_fuel=(0.20, 0.55, 0.25),
            reg_weights_nf=(0.40, 0.60),
        )
        probs_list_C.append(probs)
        if classes_C is None:
            classes_C = cls

    avg_probs_C = np.mean(probs_list_C, axis=0)
    bias_vec_C = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes_C], dtype=float)
    opp_pred_C = classes_C[(np.log(np.clip(avg_probs_C, 1e-9, 1.0)) + bias_vec_C).argmax(axis=1)]

    # ---------------- CONSENSUS COMMITTEE (v13 Logic) ----------------
    print("\n--> 6. Forming Committee Consensus (Majority Vote)...")
    consensus_opp = []
    for i in range(len(test_ids)):
        cA = opp_pred_A[i]
        cB = opp_pred_B[i]
        cC = opp_pred_C[i]
        counts = Counter([cA, cB, cC])
        top_cand, top_count = counts.most_common(1)[0]
        consensus_opp.append(top_cand if top_count >= 2 else cB)

    # Inactivity consistency check
    if "days_since_last_visit" in X_test_full.columns:
        fuel_low = f_pred_A < 0.10
        nonfuel_low = nf_pred_B < 0.10
        recency_high = X_test_full["days_since_last_visit"].to_numpy() > 60.0
        inactive_mask = fuel_low & nonfuel_low & recency_high
        for i in np.where(inactive_mask)[0]:
            consensus_opp[i] = "Inactivity"

    # ---------------- ASSEMBLE SUBMISSION (v14 Logic) ----------------
    print("\n--> 7. Assembling final All-Star submission...")
    sub_v17 = pd.DataFrame(
        {
            ID_COL: test_ids,
            "CLV_fuel": f_pred_A,
            "CLV_nonfuel": nf_pred_B,
            "Opportunity": consensus_opp,
        }
    )

    validate_submission_format(sub_v17, expected_ids=test_ids)
    sub_v17.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved reseeded v14 submission to: {output_path}")
    print(f"Total time elapsed: {time.time() - t0:.1f}s")

    print("\nOpportunity Distribution in Reseeded Submission:")
    print(sub_v17["Opportunity"].value_counts())

    return output_path


if __name__ == "__main__":
    run_v14_with_new_seeds()
