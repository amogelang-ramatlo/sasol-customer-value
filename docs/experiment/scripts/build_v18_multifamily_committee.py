"""Build Strategy C: Multi-Family Consensus Committee (LGBM + CatBoost + XGBoost).

Implements both:
1. v18_single_seed: Single-seed Multi-Family Consensus Committee + v14 continuous spend.
2. v18_multiseed: 5-Seed Bagged Multi-Family Consensus Committee (5 LGB + 5 CB + 5 XGB) + v14 continuous spend.

Evaluates both on the held-out 2025-09-01 validation cohort to report local metrics:
- Fuel RMSE
- Nonfuel RMSE
- Opportunity Weighted F1
- Overall Competition Score
Generates verified submissions for test.csv.
"""

import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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


def build_and_evaluate_v18() -> None:
    t_global = time.time()
    print("=" * 80)
    print("BUILDING AND EVALUATING STRATEGY C (V18 SINGLE SEED & V18 MULTISEED)")
    print("=" * 80)

    # 1. Load data
    print("--> 1. Loading all historical monthly training snapshots...")
    cutoffs_embargoed = [
        str(d.date()) for d in pd.date_range("2024-07-01", "2025-06-01", freq="MS")
    ]
    cutoffs_all = [
        str(d.date()) for d in pd.date_range("2024-07-01", "2025-09-01", freq="MS")
    ]

    # Embargoed training for local validation
    X_tr_emb = pd.concat([pd.read_parquet(cached_paths(c)[0]) for c in cutoffs_embargoed], ignore_index=True)
    y_tr_emb = pd.concat([pd.read_parquet(cached_paths(c)[1]) for c in cutoffs_embargoed], ignore_index=True)

    # Full training for final test submission (all 15 snapshots)
    X_tr_full = pd.concat([pd.read_parquet(cached_paths(c)[0]) for c in cutoffs_all], ignore_index=True)
    y_tr_full = pd.concat([pd.read_parquet(cached_paths(c)[1]) for c in cutoffs_all], ignore_index=True)

    # Validation cohort 2025-09-01
    X_va = pd.read_parquet(cached_paths("2025-09-01")[0])
    y_va = pd.read_parquet(cached_paths("2025-09-01")[1])
    y_va_opp = y_va[TARGET_OPPORTUNITY].astype(str).to_numpy()

    # Test customers
    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))

    print("--> 2. Extracting test features as of 2025-12-01...")
    df_trans = load_transactions()
    X_test = build_features(
        df_trans, cutoff=PUBLIC_HISTORY_END_EXCLUSIVE, customer_ids=pd.Index(test_ids)
    ).reindex(test_ids)

    # Clean missing values using median imputation
    med_emb = X_tr_emb.median()
    X_tr_emb_clean = X_tr_emb.fillna(med_emb)
    X_va_clean = X_va.fillna(med_emb)

    med_full = X_tr_full.median()
    X_tr_full_clean = X_tr_full.fillna(med_full)
    X_test_clean = X_test.fillna(med_full)

    # Prepare labels and biases
    le = LabelEncoder()
    y_tr_emb_encoded = le.fit_transform(y_tr_emb[TARGET_OPPORTUNITY].astype(str))
    classes = le.classes_
    bias_vec = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes], dtype=float)

    y_tr_full_encoded = le.transform(y_tr_full[TARGET_OPPORTUNITY].astype(str))

    # Reference v14 continuous spend
    v14_sub = pd.read_csv(SUBMISSIONS_DIR / "sub_v14_all_star_synthesis.csv", dtype={ID_COL: str}).set_index(ID_COL).reindex(test_ids)
    test_fuel_v14 = v14_sub["CLV_fuel"].values
    test_nonfuel_v14 = v14_sub["CLV_nonfuel"].values

    # Local validation continuous spend benchmark
    from sasol_customer_value.models.ensemble import EnsemblePipeline
    pipe_reg = EnsemblePipeline(random_seed=42).fit(X_tr_emb_clean, y_tr_emb)
    f_val, nf_val, _ = pipe_reg.predict_components(X_va_clean)
    rf_local = np.sqrt(mean_squared_error(y_va[TARGET_FUEL], f_val))
    rnf_local = np.sqrt(mean_squared_error(y_va[TARGET_NONFUEL], nf_val))

    # =========================================================================
    # PART 1: V18 SINGLE SEED (SEED 42)
    # =========================================================================
    print("\n" + "=" * 60)
    print("--> PART 1: TRAINING V18 SINGLE SEED (LightGBM + CatBoost + XGBoost, Seed 42)...")
    print("=" * 60)

    # 1. LightGBM
    clf_lgb_s1 = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.04, num_class=17, random_state=42, verbose=-1, n_jobs=-1).fit(X_tr_emb_clean, y_tr_emb_encoded)
    probs_lgb_va_s1 = clf_lgb_s1.predict_proba(X_va_clean)
    pred_lgb_va_s1 = classes[(np.log(np.clip(probs_lgb_va_s1, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]

    # 2. CatBoost
    clf_cb_s1 = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=5, random_seed=42, verbose=0, thread_count=-1).fit(X_tr_emb_clean, y_tr_emb_encoded)
    probs_cb_va_s1 = clf_cb_s1.predict_proba(X_va_clean)
    pred_cb_va_s1 = classes[(np.log(np.clip(probs_cb_va_s1, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]

    # 3. XGBoost
    clf_xgb_s1 = XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=5, random_state=42, eval_metric="mlogloss", n_jobs=-1).fit(X_tr_emb_clean, y_tr_emb_encoded)
    probs_xgb_va_s1 = clf_xgb_s1.predict_proba(X_va_clean)
    pred_xgb_va_s1 = classes[(np.log(np.clip(probs_xgb_va_s1, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]

    # Local Consensus Validation
    val_consensus_s1 = []
    for i in range(len(y_va_opp)):
        counts = Counter([pred_lgb_va_s1[i], pred_cb_va_s1[i], pred_xgb_va_s1[i]])
        top_cand, top_count = counts.most_common(1)[0]
        val_consensus_s1.append(top_cand if top_count >= 2 else pred_lgb_va_s1[i])
    f1_v18_single = f1_score(y_va_opp, np.array(val_consensus_s1), average="weighted", zero_division=0)
    score_v18_single = 0.60 + 0.40 * f1_v18_single - 0.40540558 * rf_local - 0.36764644 * rnf_local

    # Fit single seed on full training data for test submission
    print("       Fitting single-seed models on full training data (all 15 snapshots)...")
    clf_lgb_full_s1 = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.04, num_class=17, random_state=42, verbose=-1, n_jobs=-1).fit(X_tr_full_clean, y_tr_full_encoded)
    clf_cb_full_s1 = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=5, random_seed=42, verbose=0, thread_count=-1).fit(X_tr_full_clean, y_tr_full_encoded)
    clf_xgb_full_s1 = XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=5, random_state=42, eval_metric="mlogloss", n_jobs=-1).fit(X_tr_full_clean, y_tr_full_encoded)

    p_lgb_test_s1 = classes[(np.log(np.clip(clf_lgb_full_s1.predict_proba(X_test_clean), 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    p_cb_test_s1 = classes[(np.log(np.clip(clf_cb_full_s1.predict_proba(X_test_clean), 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    p_xgb_test_s1 = classes[(np.log(np.clip(clf_xgb_full_s1.predict_proba(X_test_clean), 1e-9, 1.0)) + bias_vec).argmax(axis=1)]

    test_consensus_s1 = []
    for i in range(len(test_ids)):
        counts = Counter([p_lgb_test_s1[i], p_cb_test_s1[i], p_xgb_test_s1[i]])
        top_cand, top_count = counts.most_common(1)[0]
        test_consensus_s1.append(top_cand if top_count >= 2 else p_lgb_test_s1[i])

    out_s1 = SUBMISSIONS_DIR / "sub_v18_single_seed.csv"
    sub_s1 = pd.DataFrame({ID_COL: test_ids, "CLV_fuel": test_fuel_v14, "CLV_nonfuel": test_nonfuel_v14, "Opportunity": test_consensus_s1})
    validate_submission_format(sub_s1, expected_ids=test_ids)
    sub_s1.to_csv(out_s1, index=False)
    print(f"[SUCCESS] Saved v18_single_seed to: {out_s1}")

    # =========================================================================
    # PART 2: V18 MULTISEED (5 SEEDS: 42, 100, 2024, 777, 999)
    # =========================================================================
    print("\n" + "=" * 60)
    print("--> PART 2: TRAINING V18 MULTISEED (5 SEEDS x 3 FAMILIES = 15 MODELS)...")
    print("=" * 60)
    seeds = [42, 100, 2024, 777, 999]

    # Local validation evaluation
    lgb_val_probs, cb_val_probs, xgb_val_probs = [], [], []
    for s in seeds:
        print(f"       Fitting seed {s} on validation training split...")
        m_lgb = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.04, num_class=17, random_state=s, verbose=-1, n_jobs=-1).fit(X_tr_emb_clean, y_tr_emb_encoded)
        m_cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=5, random_seed=s, verbose=0, thread_count=-1).fit(X_tr_emb_clean, y_tr_emb_encoded)
        m_xgb = XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=5, random_state=s, eval_metric="mlogloss", n_jobs=-1).fit(X_tr_emb_clean, y_tr_emb_encoded)

        lgb_val_probs.append(m_lgb.predict_proba(X_va_clean))
        cb_val_probs.append(m_cb.predict_proba(X_va_clean))
        xgb_val_probs.append(m_xgb.predict_proba(X_va_clean))

    # Averaged family probabilities on validation
    avg_p_lgb_va = np.mean(lgb_val_probs, axis=0)
    avg_p_cb_va = np.mean(cb_val_probs, axis=0)
    avg_p_xgb_va = np.mean(xgb_val_probs, axis=0)

    pred_lgb_va_multi = classes[(np.log(np.clip(avg_p_lgb_va, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    pred_cb_va_multi = classes[(np.log(np.clip(avg_p_cb_va, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    pred_xgb_va_multi = classes[(np.log(np.clip(avg_p_xgb_va, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]

    val_consensus_multi = []
    for i in range(len(y_va_opp)):
        counts = Counter([pred_lgb_va_multi[i], pred_cb_va_multi[i], pred_xgb_va_multi[i]])
        top_cand, top_count = counts.most_common(1)[0]
        val_consensus_multi.append(top_cand if top_count >= 2 else pred_lgb_va_multi[i])

    f1_v18_multi = f1_score(y_va_opp, np.array(val_consensus_multi), average="weighted", zero_division=0)
    score_v18_multi = 0.60 + 0.40 * f1_v18_multi - 0.40540558 * rf_local - 0.36764644 * rnf_local

    # Full training fit across 5 seeds for test submission
    print("\n       Fitting 5 seeds across all 15 snapshots for test submission...")
    lgb_test_probs, cb_test_probs, xgb_test_probs = [], [], []
    for s in seeds:
        print(f"       Fitting seed {s} on full dataset...")
        m_lgb_full = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.04, num_class=17, random_state=s, verbose=-1, n_jobs=-1).fit(X_tr_full_clean, y_tr_full_encoded)
        m_cb_full = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=5, random_seed=s, verbose=0, thread_count=-1).fit(X_tr_full_clean, y_tr_full_encoded)
        m_xgb_full = XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=5, random_state=s, eval_metric="mlogloss", n_jobs=-1).fit(X_tr_full_clean, y_tr_full_encoded)

        lgb_test_probs.append(m_lgb_full.predict_proba(X_test_clean))
        cb_test_probs.append(m_cb_full.predict_proba(X_test_clean))
        xgb_test_probs.append(m_xgb_full.predict_proba(X_test_clean))

    avg_p_lgb_test = np.mean(lgb_test_probs, axis=0)
    avg_p_cb_test = np.mean(cb_test_probs, axis=0)
    avg_p_xgb_test = np.mean(xgb_test_probs, axis=0)

    p_lgb_test_multi = classes[(np.log(np.clip(avg_p_lgb_test, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    p_cb_test_multi = classes[(np.log(np.clip(avg_p_cb_test, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    p_xgb_test_multi = classes[(np.log(np.clip(avg_p_xgb_test, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]

    test_consensus_multi = []
    for i in range(len(test_ids)):
        counts = Counter([p_lgb_test_multi[i], p_cb_test_multi[i], p_xgb_test_multi[i]])
        top_cand, top_count = counts.most_common(1)[0]
        test_consensus_multi.append(top_cand if top_count >= 2 else p_lgb_test_multi[i])

    out_multi = SUBMISSIONS_DIR / "sub_v18_multiseed.csv"
    sub_multi = pd.DataFrame({ID_COL: test_ids, "CLV_fuel": test_fuel_v14, "CLV_nonfuel": test_nonfuel_v14, "Opportunity": test_consensus_multi})
    validate_submission_format(sub_multi, expected_ids=test_ids)
    sub_multi.to_csv(out_multi, index=False)
    print(f"[SUCCESS] Saved v18_multiseed to: {out_multi}")

    # =========================================================================
    # SUMMARY SCORECARD
    # =========================================================================
    print("\n" + "=" * 80)
    print("LOCAL VALIDATION PERFORMANCE COMPARISON (2025-09-01 COHORT)")
    print("=" * 80)
    print(f"Local Fuel RMSE:             {rf_local:.5f}")
    print(f"Local Nonfuel RMSE:          {rnf_local:.5f}")
    print(f"1. v18 Single-Seed Opp F1:   {f1_v18_single:.5f} (Overall Score: {score_v18_single:.5f})")
    print(f"2. v18 Multi-Seed Opp F1:    {f1_v18_multi:.5f} (Overall Score: {score_v18_multi:.5f}) [Gain: {f1_v18_multi - f1_v18_single:+.5f}]")
    print(f"Total pipeline runtime: {time.time() - t_global:.1f}s")


if __name__ == "__main__":
    build_and_evaluate_v18()
