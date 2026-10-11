"""Experiment C: Single-Seed Multi-Family Consensus Committee (LightGBM + CatBoost + XGBoost).

Protocol:
1. Train on embargoed historical snapshots (2024-07-01 to 2025-06-01, 50,280 rows).
2. Evaluate strictly on held-out 2025-09-01 validation cohort (5,238 customers).
3. Evaluate:
   - LightGBM alone (v14 baseline counterpart)
   - CatBoost alone
   - XGBoost alone
   - Discrete Committee Consensus Majority Vote (LGB + CB + XGB)
   - Local Weighted F1, Fuel RMSE, Nonfuel RMSE, and Overall Competition Score
"""

import time
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from sasol_customer_value.config import (
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.data.snapshots import cached_paths
from sasol_customer_value.models.ensemble import DEFAULT_OPP_BIASES


def run_experiment_c(seed: int = 42) -> dict:
    t0 = time.time()
    print("=" * 80)
    print("EXPERIMENT C: MULTI-FAMILY CONSENSUS COMMITTEE (SINGLE SEED)")
    print(f"Seed: {seed}")
    print("=" * 80)

    # 1. Load data
    print("--> 1. Loading embargoed historical snapshots...")
    cutoffs_train = [
        str(d.date()) for d in pd.date_range("2024-07-01", "2025-06-01", freq="MS")
    ]
    X_tr = pd.concat(
        [pd.read_parquet(cached_paths(c)[0]) for c in cutoffs_train], ignore_index=True
    )
    y_tr = pd.concat(
        [pd.read_parquet(cached_paths(c)[1]) for c in cutoffs_train], ignore_index=True
    )

    X_va = pd.read_parquet(cached_paths("2025-09-01")[0])
    y_va = pd.read_parquet(cached_paths("2025-09-01")[1])

    y_va_opp = y_va[TARGET_OPPORTUNITY].astype(str).to_numpy()
    y_tr_opp = y_tr[TARGET_OPPORTUNITY].astype(str).to_numpy()

    # Fixed label encoding for models requiring integer targets (XGBoost)
    le = LabelEncoder()
    y_tr_opp_encoded = le.fit_transform(y_tr_opp)
    classes = le.classes_
    bias_vec = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes], dtype=float)

    # Clean feature matrices: fillna with median for stability
    med = X_tr.median()
    X_tr_clean = X_tr.fillna(med)
    X_va_clean = X_va.fillna(med)

    # 2. Model 1: LightGBM Classifier
    print("\n--> 2. Training Model 1: LightGBM Classifier...")
    t_start = time.time()
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.04,
        num_class=17,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    ).fit(X_tr_clean, y_tr_opp_encoded)
    probs_lgb = clf_lgb.predict_proba(X_va_clean)
    pred_lgb = classes[(np.log(np.clip(probs_lgb, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    f1_lgb = f1_score(y_va_opp, pred_lgb, average="weighted", zero_division=0)
    print(f"       LGB trained in {time.time() - t_start:.1f}s | Local F1: {f1_lgb:.5f}")

    # 3. Model 2: CatBoost Classifier
    print("\n--> 3. Training Model 2: CatBoost Classifier...")
    t_start = time.time()
    clf_cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=5,
        random_seed=seed,
        verbose=0,
        thread_count=-1,
    ).fit(X_tr_clean, y_tr_opp_encoded)
    probs_cb = clf_cb.predict_proba(X_va_clean)
    pred_cb = classes[(np.log(np.clip(probs_cb, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    f1_cb = f1_score(y_va_opp, pred_cb, average="weighted", zero_division=0)
    print(f"       CatBoost trained in {time.time() - t_start:.1f}s | Local F1: {f1_cb:.5f}")

    # 4. Model 3: XGBoost Classifier
    print("\n--> 4. Training Model 3: XGBoost Classifier...")
    t_start = time.time()
    clf_xgb = XGBClassifier(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=5,
        random_state=seed,
        eval_metric="mlogloss",
        n_jobs=-1,
    ).fit(X_tr_clean, y_tr_opp_encoded)
    probs_xgb = clf_xgb.predict_proba(X_va_clean)
    pred_xgb = classes[(np.log(np.clip(probs_xgb, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    f1_xgb = f1_score(y_va_opp, pred_xgb, average="weighted", zero_division=0)
    print(f"       XGBoost trained in {time.time() - t_start:.1f}s | Local F1: {f1_xgb:.5f}")

    # 5. Multi-Family Consensus Committee (Majority Vote)
    print("\n--> 5. Computing Multi-Family Committee Majority Vote (LGB + CB + XGB)...")
    consensus_preds = []
    for i in range(len(y_va_opp)):
        cL = pred_lgb[i]
        cC = pred_cb[i]
        cX = pred_xgb[i]
        counts = Counter([cL, cC, cX])
        top_cand, top_count = counts.most_common(1)[0]
        # If there is a 2-vote majority, take it. If 3-way tie, default to LightGBM
        consensus_preds.append(top_cand if top_count >= 2 else cL)

    pred_committee = np.array(consensus_preds)
    f1_committee = f1_score(y_va_opp, pred_committee, average="weighted", zero_division=0)

    # 6. Also test Soft Probability Ensemble (Mean Log-Odds)
    log_probs_blend = (
        np.log(np.clip(probs_lgb, 1e-9, 1.0))
        + np.log(np.clip(probs_cb, 1e-9, 1.0))
        + np.log(np.clip(probs_xgb, 1e-9, 1.0))
    ) / 3.0 + bias_vec
    pred_soft = classes[log_probs_blend.argmax(axis=1)]
    f1_soft = f1_score(y_va_opp, pred_soft, average="weighted", zero_division=0)

    # Load baseline v14 regressor predictions on validation cohort for overall composite score
    from sasol_customer_value.models.ensemble import EnsemblePipeline

    pipe_reg = EnsemblePipeline(random_seed=seed).fit(X_tr_clean, y_tr)
    f_pred, nf_pred, _ = pipe_reg.predict_components(X_va_clean)

    rf = np.sqrt(mean_squared_error(y_va[TARGET_FUEL], f_pred))
    rnf = np.sqrt(mean_squared_error(y_va[TARGET_NONFUEL], nf_pred))

    score_base = 0.60 + 0.40 * f1_lgb - 0.40540558 * rf - 0.36764644 * rnf
    score_committee = 0.60 + 0.40 * f1_committee - 0.40540558 * rf - 0.36764644 * rnf
    score_soft = 0.60 + 0.40 * f1_soft - 0.40540558 * rf - 0.36764644 * rnf

    print("\n" + "=" * 80)
    print("LOCAL VALIDATION RESULTS (2025-09-01 COHORT)")
    print("=" * 80)
    print(f"Fuel RMSE:                  {rf:.5f}")
    print(f"Nonfuel RMSE:               {rnf:.5f}")
    print(f"1. Single LightGBM F1:      {f1_lgb:.5f} (Overall: {score_base:.5f})")
    print(f"2. Single CatBoost F1:      {f1_cb:.5f}")
    print(f"3. Single XGBoost F1:       {f1_xgb:.5f}")
    print(f"4. Committee Consensus F1:  {f1_committee:.5f} (Overall: {score_committee:.5f}) [Delta: {f1_committee - f1_lgb:+.5f}]")
    print(f"5. Soft Probability F1:     {f1_soft:.5f} (Overall: {score_soft:.5f}) [Delta: {f1_soft - f1_lgb:+.5f}]")
    print(f"Total experiment time: {time.time() - t0:.1f}s")

    return {
        "f1_lgb": f1_lgb,
        "f1_cb": f1_cb,
        "f1_xgb": f1_xgb,
        "f1_committee": f1_committee,
        "f1_soft": f1_soft,
        "rf": rf,
        "rnf": rnf,
        "score_committee": score_committee,
    }


if __name__ == "__main__":
    run_experiment_c(seed=42)
