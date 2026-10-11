"""Experiment B: Two-Stage Hierarchical Classification.

Hierarchy:
Stage 1 (Binary): Inactivity vs Active (Stable + Fuel Growth + Store Progression).
Stage 2 (Multiclass among Active): Predict remaining 16 active classes.

Final probability assignment:
- P(Inactivity) = P_stage1(Inactivity)
- P(class_k) = P_stage1(Active) * P_stage2(class_k | Active)
Apply DEFAULT_OPP_BIASES and evaluate Weighted F1 against flat baseline.
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from sasol_customer_value.config import (
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.data.snapshots import cached_paths
from sasol_customer_value.models.ensemble import DEFAULT_OPP_BIASES


def run_experiment_b(seed: int = 42) -> dict:
    t0 = time.time()
    print("=" * 80)
    print("EXPERIMENT B: TWO-STAGE HIERARCHICAL CLASSIFICATION (SINGLE SEED)")
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

    med = X_tr.median()
    X_tr_clean = X_tr.fillna(med)
    X_va_clean = X_va.fillna(med)

    # Baseline flat model for direct comparison
    le_all = LabelEncoder()
    y_tr_all = le_all.fit_transform(y_tr_opp)
    classes_all = le_all.classes_
    bias_all = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes_all], dtype=float)

    clf_flat = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.04,
        num_class=17,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    ).fit(X_tr_clean, y_tr_all)
    probs_flat = clf_flat.predict_proba(X_va_clean)
    pred_flat = classes_all[(np.log(np.clip(probs_flat, 1e-9, 1.0)) + bias_all).argmax(axis=1)]
    f1_flat = f1_score(y_va_opp, pred_flat, average="weighted", zero_division=0)
    print(f"\n--> Flat Baseline LightGBM F1: {f1_flat:.5f}")

    # 2. Stage 1: Binary Classifier (Inactivity vs Active)
    print("\n--> 2. Fitting Stage 1: Inactivity vs Active Binary Classifier...")
    y_tr_is_inact = (y_tr_opp == "Inactivity").astype(int)
    y_va_is_inact = (y_va_opp == "Inactivity").astype(int)

    clf_stage1 = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.04,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    ).fit(X_tr_clean, y_tr_is_inact)
    probs_stage1 = clf_stage1.predict_proba(X_va_clean)  # col 0 = Active, col 1 = Inactive
    p_inact_va = probs_stage1[:, 1]
    p_active_va = probs_stage1[:, 0]
    f1_stage1 = f1_score(y_va_is_inact, p_inact_va > 0.5, average="binary")
    print(f"       Stage 1 Inactivity Binary F1: {f1_stage1:.5f}")

    # 3. Stage 2: 16-Class Classifier among Active Customers
    print("\n--> 3. Fitting Stage 2: 16-Class Multiclass Classifier (trained on active customers only)...")
    active_mask_tr = (y_tr_opp != "Inactivity")
    X_tr_active = X_tr_clean[active_mask_tr]
    y_tr_active = y_tr_opp[active_mask_tr]

    le_active = LabelEncoder()
    y_tr_active_encoded = le_active.fit_transform(y_tr_active)
    classes_active = le_active.classes_
    print(f"       Active training rows: {len(X_tr_active):,} across {len(classes_active)} classes")

    clf_stage2 = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.04,
        num_class=len(classes_active),
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    ).fit(X_tr_active, y_tr_active_encoded)
    probs_stage2 = clf_stage2.predict_proba(X_va_clean)

    # 4. Hierarchical Probability Synthesis
    print("\n--> 4. Synthesizing Full 17-Class Probabilities...")
    inact_idx_all = np.where(classes_all == "Inactivity")[0][0]
    P_hier = np.zeros((len(X_va_clean), len(classes_all)), dtype=float)
    P_hier[:, inact_idx_all] = p_inact_va

    for k, cls_name in enumerate(classes_active):
        idx_all = np.where(classes_all == cls_name)[0][0]
        P_hier[:, idx_all] = p_active_va * probs_stage2[:, k]

    # Apply calibrated decision biases
    log_p_hier = np.log(np.clip(P_hier, 1e-9, 1.0)) + bias_all
    pred_hier = classes_all[log_p_hier.argmax(axis=1)]
    f1_hier = f1_score(y_va_opp, pred_hier, average="weighted", zero_division=0)
    print(f"       Hierarchical Model F1: {f1_hier:.5f} (Delta: {f1_hier - f1_flat:+.5f})")

    # 5. Also evaluate with CatBoost for Stage 2
    print("\n--> 5. Fitting Stage 2 with CatBoost...")
    clf_cb_stage2 = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=5,
        random_seed=seed,
        verbose=0,
        thread_count=-1,
    ).fit(X_tr_active, y_tr_active_encoded)
    probs_stage2_cb = clf_cb_stage2.predict_proba(X_va_clean)

    P_hier_cb = np.zeros((len(X_va_clean), len(classes_all)), dtype=float)
    P_hier_cb[:, inact_idx_all] = p_inact_va
    for k, cls_name in enumerate(classes_active):
        idx_all = np.where(classes_all == cls_name)[0][0]
        P_hier_cb[:, idx_all] = p_active_va * probs_stage2_cb[:, k]

    log_p_hier_cb = np.log(np.clip(P_hier_cb, 1e-9, 1.0)) + bias_all
    pred_hier_cb = classes_all[log_p_hier_cb.argmax(axis=1)]
    f1_hier_cb = f1_score(y_va_opp, pred_hier_cb, average="weighted", zero_division=0)
    print(f"       Hierarchical CatBoost F1: {f1_hier_cb:.5f} (Delta: {f1_hier_cb - f1_flat:+.5f})")

    # Consensus of Hierarchical LGB and Hierarchical CB
    from collections import Counter
    consensus_hier = []
    for i in range(len(y_va_opp)):
        counts = Counter([pred_hier[i], pred_hier_cb[i]])
        top_cand, top_count = counts.most_common(1)[0]
        consensus_hier.append(top_cand if top_count >= 2 else pred_hier[i])
    f1_hier_comm = f1_score(y_va_opp, np.array(consensus_hier), average="weighted", zero_division=0)
    print(f"       Hierarchical Committee F1: {f1_hier_comm:.5f} (Delta: {f1_hier_comm - f1_flat:+.5f})")

    print("\n" + "=" * 80)
    print("EXPERIMENT B SUMMARY")
    print("=" * 80)
    print(f"Flat Baseline F1:          {f1_flat:.5f}")
    print(f"Hierarchical LightGBM F1:  {f1_hier:.5f} ({f1_hier - f1_flat:+.5f})")
    print(f"Hierarchical CatBoost F1:  {f1_hier_cb:.5f} ({f1_hier_cb - f1_flat:+.5f})")
    print(f"Hierarchical Committee F1: {f1_hier_comm:.5f} ({f1_hier_comm - f1_flat:+.5f})")
    print(f"Total time elapsed: {time.time() - t0:.1f}s")

    return {
        "f1_flat": f1_flat,
        "f1_hier": f1_hier,
        "f1_hier_cb": f1_hier_cb,
        "f1_hier_comm": f1_hier_comm,
    }


if __name__ == "__main__":
    run_experiment_b()
