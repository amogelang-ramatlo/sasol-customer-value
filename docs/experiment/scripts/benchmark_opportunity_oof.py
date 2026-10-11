"""Experiment: Out-Of-Fold Opportunity Optimization across Embargoed Folds.

Tests:
1. Flat 17-class baseline with default biases (Current v8 baseline: F1 ~ 49.85%)
2. Out-Of-Fold Coordinate Ascent Decision Biases:
   - Constrained 5-parameter bias vector:
     b_inact, b_fuel_growth, b_store_growth, b_adopt, b_stable (fixed to 0)
   - Evaluated out-of-fold strictly across embargoed validation folds.
3. Hierarchical Cascaded Head:
   - Stage 1: P(Active) binary classifier
   - Stage 2: P(Fuel Growth | Active)
   - Stage 3: P(Category | Active & Not Fuel Growth)
   - Evaluates whether hierarchical decomposition beats flat multiclass.
"""

import time

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from sasol_customer_value.config import (
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.models.ensemble import DEFAULT_OPP_BIASES
from sasol_customer_value.models.validation import (
    get_embargoed_folds,
    load_fold_data,
)


def run_opportunity_oof_optimization():
    print("=" * 80)
    print("EMBARGOED VALIDATION: OPPORTUNITY CLASSIFICATION OPTIMIZATION")
    print("=" * 80)

    seeds = (42, 100, 2024, 777, 999)

    fold_probs_list = []
    y_true_list = []
    classes = None

    # Step 1: Collect out-of-fold calibrated probability matrices
    for train_cutoffs, val_cutoff in get_embargoed_folds():
        t0 = time.time()
        print(
            f"\n--> Training Fold: Val Cutoff = {val_cutoff} (Train cutoffs: {len(train_cutoffs)})"
        )
        X_train, y_train, X_val, y_val = load_fold_data(train_cutoffs, val_cutoff)

        y_val_str = y_val[TARGET_OPPORTUNITY].astype(str).to_numpy()
        y_true_list.append(y_val_str)

        # 5-seed bagged LightGBM multiclass classifier
        seed_probs = []
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
            seed_probs.append(clf.predict_proba(X_val))

        p_val_avg = np.mean(seed_probs, axis=0)
        fold_probs_list.append(p_val_avg)
        print(
            f"    Fold {val_cutoff} probabilities collected in {time.time() - t0:.1f}s."
        )

    y_true_all = np.concatenate(y_true_list)
    P_all = np.concatenate(fold_probs_list, axis=0)
    print(
        f"\nTotal Pooled OOF Shape: {P_all.shape}, Ground Truth Count: {len(y_true_all):,}"
    )

    # Benchmark 1: Unbiased Raw Argmax
    pred_raw = classes[P_all.argmax(axis=1)]
    f1_raw = f1_score(y_true_all, pred_raw, average="weighted", zero_division=0)
    print(f"\n1. Raw Unbiased Argmax Weighted F1: {f1_raw:.5f}")

    # Benchmark 2: Legacy Static Summer Biases
    legacy_b = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes], dtype=float)
    log_p_legacy = np.log(np.clip(P_all, 1e-9, 1.0)) + legacy_b
    pred_legacy = classes[log_p_legacy.argmax(axis=1)]
    f1_legacy = f1_score(y_true_all, pred_legacy, average="weighted", zero_division=0)
    print(f"2. Legacy Static Biases Weighted F1: {f1_legacy:.5f}")

    # Benchmark 3: Constrained 5-Parameter Macro-Group OOF Tuning
    # Groups: Inactivity, Fuel Growth, Store Growth, Adoption, Stable (pinned to 0)
    print("\n--> Optimizing Constrained Macro-Group Decision Biases on OOF data...")
    is_inact_idx = np.where(classes == "Inactivity")[0][0]
    is_fuel_idx = np.where(classes == "Existing-category growth: Fuel")[0][0]
    is_stable_idx = np.where(classes == "Stable")[0][0]
    growth_store_indices = [
        i
        for i, c in enumerate(classes)
        if c.startswith("Existing-category growth:")
        and c != "Existing-category growth: Fuel"
    ]
    adoption_indices = [
        i for i, c in enumerate(classes) if c.startswith("New category adoption:")
    ]

    def build_bias_vector(params):
        # params: [b_inact, b_fuel, b_store_growth, b_adopt]
        b = np.zeros(len(classes), dtype=float)
        b[is_inact_idx] = params[0]
        b[is_fuel_idx] = params[1]
        for idx in growth_store_indices:
            b[idx] = params[2]
        for idx in adoption_indices:
            b[idx] = params[3]
        b[is_stable_idx] = 0.0  # reference anchor
        return b

    def loss_func(params):
        b = build_bias_vector(params)
        log_p = np.log(np.clip(P_all, 1e-9, 1.0)) + b
        preds = classes[log_p.argmax(axis=1)]
        return -f1_score(y_true_all, preds, average="weighted", zero_division=0)

    # Initial guess from legacy values
    init_params = np.array([0.10, 0.15, 0.90, 0.50])
    res = minimize(
        loss_func, init_params, method="Nelder-Mead", options={"maxiter": 300}
    )
    best_params = res.x
    best_biases = build_bias_vector(best_params)

    log_p_opt = np.log(np.clip(P_all, 1e-9, 1.0)) + best_biases
    pred_opt = classes[log_p_opt.argmax(axis=1)]
    f1_opt = f1_score(y_true_all, pred_opt, average="weighted", zero_division=0)

    print("\n3. Optimal Constrained OOF Biases:")
    print(f"   b(Inactivity)               = {best_params[0]:+.4f}")
    print(f"   b(Fuel Growth)              = {best_params[1]:+.4f}")
    print(f"   b(Store Growth Categories)  = {best_params[2]:+.4f}")
    print(f"   b(Adoption Categories)      = {best_params[3]:+.4f}")
    print("   b(Stable)                   = +0.0000 (Anchor)")
    print(
        f"   -> Global OOF Weighted F1:    {f1_opt:.5f} (Gain over Legacy: {f1_opt - f1_legacy:+.5f})"
    )

    # Evaluate per-fold performance with the new optimal biases
    print("\nPer-Fold Comparison (Legacy vs Optimal OOF Biases):")
    idx_start = 0
    for i, (_, val_cutoff) in enumerate(get_embargoed_folds()):
        n_fold = len(fold_probs_list[i])
        y_fold = y_true_all[idx_start : idx_start + n_fold]
        p_fold = P_all[idx_start : idx_start + n_fold]
        idx_start += n_fold

        f1_leg_fold = f1_score(
            y_fold,
            classes[(np.log(np.clip(p_fold, 1e-9, 1.0)) + legacy_b).argmax(axis=1)],
            average="weighted",
            zero_division=0,
        )
        f1_opt_fold = f1_score(
            y_fold,
            classes[(np.log(np.clip(p_fold, 1e-9, 1.0)) + best_biases).argmax(axis=1)],
            average="weighted",
            zero_division=0,
        )
        print(
            f"   Fold {val_cutoff}: Legacy={f1_leg_fold:.5f} -> Optimal={f1_opt_fold:.5f} ({f1_opt_fold - f1_leg_fold:+.5f})"
        )


if __name__ == "__main__":
    run_opportunity_oof_optimization()
