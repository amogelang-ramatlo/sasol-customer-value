"""Build and evaluate Model Best 2 (Single-Seed Committee Consensus) on Seed 42.

Model Best 2 Architecture:
- Continuous spend: Model B (Seed 42)
- Opportunity: Committee Consensus Majority Vote among:
  * Model A (202 features, Seed 42)
  * Model B (218 features, Seed 42)
  * Model C (225 features, Seed 42)
- Output: submissions/sub_model_best2.csv

Evaluates on the held-out 2025-09-01 validation cohort to report local metrics:
- Local Fuel RMSE
- Local Nonfuel RMSE
- Local Opportunity Weighted F1
- Local Overall Composite Score
"""

import argparse
import time
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import f1_score, mean_squared_error
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
    X_eval: pd.DataFrame,
    seed: int,
    reg_weights_fuel: tuple[float, float, float],
    reg_weights_nf: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train single-seed regressors and classifier, returning predictions."""
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

    p_lgb_f = np.clip(lgb_f.predict(X_eval), 0.0, None)
    p_cb_f = np.clip(cb_f.predict(X_eval), 0.0, None)
    p_xgb_f = np.clip(xgb_f.predict(X_eval), 0.0, None)
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

    p_lgb_nf = np.clip(lgb_nf.predict(X_eval), 0.0, None)
    p_cb_nf = np.clip(cb_nf.predict(X_eval), 0.0, None)
    pred_nonfuel = reg_weights_nf[0] * p_lgb_nf + reg_weights_nf[1] * p_cb_nf

    # Opportunity classifier (LightGBM)
    clf = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.04,
        num_class=17,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    ).fit(X_train, y_opp)

    probs = clf.predict_proba(X_eval)
    classes = clf.classes_

    return pred_fuel, pred_nonfuel, probs, classes


def apply_bias(
    probs: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Apply summer holiday log-odds biases to predicted probabilities."""
    bias_vec = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes], dtype=float)
    log_probs = np.log(np.clip(probs, 1e-9, 1.0)) + bias_vec
    return classes[log_probs.argmax(axis=1)].copy()


def build_and_evaluate_single_seed(seed: int = 42) -> None:
    t0 = time.time()
    print("=" * 80)
    print(f"BUILDING & EVALUATING MODEL BEST 2 (SEED {seed})")
    print("=" * 80)

    # 1. Load data
    print("--> 1. Loading historical snapshots...")
    cutoffs_embargoed = [
        str(d.date()) for d in pd.date_range("2024-07-01", "2025-06-01", freq="MS")
    ]
    cutoffs_all = [
        str(d.date()) for d in pd.date_range("2024-07-01", "2025-09-01", freq="MS")
    ]

    # Embargoed training for local validation
    X_tr_emb = pd.concat(
        [pd.read_parquet(cached_paths(c)[0]) for c in cutoffs_embargoed],
        ignore_index=True,
    )
    y_tr_emb = pd.concat(
        [pd.read_parquet(cached_paths(c)[1]) for c in cutoffs_embargoed],
        ignore_index=True,
    )

    # Full training for final test submission (all 15 snapshots)
    X_tr_full = pd.concat(
        [pd.read_parquet(cached_paths(c)[0]) for c in cutoffs_all], ignore_index=True
    )
    y_tr_full = pd.concat(
        [pd.read_parquet(cached_paths(c)[1]) for c in cutoffs_all], ignore_index=True
    )

    # Validation cohort 2025-09-01
    X_va = pd.read_parquet(cached_paths("2025-09-01")[0])
    y_va = pd.read_parquet(cached_paths("2025-09-01")[1])
    y_va_opp = y_va[TARGET_OPPORTUNITY].astype(str).to_numpy()

    # Test customers
    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))

    print("--> 2. Extracting test features as of 2025-12-01...")
    df_trans = load_transactions()
    X_test_full = build_features(
        df_trans, cutoff=PUBLIC_HISTORY_END_EXCLUSIVE, customer_ids=pd.Index(test_ids)
    ).reindex(test_ids)

    cols_202 = list(X_tr_full.columns[:202])
    cols_218 = list(X_tr_full.columns[:218])
    cols_225 = list(X_tr_full.columns[:225])

    # Clean missing values using median imputation
    med_emb = X_tr_emb.median()
    X_tr_emb_clean = X_tr_emb.fillna(med_emb)
    X_va_clean = X_va.fillna(med_emb)

    med_full = X_tr_full.median()
    X_tr_full_clean = X_tr_full.fillna(med_full)
    X_test_full_clean = X_test_full.fillna(med_full)

    # =========================================================================
    # LOCAL VALIDATION EVALUATION (ON 2025-09-01 COHORT)
    # =========================================================================
    print("\n" + "=" * 60)
    print("--> LOCAL VALIDATION: FITTING MODELS ON EMBARGOED TRAINING SPLIT...")
    print("=" * 60)

    # Model A on val (202 features, seed)
    print("    Training Model A on validation split (202 features)...")
    _f_va_A, _nf_va_A, probs_va_A, classes_A = train_single_model(
        X_tr_emb_clean[cols_202],
        y_tr_emb,
        X_va_clean[cols_202],
        seed=seed,
        reg_weights_fuel=(0.50, 0.30, 0.20),
        reg_weights_nf=(0.50, 0.50),
    )
    opp_va_A = apply_bias(probs_va_A, classes_A)

    # Model B on val (218 features, seed)
    print("    Training Model B on validation split (218 features)...")
    f_va_B, nf_va_B, probs_va_B, classes_B = train_single_model(
        X_tr_emb_clean[cols_218],
        y_tr_emb,
        X_va_clean[cols_218],
        seed=seed,
        reg_weights_fuel=(0.20, 0.55, 0.25),
        reg_weights_nf=(0.40, 0.60),
    )
    opp_va_B = apply_bias(probs_va_B, classes_B)

    # Model C on val (225 features, seed)
    print("    Training Model C on validation split (225 features)...")
    _f_va_C, _nf_va_C, probs_va_C, classes_C = train_single_model(
        X_tr_emb_clean[cols_225],
        y_tr_emb,
        X_va_clean[cols_225],
        seed=seed,
        reg_weights_fuel=(0.20, 0.55, 0.25),
        reg_weights_nf=(0.40, 0.60),
    )
    opp_va_C = apply_bias(probs_va_C, classes_C)

    # Consensus on validation split
    opp_va_consensus = []
    for i in range(len(y_va_opp)):
        counts = Counter([opp_va_A[i], opp_va_B[i], opp_va_C[i]])
        top_cand, top_count = counts.most_common(1)[0]
        opp_va_consensus.append(top_cand if top_count >= 2 else opp_va_B[i])
    opp_va_consensus = np.array(opp_va_consensus)

    # Compute Local Metrics
    rmse_fuel_B = float(np.sqrt(mean_squared_error(y_va[TARGET_FUEL], f_va_B)))
    rmse_nf_B = float(np.sqrt(mean_squared_error(y_va[TARGET_NONFUEL], nf_va_B)))

    f1_model2 = float(
        f1_score(y_va_opp, opp_va_consensus, average="weighted", zero_division=0)
    )
    score_model2 = (
        0.60 + 0.40 * f1_model2 - 0.40540558 * rmse_fuel_B - 0.36764644 * rmse_nf_B
    )

    # =========================================================================
    # FULL TRAINING & TEST INFERENCE
    # =========================================================================
    print("\n" + "=" * 60)
    print(
        "--> FULL TRAINING: FITTING MODELS ACROSS ALL 15 SNAPSHOTS FOR TEST SUBMISSIONS..."
    )
    print("=" * 60)

    # Model A on full (202 features, seed)
    print("    Training Model A on full dataset (202 features)...")
    _f_test_A, _nf_test_A, probs_test_A, classes_test_A = train_single_model(
        X_tr_full_clean[cols_202],
        y_tr_full,
        X_test_full_clean[cols_202],
        seed=seed,
        reg_weights_fuel=(0.50, 0.30, 0.20),
        reg_weights_nf=(0.50, 0.50),
    )
    opp_test_A = apply_bias(probs_test_A, classes_test_A)

    # Model B on full (218 features, seed)
    print("    Training Model B on full dataset (218 features)...")
    f_test_B, nf_test_B, probs_test_B, classes_test_B = train_single_model(
        X_tr_full_clean[cols_218],
        y_tr_full,
        X_test_full_clean[cols_218],
        seed=seed,
        reg_weights_fuel=(0.20, 0.55, 0.25),
        reg_weights_nf=(0.40, 0.60),
    )
    opp_test_B = apply_bias(probs_test_B, classes_test_B)

    # Model C on full (225 features, seed)
    print("    Training Model C on full dataset (225 features)...")
    _f_test_C, _nf_test_C, probs_test_C, classes_test_C = train_single_model(
        X_tr_full_clean[cols_225],
        y_tr_full,
        X_test_full_clean[cols_225],
        seed=seed,
        reg_weights_fuel=(0.20, 0.55, 0.25),
        reg_weights_nf=(0.40, 0.60),
    )
    opp_test_C = apply_bias(probs_test_C, classes_test_C)

    # Consensus on test split
    opp_test_consensus = []
    for i in range(len(test_ids)):
        counts = Counter([opp_test_A[i], opp_test_B[i], opp_test_C[i]])
        top_cand, top_count = counts.most_common(1)[0]
        opp_test_consensus.append(top_cand if top_count >= 2 else opp_test_B[i])
    opp_test_consensus = np.array(opp_test_consensus)

    # Save Model Best 2 submission (sub_model_best2.csv)
    out_model = SUBMISSIONS_DIR / "sub_model_best2.csv"
    sub_model = pd.DataFrame(
        {
            ID_COL: test_ids,
            "CLV_fuel": f_test_B,
            "CLV_nonfuel": nf_test_B,
            "Opportunity": opp_test_consensus,
        }
    )
    validate_submission_format(sub_model, expected_ids=test_ids)
    sub_model.to_csv(out_model, index=False)
    print(f"[SUCCESS] Saved Model Best 2 to: {out_model}")

    # Summary Scorecard
    print("\n" + "=" * 80)
    print(f"LOCAL VALIDATION SCORECARD (2025-09-01 COHORT | SEED {seed})")
    print("=" * 80)
    print(f"Local Fuel RMSE:                 {rmse_fuel_B:.5f}")
    print(f"Local Nonfuel RMSE:              {rmse_nf_B:.5f}")
    print(f"Model Best 2 Opportunity F1:     {f1_model2:.5f}")
    print(f"Model Best 2 Overall Score:      {score_model2:.5f}")
    print("=" * 80)
    print(f"Total pipeline runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed value to train single models"
    )
    args = parser.parse_args()
    build_and_evaluate_single_seed(seed=args.seed)
