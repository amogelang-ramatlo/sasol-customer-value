"""Standalone script to build Flagship Model (Target Synthesis Architecture).

Target Synthesis Architecture:
1. CLV_fuel:
   Sourced from Model A (0.50 LGB / 0.30 CB / 0.20 XGB on Seed 42, 202 features)
2. CLV_nonfuel:
   Sourced from Model B (5-seed bagged 0.40 LGB / 0.60 CB, 218 features)
3. Opportunity:
   Sourced from Committee Consensus Majority Vote (Model A, Model B, Model C on summer-calibrated biases)

Objective Mathematical Formulation:
Overall = 0.60 + 0.40 * F1(Opp) - 0.4054056 * RMSE(Fuel) - 0.3676464 * RMSE(Nonfuel)
"""

import argparse
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


def train_model_best1_from_scratch(
    output_path: Path = SUBMISSIONS_DIR / "sub_model_best1.csv",
) -> Path:
    """Train all component models from scratch across all 15 historical snapshots and synthesize."""
    t0 = time.time()
    print("=" * 80)
    print("TRAINING PRIMARY MODEL BEST 1 FROM SCRATCH")
    print("=" * 80)

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

    df_trans = load_transactions()
    X_test_full = build_features(
        df_trans, cutoff=PUBLIC_HISTORY_END_EXCLUSIVE, customer_ids=pd.Index(test_ids)
    ).reindex(test_ids)

    cols_202 = list(X_full.columns[:202])
    cols_218 = list(X_full.columns[:218])
    cols_225 = list(X_full.columns[:225])

    # 1. Model A (5 seeds, 202 features) -> CLV_fuel
    print("--> 1. Training Model A (202 features, Seeds 42, 100, 2024, 777, 999)...")
    seeds = [42, 100, 2024, 777, 999]
    f_list_A, probs_list_A = [], []
    classes_A = None
    for s in seeds:
        print(f"       Fitting seed {s}...")
        p_f, _, probs, cls = train_single_model(
            X_full[cols_202],
            y_full,
            X_test_full[cols_202],
            seed=s,
            reg_weights_fuel=(0.50, 0.30, 0.20),
            reg_weights_nf=(0.50, 0.50),
        )
        f_list_A.append(p_f)
        probs_list_A.append(probs)
        if classes_A is None:
            classes_A = cls

    f_pred_A = np.mean(f_list_A, axis=0)
    avg_probs_A = np.mean(probs_list_A, axis=0)
    bias_vec_A = np.array(
        [DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes_A], dtype=float
    )
    opp_pred_A = classes_A[
        (np.log(np.clip(avg_probs_A, 1e-9, 1.0)) + bias_vec_A).argmax(axis=1)
    ]

    # 2. Model B (5 seeds, 218 features) -> CLV_nonfuel
    print("--> 2. Training Model B (218 features, Seeds 42, 100, 2024, 777, 999)...")
    seeds = [42, 100, 2024, 777, 999]
    nf_list_B, probs_list_B = [], []
    classes_B = None
    for s in seeds:
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
    bias_vec_B = np.array(
        [DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes_B], dtype=float
    )
    opp_pred_B = classes_B[
        (np.log(np.clip(avg_probs_B, 1e-9, 1.0)) + bias_vec_B).argmax(axis=1)
    ]

    # 3. Model C (5 seeds, 225 features) -> Consensus voter
    print("--> 3. Training Model C (225 features, Seeds 42, 100, 2024, 777, 999)...")
    probs_list_C = []
    classes_C = None
    for s in seeds:
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
    bias_vec_C = np.array(
        [DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes_C], dtype=float
    )
    opp_pred_C = classes_C[
        (np.log(np.clip(avg_probs_C, 1e-9, 1.0)) + bias_vec_C).argmax(axis=1)
    ]

    # 4. Majority Consensus Vote
    print("--> 4. Forming Committee Majority Consensus...")
    consensus_opp = []
    for i in range(len(test_ids)):
        counts = Counter([opp_pred_A[i], opp_pred_B[i], opp_pred_C[i]])
        top_cand, top_count = counts.most_common(1)[0]
        consensus_opp.append(top_cand if top_count >= 2 else opp_pred_B[i])

    sub_best1 = pd.DataFrame(
        {
            ID_COL: test_ids,
            "CLV_fuel": f_pred_A,
            "CLV_nonfuel": nf_pred_B,
            "Opportunity": consensus_opp,
        }
    )
    validate_submission_format(sub_best1, expected_ids=test_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub_best1.to_csv(output_path, index=False)
    print(
        f"\n[SUCCESS] Trained and verified Model Best 1 at: {output_path} (Elapsed: {time.time() - t0:.1f}s)"
    )
    return output_path


def build_sub_model_best1(
    output_dir: Path = SUBMISSIONS_DIR, train_scratch: bool = False
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sub_model_best1.csv"

    # If file exists and train_scratch is False, keep existing verified file
    if out_path.exists() and not train_scratch:
        print(f"[INFO] Using existing verified output: {out_path}")
        df_existing = pd.read_csv(out_path)
        print("\nTarget Summary:")
        print(f"CLV_fuel mean:    {df_existing['CLV_fuel'].mean():.4f}")
        print(f"CLV_nonfuel mean: {df_existing['CLV_nonfuel'].mean():.4f}")
        print("\nOpportunity Distribution:")
        print(df_existing["Opportunity"].value_counts())
        return out_path

    return train_model_best1_from_scratch(output_path=out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help="Train all component models from scratch",
    )
    args = parser.parse_args()
    build_sub_model_best1(train_scratch=args.train_from_scratch)
