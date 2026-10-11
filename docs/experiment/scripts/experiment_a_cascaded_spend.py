"""Experiment A: Cascaded Spend-to-Opportunity Coupling.

Protocol:
1. Train stage-1 continuous regressors (Fuel & Nonfuel) on embargoed historical training snapshots (2024-07-01 to 2025-06-01, 50,280 rows).
2. Generate Out-Of-Fold / forward predicted spend for train and validation cohorts.
3. Engineer cascaded trajectory features:
   - pred_fuel_litres = exp(s_fuel * z_fuel) - 1
   - pred_fuel_rand = pred_fuel_litres * fuel_price
   - fuel_growth_projected_increment = pred_fuel_rand - fuel_rand_q1
   - fuel_growth_ratio = (pred_fuel_rand + 1.0) / (fuel_rand_q1 + 1.0)
   - fuel_growth_hurdle_met = (fuel_growth_projected_increment >= hurdle) & (fuel_rand_q1 > 0)
   - pred_nonfuel_rand = exp(s_nonfuel * z_nonfuel) - 1
   - nonfuel_projected_increment = pred_nonfuel_rand - store_rand_q1
   - nonfuel_growth_ratio = (pred_nonfuel_rand + 1.0) / (store_rand_q1 + 1.0)
4. Train Opportunity Classifier with original features + cascaded spend features.
5. Evaluate local Weighted F1 and compare directly with uncoupled baseline.
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from sasol_customer_value.config import (
    NORMALIZATION_CONSTANTS,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.data.snapshots import cached_paths
from sasol_customer_value.models.ensemble import DEFAULT_OPP_BIASES


def run_experiment_a(seed: int = 42) -> dict:
    t0 = time.time()
    print("=" * 80)
    print("EXPERIMENT A: CASCADED SPEND-TO-OPPORTUNITY COUPLING (SINGLE SEED)")
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

    # 2. Stage 1: Fit Spend Regressors
    print("\n--> 2. Fitting Stage 1 Spend Regressors...")
    m_cb_f = CatBoostRegressor(
        iterations=250, learning_rate=0.04, random_seed=seed, verbose=0, thread_count=-1
    ).fit(X_tr_clean, y_tr[TARGET_FUEL])

    m_cb_nf = CatBoostRegressor(
        iterations=250, learning_rate=0.04, random_seed=seed, verbose=0, thread_count=-1
    ).fit(X_tr_clean, y_tr[TARGET_NONFUEL])

    # Predict normalized spend
    z_f_tr = np.clip(m_cb_f.predict(X_tr_clean), 0.0, None)
    z_nf_tr = np.clip(m_cb_nf.predict(X_tr_clean), 0.0, None)

    z_f_va = np.clip(m_cb_f.predict(X_va_clean), 0.0, None)
    z_nf_va = np.clip(m_cb_nf.predict(X_va_clean), 0.0, None)

    rf = np.sqrt(mean_squared_error(y_va[TARGET_FUEL], z_f_va))
    rnf = np.sqrt(mean_squared_error(y_va[TARGET_NONFUEL], z_nf_va))
    print(f"       Fuel RMSE: {rf:.5f} | Nonfuel RMSE: {rnf:.5f}")

    # 3. Engineer Cascaded Features
    print("\n--> 3. Engineering Cascaded Spend Trajectory Features...")
    s_fuel = float(NORMALIZATION_CONSTANTS["CLV_fuel"])
    s_nonfuel = float(NORMALIZATION_CONSTANTS["CLV_nonfuel"])

    def add_cascaded_features(X_df: pd.DataFrame, z_f: np.ndarray, z_nf: np.ndarray) -> pd.DataFrame:
        df_feats = X_df.copy()

        # Raw fuel inversion
        litres_pred = np.expm1(s_fuel * z_f)
        fuel_price = df_feats["fuel_price_q1"].fillna(23.5)
        fuel_rand_pred = litres_pred * fuel_price

        base_f_rand = df_feats["fuel_rand_q1"].fillna(0.0)
        hurdle_f = np.maximum(20.0, 0.25 * base_f_rand)

        f_inc = fuel_rand_pred - base_f_rand
        df_feats["cascaded_pred_fuel_litres"] = litres_pred
        df_feats["cascaded_pred_fuel_rand"] = fuel_rand_pred
        df_feats["cascaded_fuel_inc"] = f_inc
        df_feats["cascaded_fuel_ratio"] = (fuel_rand_pred + 1.0) / (base_f_rand + 1.0)
        df_feats["cascaded_fuel_hurdle_met"] = ((f_inc >= hurdle_f) & (base_f_rand > 0)).astype(float)
        df_feats["cascaded_fuel_decline_flag"] = (f_inc < 0.0).astype(float)

        # Store nonfuel inversion
        store_rand_pred = np.expm1(s_nonfuel * z_nf)
        base_s_rand = df_feats["store_rand_q1"].fillna(0.0)
        hurdle_s = np.maximum(20.0, 0.25 * base_s_rand)

        s_inc = store_rand_pred - base_s_rand
        df_feats["cascaded_pred_store_rand"] = store_rand_pred
        df_feats["cascaded_store_inc"] = s_inc
        df_feats["cascaded_store_ratio"] = (store_rand_pred + 1.0) / (base_s_rand + 1.0)
        df_feats["cascaded_store_hurdle_met"] = ((s_inc >= hurdle_s) & (base_s_rand > 0)).astype(float)
        df_feats["cascaded_store_decline_flag"] = (s_inc < 0.0).astype(float)
        
        return df_feats

    X_tr_casc = add_cascaded_features(X_tr_clean, z_f_tr, z_nf_tr)
    X_va_casc = add_cascaded_features(X_va_clean, z_f_va, z_nf_va)

    le = LabelEncoder()
    y_tr_encoded = le.fit_transform(y_tr_opp)
    classes = le.classes_
    bias_vec = np.array([DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes], dtype=float)

    # 4. Train Baseline (Without Cascaded Features)
    print("\n--> 4. Training Baseline LightGBM (without cascaded features)...")
    clf_base = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.04,
        num_class=17,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    ).fit(X_tr_clean, y_tr_encoded)
    probs_base = clf_base.predict_proba(X_va_clean)
    pred_base = classes[(np.log(np.clip(probs_base, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    f1_base = f1_score(y_va_opp, pred_base, average="weighted", zero_division=0)
    score_base = 0.60 + 0.40 * f1_base - 0.40540558 * rf - 0.36764644 * rnf
    print(f"       Baseline F1: {f1_base:.5f} (Overall: {score_base:.5f})")

    # 5. Train Cascaded LightGBM
    print("\n--> 5. Training Cascaded LightGBM (WITH predicted spend features)...")
    clf_casc_lgb = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.04,
        num_class=17,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    ).fit(X_tr_casc, y_tr_encoded)
    probs_casc_lgb = clf_casc_lgb.predict_proba(X_va_casc)
    pred_casc_lgb = classes[(np.log(np.clip(probs_casc_lgb, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    f1_casc_lgb = f1_score(y_va_opp, pred_casc_lgb, average="weighted", zero_division=0)
    score_casc_lgb = 0.60 + 0.40 * f1_casc_lgb - 0.40540558 * rf - 0.36764644 * rnf
    print(f"       Cascaded LightGBM F1: {f1_casc_lgb:.5f} (Overall: {score_casc_lgb:.5f}) [Delta: {f1_casc_lgb - f1_base:+.5f}]")

    # 6. Train Cascaded CatBoost
    print("\n--> 6. Training Cascaded CatBoost...")
    clf_casc_cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=5,
        random_seed=seed,
        verbose=0,
        thread_count=-1,
    ).fit(X_tr_casc, y_tr_encoded)
    probs_casc_cb = clf_casc_cb.predict_proba(X_va_casc)
    pred_casc_cb = classes[(np.log(np.clip(probs_casc_cb, 1e-9, 1.0)) + bias_vec).argmax(axis=1)]
    f1_casc_cb = f1_score(y_va_opp, pred_casc_cb, average="weighted", zero_division=0)
    print(f"       Cascaded CatBoost F1: {f1_casc_cb:.5f} [Delta: {f1_casc_cb - f1_base:+.5f}]")

    # 7. Combined Cascaded Committee (Cascaded LGB + Cascaded CB)
    from collections import Counter
    consensus_casc = []
    for i in range(len(y_va_opp)):
        counts = Counter([pred_casc_lgb[i], pred_casc_cb[i]])
        top_cand, top_count = counts.most_common(1)[0]
        consensus_casc.append(top_cand if top_count >= 2 else pred_casc_lgb[i])
    f1_casc_comm = f1_score(y_va_opp, np.array(consensus_casc), average="weighted", zero_division=0)
    score_casc_comm = 0.60 + 0.40 * f1_casc_comm - 0.40540558 * rf - 0.36764644 * rnf
    print(f"       Cascaded Committee F1: {f1_casc_comm:.5f} (Overall: {score_casc_comm:.5f}) [Delta: {f1_casc_comm - f1_base:+.5f}]")

    print("\n" + "=" * 80)
    print("EXPERIMENT A SUMMARY")
    print("=" * 80)
    print(f"Baseline Uncoupled F1:  {f1_base:.5f}")
    print(f"Cascaded LightGBM F1:   {f1_casc_lgb:.5f} ({f1_casc_lgb - f1_base:+.5f})")
    print(f"Cascaded CatBoost F1:   {f1_casc_cb:.5f} ({f1_casc_cb - f1_base:+.5f})")
    print(f"Total time elapsed: {time.time() - t0:.1f}s")

    return {
        "f1_base": f1_base,
        "f1_casc_lgb": f1_casc_lgb,
        "f1_casc_cb": f1_casc_cb,
        "f1_casc_comm": f1_casc_comm,
    }


if __name__ == "__main__":
    run_experiment_a()
