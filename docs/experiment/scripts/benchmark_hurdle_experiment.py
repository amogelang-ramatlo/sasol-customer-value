"""Benchmarking Hurdle Regressors vs Direct Regressors across all Embargoed Folds.

Experiments:
1. Direct Multi-Seed Bagged Regressor (v8 baseline)
2. Pure Hurdle Regressor (Two-Stage Gate + Positive Spend)
3. Optimal Convex Blend:
   pred = alpha * pred_direct + (1 - alpha) * pred_hurdle
   Evaluates alpha from 0.0 to 1.0 on out-of-fold validation data.
"""

import time

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

from sasol_customer_value.config import TARGET_FUEL, TARGET_NONFUEL
from sasol_customer_value.models.hurdle import BaggedHurdleRegressor
from sasol_customer_value.models.validation import (
    get_embargoed_folds,
    load_fold_data,
)


def run_hurdle_experiment():
    print("=" * 80)
    print("EMBARGOED VALIDATION: HURDLE VS DIRECT REGRESSORS")
    print("=" * 80)

    seeds = (42, 100, 2024, 777, 999)

    fold_direct_f, fold_hurdle_f = [], []
    fold_direct_nf, fold_hurdle_nf = [], []
    y_true_f_list, y_true_nf_list = [], []

    for train_cutoffs, val_cutoff in get_embargoed_folds():
        t0 = time.time()
        print(
            f"\n--> Evaluating Fold: Val Cutoff = {val_cutoff} (Train cutoffs: {len(train_cutoffs)})"
        )
        X_train, y_train, X_val, y_val = load_fold_data(train_cutoffs, val_cutoff)

        y_f_val = y_val[TARGET_FUEL].to_numpy(dtype=float)
        y_nf_val = y_val[TARGET_NONFUEL].to_numpy(dtype=float)
        y_true_f_list.append(y_f_val)
        y_true_nf_list.append(y_nf_val)

        # -------------------------------------------------------------
        # 1. Direct Multi-Seed Regressors (Baseline v8)
        # -------------------------------------------------------------
        print("    Fitting Direct 5-Seed Regressors...")
        d_f_preds, d_nf_preds = [], []
        for s in seeds:
            # Fuel (CatBoost + LightGBM blend)
            m_cb_f = CatBoostRegressor(
                iterations=200,
                learning_rate=0.04,
                random_seed=s,
                verbose=0,
                thread_count=-1,
            )
            m_cb_f.fit(X_train, y_train[TARGET_FUEL])
            d_f_preds.append(np.clip(m_cb_f.predict(X_val), 0, None))

            # Nonfuel (CatBoost 60% + LightGBM 40%)
            m_cb_nf = CatBoostRegressor(
                iterations=200,
                learning_rate=0.04,
                random_seed=s,
                verbose=0,
                thread_count=-1,
            )
            m_lgb_nf = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.03,
                random_state=s,
                n_jobs=-1,
                verbose=-1,
            )
            m_cb_nf.fit(X_train, y_train[TARGET_NONFUEL])
            m_lgb_nf.fit(X_train, y_train[TARGET_NONFUEL])
            p_nf = 0.60 * np.clip(m_cb_nf.predict(X_val), 0, None) + 0.40 * np.clip(
                m_lgb_nf.predict(X_val), 0, None
            )
            d_nf_preds.append(p_nf)

        p_direct_f = np.mean(d_f_preds, axis=0)
        p_direct_nf = np.mean(d_nf_preds, axis=0)
        fold_direct_f.append(p_direct_f)
        fold_direct_nf.append(p_direct_nf)

        rmse_dir_f = np.sqrt(mean_squared_error(y_f_val, p_direct_f))
        rmse_dir_nf = np.sqrt(mean_squared_error(y_nf_val, p_direct_nf))

        # -------------------------------------------------------------
        # 2. Hurdle Regressors
        # -------------------------------------------------------------
        print("    Fitting Hurdle 5-Seed Regressors (Gate + Positive Spend)...")
        hurdle_f = BaggedHurdleRegressor(seeds=seeds)
        hurdle_nf = BaggedHurdleRegressor(seeds=seeds)

        hurdle_f.fit(X_train, y_train[TARGET_FUEL])
        hurdle_nf.fit(X_train, y_train[TARGET_NONFUEL])

        p_hurdle_f = hurdle_f.predict(X_val)
        p_hurdle_nf = hurdle_nf.predict(X_val)
        fold_hurdle_f.append(p_hurdle_f)
        fold_hurdle_nf.append(p_hurdle_nf)

        rmse_hur_f = np.sqrt(mean_squared_error(y_f_val, p_hurdle_f))
        rmse_hur_nf = np.sqrt(mean_squared_error(y_nf_val, p_hurdle_nf))

        # 50/50 Blend check
        p_blend_f = 0.50 * p_direct_f + 0.50 * p_hurdle_f
        p_blend_nf = 0.50 * p_direct_nf + 0.50 * p_hurdle_nf
        rmse_bld_f = np.sqrt(mean_squared_error(y_f_val, p_blend_f))
        rmse_bld_nf = np.sqrt(mean_squared_error(y_nf_val, p_blend_nf))

        print(f"    Fold {val_cutoff} Results ({time.time() - t0:.1f}s):")
        print(
            f"      Fuel RMSE:    Direct={rmse_dir_f:.5f} | Hurdle={rmse_hur_f:.5f} | 50/50 Blend={rmse_bld_f:.5f} ({rmse_bld_f - rmse_dir_f:+.5f})"
        )
        print(
            f"      Nonfuel RMSE: Direct={rmse_dir_nf:.5f} | Hurdle={rmse_hur_nf:.5f} | 50/50 Blend={rmse_bld_nf:.5f} ({rmse_bld_nf - rmse_dir_nf:+.5f})"
        )

    # Global Out-Of-Fold Analysis
    print("\n" + "=" * 80)
    print("GLOBAL OUT-OF-FOLD (OOF) CONVEX BLEND OPTIMIZATION")
    print("=" * 80)

    y_f_all = np.concatenate(y_true_f_list)
    y_nf_all = np.concatenate(y_true_nf_list)
    d_f_all = np.concatenate(fold_direct_f)
    h_f_all = np.concatenate(fold_hurdle_f)
    d_nf_all = np.concatenate(fold_direct_nf)
    h_nf_all = np.concatenate(fold_hurdle_nf)

    print("--- Fuel Blend Optimization (weight on Direct) ---")
    best_w_f, best_rmse_f = 1.0, np.sqrt(mean_squared_error(y_f_all, d_f_all))
    for w in np.linspace(0.0, 1.0, 21):
        pred = w * d_f_all + (1 - w) * h_f_all
        rmse = np.sqrt(mean_squared_error(y_f_all, pred))
        print(f"  w_direct = {w:.2f} -> Global Fuel RMSE = {rmse:.5f}")
        if rmse < best_rmse_f:
            best_rmse_f, best_w_f = rmse, w

    print("\n--- Non-Fuel Blend Optimization (weight on Direct) ---")
    best_w_nf, best_rmse_nf = 1.0, np.sqrt(mean_squared_error(y_nf_all, d_nf_all))
    for w in np.linspace(0.0, 1.0, 21):
        pred = w * d_nf_all + (1 - w) * h_nf_all
        rmse = np.sqrt(mean_squared_error(y_nf_all, pred))
        print(f"  w_direct = {w:.2f} -> Global Nonfuel RMSE = {rmse:.5f}")
        if rmse < best_rmse_nf:
            best_rmse_nf, best_w_nf = rmse, w

    base_f = np.sqrt(mean_squared_error(y_f_all, d_f_all))
    base_nf = np.sqrt(mean_squared_error(y_nf_all, d_nf_all))
    print("\n" + "=" * 80)
    print("FINAL SUMMARY OF HURDLE EXPERIMENT")
    print("=" * 80)
    print(
        f"Fuel RMSE:    Baseline Direct={base_f:.5f} -> Optimal Blend (w={best_w_f:.2f})={best_rmse_f:.5f} (Delta: {best_rmse_f - base_f:+.5f})"
    )
    print(
        f"Nonfuel RMSE: Baseline Direct={base_nf:.5f} -> Optimal Blend (w={best_w_nf:.2f})={best_rmse_nf:.5f} (Delta: {best_rmse_nf - base_nf:+.5f})"
    )
    print("=" * 80)


if __name__ == "__main__":
    run_hurdle_experiment()
