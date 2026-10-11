"""Benchmark calibration script: Evaluates validation folds across model configurations.

Pipelines evaluated:
1. Baseline LightGBM
2. Model Best 1 (Primary: Target Synthesis Architecture)
3. Model Best 2 (Secondary: Single-Seed Committee Consensus)

Checks:
- Strictly enforces 3-month embargo (zero target leakage across outcome windows).
- Confirms ranking and stability of local embargoed validation scores across folds.
"""

import time
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sasol_customer_value.config import (
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.models.ensemble import DEFAULT_OPP_BIASES
from sasol_customer_value.models.metrics import compute_competition_metrics
from sasol_customer_value.models.validation import (
    FoldResult,
    get_embargoed_folds,
    load_fold_data,
    summarize_validation_results,
)


def evaluate_embargoed_pipeline(pipeline_name: str) -> pd.DataFrame:
    print("\n=======================================================")
    print(f"RUNNING EMBARGOED EVALUATION FOR: {pipeline_name.upper()}")
    print("=======================================================")

    fold_results: list[FoldResult] = []

    for train_cutoffs, val_cutoff in get_embargoed_folds():
        t0 = time.time()
        print(f"\n--> Fold: Val Cutoff = {val_cutoff}")
        print(
            f"    Train Cutoffs ({len(train_cutoffs)}): {train_cutoffs[0]} to {train_cutoffs[-1]}"
        )

        X_train, y_train, X_val, y_val = load_fold_data(train_cutoffs, val_cutoff)
        n_train, n_val = len(X_train), len(X_val)
        print(
            f"    Loaded {n_train:,} train rows and {n_val:,} val rows across {X_train.shape[1]} features."
        )

        # -------------------------------------------------------------
        # Pipeline 1: Baseline LightGBM
        # -------------------------------------------------------------
        if pipeline_name == "baseline":
            # Fuel regressor
            m_f = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.03,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            m_f.fit(X_train, y_train[TARGET_FUEL])
            pred_f = np.clip(m_f.predict(X_val), 0.0, None)

            # Nonfuel regressor
            m_nf = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.03,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            m_nf.fit(X_train, y_train[TARGET_NONFUEL])
            pred_nf = np.clip(m_nf.predict(X_val), 0.0, None)

            # Opportunity classifier (uncalibrated)
            clf = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.04,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            clf.fit(X_train, y_train[TARGET_OPPORTUNITY].astype(str))
            pred_opp = clf.predict(X_val)

        # -------------------------------------------------------------
        # Pipeline 2: Model Best 2 (Single-Seed) or Model Best 1 (Synthesis)
        # -------------------------------------------------------------
        elif pipeline_name in ["model_best1", "model_best2"]:
            is_model_best2 = pipeline_name == "model_best2"
            is_model_best1 = pipeline_name == "model_best1"

            # Model Best 2 is pure single seed (42), Model Best 1 uses multi-seed bagging
            seeds = [42] if is_model_best2 else [42, 100, 2024, 777, 999]
            f_preds, nf_preds, opp_probs_list = [], [], []
            classes = None

            for s in seeds:
                # Fuel
                if is_model_best1:
                    # Model Best 1 Fuel uses 3-model blend per seed
                    m_f_lgb = lgb.LGBMRegressor(
                        n_estimators=150,
                        learning_rate=0.03,
                        random_state=s,
                        n_jobs=-1,
                        verbose=-1,
                    )
                    m_f_cb = CatBoostRegressor(
                        iterations=250,
                        learning_rate=0.04,
                        random_seed=s,
                        verbose=0,
                        thread_count=-1,
                    )
                    m_f_xgb = XGBRegressor(
                        n_estimators=150,
                        learning_rate=0.03,
                        max_depth=5,
                        random_state=s,
                        n_jobs=-1,
                        verbosity=0,
                    )
                    m_f_lgb.fit(X_train, y_train[TARGET_FUEL])
                    m_f_cb.fit(X_train, y_train[TARGET_FUEL])
                    m_f_xgb.fit(X_train, y_train[TARGET_FUEL])
                    p_f = (
                        0.50 * np.clip(m_f_lgb.predict(X_val), 0, None)
                        + 0.30 * np.clip(m_f_cb.predict(X_val), 0, None)
                        + 0.20 * np.clip(m_f_xgb.predict(X_val), 0, None)
                    )
                    f_preds.append(p_f)
                else:
                    # Model Best 2 Fuel
                    m_f_cb = CatBoostRegressor(
                        iterations=250,
                        learning_rate=0.04,
                        random_seed=s,
                        verbose=0,
                        thread_count=-1,
                    )
                    m_f_cb.fit(X_train, y_train[TARGET_FUEL])
                    f_preds.append(np.clip(m_f_cb.predict(X_val), 0, None))

                # Nonfuel
                m_nf_cb = CatBoostRegressor(
                    iterations=250,
                    learning_rate=0.04,
                    random_seed=s,
                    verbose=0,
                    thread_count=-1,
                )
                m_nf_lgb = lgb.LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.03,
                    random_state=s,
                    n_jobs=-1,
                    verbose=-1,
                )
                m_nf_cb.fit(X_train, y_train[TARGET_NONFUEL])
                m_nf_lgb.fit(X_train, y_train[TARGET_NONFUEL])
                p_nf = 0.60 * np.clip(m_nf_cb.predict(X_val), 0, None) + 0.40 * np.clip(
                    m_nf_lgb.predict(X_val), 0, None
                )
                nf_preds.append(p_nf)

                # Opportunity
                clf = lgb.LGBMClassifier(
                    n_estimators=120,
                    learning_rate=0.04,
                    random_state=s,
                    n_jobs=-1,
                    verbose=-1,
                )
                clf.fit(X_train, y_train[TARGET_OPPORTUNITY].astype(str))
                classes = clf.classes_
                opp_probs_list.append(clf.predict_proba(X_val))

            pred_f = np.mean(f_preds, axis=0)
            pred_nf = np.mean(nf_preds, axis=0)
            avg_probs = np.mean(opp_probs_list, axis=0)
            bias_vec = np.array(
                [DEFAULT_OPP_BIASES.get(c, 0.0) for c in classes], dtype=float
            )

            if is_model_best2:
                pred_opp = classes[
                    (np.log(np.clip(avg_probs, 1e-9, 1.0)) + bias_vec).argmax(axis=1)
                ]
            else:
                # Model Best 1: Consensus Majority Vote
                p_bagged = classes[
                    (np.log(np.clip(avg_probs, 1e-9, 1.0)) + bias_vec).argmax(axis=1)
                ]
                p_seed42 = classes[
                    (np.log(np.clip(opp_probs_list[0], 1e-9, 1.0)) + bias_vec).argmax(
                        axis=1
                    )
                ]
                # consensus
                pred_opp_list = []
                for i in range(len(X_val)):
                    counts = Counter([p_bagged[i], p_seed42[i]])
                    top_c, _ = counts.most_common(1)[0]
                    pred_opp_list.append(top_c)
                pred_opp = np.array(pred_opp_list)

        df_pred = pd.DataFrame(
            {
                TARGET_FUEL: pred_f,
                TARGET_NONFUEL: pred_nf,
                TARGET_OPPORTUNITY: pred_opp,
            }
        )

        metrics = compute_competition_metrics(y_val, df_pred)
        fold_res = FoldResult(
            val_cutoff=val_cutoff,
            n_train_rows=n_train,
            n_val_rows=n_val,
            rmse_fuel=metrics["rmse_fuel"],
            rmse_nonfuel=metrics["rmse_nonfuel"],
            f1_opportunity=metrics["weighted_f1_opportunity"],
            lb_score=metrics["lb_score"],
            y_pred=df_pred,
        )
        fold_results.append(fold_res)
        print(
            f"    Fold {val_cutoff} Result ({time.time() - t0:.1f}s): "
            f"Fuel RMSE={metrics['rmse_fuel']:.5f} | "
            f"Nonfuel RMSE={metrics['rmse_nonfuel']:.5f} | "
            f"Opp F1={metrics['weighted_f1_opportunity']:.5f} | "
            f"Composite Score={metrics['lb_score']:.5f}"
        )

    return summarize_validation_results(fold_results, model_name=pipeline_name)


if __name__ == "__main__":
    t_start = time.time()
    results = []
    # Benchmark Baseline, Model Best 2 (Single-Seed), and Model Best 1 (Synthesis)
    for pipe in ["baseline", "model_best2", "model_best1"]:
        df_res = evaluate_embargoed_pipeline(pipe)
        results.append(df_res)

    df_all = pd.concat(results, ignore_index=True)
    print("\n" + "=" * 90)
    print("MASTER EMBARGOED VALIDATION BENCHMARK SUMMARY")
    print("=" * 90)
    print(df_all[df_all["Val Cutoff"] == "MEAN"].to_string(index=False))
    print("=" * 90)
    print(f"Complete benchmark finished in {time.time() - t_start:.1f}s")
