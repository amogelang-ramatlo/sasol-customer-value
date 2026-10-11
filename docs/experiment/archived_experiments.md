# Archived Experiments & Alternative Model Explorations

> **Repository:** `sasol-customer-value`  
> **Status:** Historical Reference & Reproducibility Archive  
> **Authoritative Selected Models:**  
> 1. **Primary Flagship:** `sub_v14_all_star_synthesis.csv` (Proven Live LB: **`0.311437`**)  
> 2. **Secondary Single-Seed:** `sub_v13_single_seed.csv` (Clean Seed 42, 225 features)  

---

## 1. Executive Summary

During development, 18 model configurations and 3 architectural research hypotheses were systematically implemented, validated against the leak-free rolling-origin Out-of-Time (OOT) evaluation protocol, and scored on the Zindi live leaderboard.

This document preserves the implementation details, configurations, evaluation metrics, limitations, and reproduction instructions for all non-selected models to ensure complete scientific reproducibility without cluttering the active codebase.

---

## 2. Comprehensive Inventory of Archived Models

### 2.1 Baseline Model (`sub_baseline.csv`)
* **Purpose:** Establish the minimum performance benchmark on single-snapshot data.
* **Archive Reference:** `src/sasol_customer_value/models/baseline.py`
* **Features:** 84 basic RFM attributes extracted from a single snapshot (`2025-09-01`).
* **Architecture & Hyperparameters:** Single `LGBMRegressor` for fuel/non-fuel (`n_estimators=100, lr=0.05`), single `LGBMClassifier` for Opportunity.
* **Recorded Metrics:** Local Fuel RMSE: `0.6165`, Nonfuel RMSE: `0.7524`, Opp F1: `46.38%`, Public LB: `~0.285`.
* **Why Not Selected:** Severe underfitting; lacked seasonal lookbacks and calibrated class odds.

---

### 2.2 Version 3: Seasonal Regressors (`sub_v3_seasonal.csv`)
* **Purpose:** Test 1-year historical seasonal lookback features for spend forecasting.
* **Archive Reference:** `scripts/archive/reproduce_v11_decoupled.py`
* **Features:** 88 features (RFM + 1-year same-season lookback).
* **Architecture:** 3-model weighted blend (0.50 LGB / 0.30 CB / 0.20 XGB).
* **Recorded Metrics:** Live LB Nonfuel RMSE reached all-time best **`0.72776`**, but Opportunity Weighted F1 remained uncalibrated at `48.64%`. Public LB: `0.29601`.
* **Why Not Selected:** Weak Opportunity performance dragged down composite competition loss.

---

### 2.3 Version 4: Frequency-Calibrated Ensemble (`sub_v4_calibrated_ensemble.csv`)
* **Purpose:** Introduce log-odds frequency calibration to resolve extreme class imbalance in Opportunity.
* **Archive Reference:** Retained in `submissions/sub_v4_calibrated_ensemble.csv`
* **Features:** 202 features (v4 feature store).
* **Architecture:** 
  - Regressors: Blend (0.50 LGB / 0.30 CB / 0.20 XGB) on Seed 42.
  - Classifier: `LGBMClassifier` (120 trees, lr=0.04) + frequency-calibrated log-odds biases.
* **Recorded Metrics:** Live LB Fuel RMSE reached all-time best **`0.589218648`**. Opp F1: `0.536735`, Public LB: `0.30922`.
* **Contribution to Selected Model:** Model A's Fuel predictions were permanently selected as the Fuel component of Flagship `v14`.

---

### 2.4 Version 5 & 6: CatBoost-Heavy & Winter/Spring Biases (`sub_v5`, `sub_v6`)
* **Purpose:** Increase CatBoost blending weight to regularize spend predictions.
* **Archive Reference:** `scripts/archive/reproduce_v10_no_lightgbm.py`
* **Features:** 218 features (v5 feature store).
* **Recorded Metrics:** Public LB: `0.30670`.
* **Why Not Selected:** Calibrating decision biases jointly on winter and spring snapshots over-penalized convenience store categories during the South African December summer holiday period.

---

### 2.5 Version 7: Summer Holiday Calibration (`sub_v7_summer_calibrated.csv`)
* **Purpose:** Align opportunity class biases with the South African December–February festive travel surge.
* **Archive Reference:** `scripts/archive/build_sub_v17_reseeded_v14.py`
* **Features:** 218 features.
* **Recorded Metrics:** Public LB: `0.30979`.
* **Discovery:** In summer, Inactivity drops from 27.4% to 16.3%, while Fuel and convenience store adoption (`Beverages`, `Food & beverage`) surge. Summer-specific calibration lifted holiday F1 from `42.42%` to `44.67%`.

---

### 2.6 Version 8: 5-Seed Bagged Tri-Model Pipeline (`sub_v8_multiseed_ensemble.csv`)
* **Purpose:** Multi-seed bagging across 5 seeds (`[42, 100, 2024, 777, 999]`) for variance reduction.
* **Archive Reference:** Retained in `submissions/sub_v8_multiseed_ensemble.csv`
* **Features:** 218 features across all 15 historical snapshots.
* **Recorded Metrics:** Live LB Nonfuel RMSE reached all-time best **`0.724033644`**. Opp F1: `0.538125`, Public LB: `0.30994`.
* **Contribution to Selected Model:** Model B's Nonfuel spend was permanently selected as the Nonfuel component of Flagship `v14`.

---

### 2.7 Version 9: Shopper Personas Multi-Seed (`sub_v9_shopper_personas.csv`)
* **Purpose:** Introduce explicit customer segmentation features (`is_fuel_only_buyer`, `is_store_only_buyer`, `is_dual_buyer`).
* **Archive Reference:** `src/sasol_customer_value/features/behavioral.py` (lines 363–385)
* **Features:** 225 features (v6 feature store).
* **Recorded Metrics:** Local validation F1 jumped to `51.92%`. Live LB Opp F1: `0.53509`.
* **Contribution to Selected Model:** Serves as Model C in the `v13`/`v14` consensus voting committee.

---

### 2.8 Version 10: Ablation Eliminating LightGBM (`sub_v10_no_lightgbm.csv`)
* **Purpose:** Test whether pure CatBoost + XGBoost could replace LightGBM for classification.
* **Archive Reference:** `scripts/archive/reproduce_v10_no_lightgbm.py`
* **Recorded Metrics:** Local Opp F1 dropped sharply to `51.17%`.
* **Lesson Learned:** LightGBM's leaf-wise tree growth is essential for calibrated multiclass output in this domain; oblivious/symmetric trees alone underperformed.

---

### 2.9 Version 12: Softmax Probability Blending (`sub_v12_3way_ensemble_opp.csv`)
* **Purpose:** Blend raw class probabilities from LightGBM, CatBoost, and XGBoost via arithmetic mean.
* **Archive Reference:** `scripts/archive/build_sub_v12_3way_ensemble.py`
* **Recorded Metrics:** Public LB dropped to `0.30056` (Opp F1: `0.5256`).
* **Why Not Selected:** Soft probability averaging increased entropy and blurred minority class boundaries (`Food & beverage` collapsed by >50%). Discrete majority voting was proven superior.

---

### 2.10 Version 13: Committee Consensus Ensemble (`sub_v13_consensus_ensemble.csv`)
* **Purpose:** Replace soft probability averaging with a discrete majority vote among top validated models (v8, v4, v9).
* **Archive Reference:** `scripts/archive/build_sub_v13_consensus_ensemble.py`
* **Recorded Metrics:** Live LB Opportunity F1 reached all-time high **`0.541244734`**. Overall LB: `0.31119`.
* **Contribution to Selected Model:** Formed the Opportunity classifier component for Flagship `v14`.

---

### 2.11 Version 15: Recency-Gated Consensus (`sub_v15_recency_calibrated.csv`)
* **Purpose:** Force customers with recent visits (<45 days) out of `Inactivity` into active classes.
* **Archive Reference:** `scripts/archive/build_sub_v15_recency_calibrated.py`
* **Recorded Metrics:** Validation F1 declined by `-0.15%`, Public LB dropped to `0.31098`.
* **Why Not Selected:** Hardcoded heuristics introduced label noise on active customers who reduced spend without lapsing.

---

### 2.12 Version 16: Two-Stage Hurdle Blend (`sub_v16_hurdle_blend.csv`)
* **Purpose:** Test two-stage hurdle regression (zero-spend classification + positive spend regression) with uniform macro-group bias tuning.
* **Archive Reference:** `scripts/archive/build_sub_v16_hurdle_blend.py`, `scripts/archive/benchmark_hurdle_experiment.py`
* **Recorded Metrics:** Public LB dropped to `0.30539` (Opp F1: `0.5277`).
* **Why Not Selected:** Uniform macro-group biases starved specific convenience categories, causing `Beverages` and `Food & beverage` to collapse into `Other`.

---

### 2.13 Version 17: Reseeded Replica (`sub_v17_v14_reseeded.csv`)
* **Purpose:** Test seed sensitivity and variance of `v14` by rerunning the entire architecture on completely new seeds `[101, 202, 303, 404, 505]`.
* **Archive Reference:** `scripts/archive/build_sub_v17_reseeded_v14.py`
* **Recorded Metrics:** Pearson correlation with `v14` was $r > 0.9999$. Confirmed that `v14` is numerically stable and invariant to random seed choices.

---

### 2.14 Version 18: Multi-Family Consensus Committee (`sub_v18_single_seed.csv`, `sub_v18_multiseed.csv`)
* **Purpose:** Committee voting among LightGBM, CatBoost, and XGBoost multiclass classifiers.
* **Archive Reference:** `scripts/archive/build_v18_multifamily_committee.py`, `scripts/archive/experiment_c_multifamily_committee.py`
* **Recorded Metrics:** 
  - Local validation F1 jumped from `0.50970` to `0.51382` (+0.00412).
  - Live LB dropped to `0.30739` (Opp F1: `0.53113`).
* **Why Not Selected:** While CatBoost and XGBoost performed well on the winter September validation split, their symmetric/depth-wise structures under-predicted summer convenience store adoption compared to Summer-Calibrated LightGBM.

---

## 3. Dedicated Hypothesis Experiments

| Experiment | Hypothesis | Outcome | Key Finding |
| :--- | :--- | :---: | :--- |
| **Experiment A (`cascaded_spend.py`)** | Passing predicted fuel and store spend as features into tree classifiers would improve Opportunity scoring. | **FAILED** (F1 dropped from `0.50970` to `0.50608`) | Regressed spend features introduced noisy threshold splits that fragmented decision boundaries. |
| **Experiment B (`hierarchical.py`)** | Two-stage hierarchical classification (Stage 1: Inactivity vs Active; Stage 2: 16 Active classes). | **MARGINAL** (F1 `0.51043`, +0.00073) | Stage 1 false-positive inactivity errors starved Stage 2 models of true active candidates. |
| **Experiment C (`multifamily_committee.py`)** | Combining leaf-wise (LGBM), symmetric (CatBoost), and depth-wise (XGBoost) inductive biases. | **LOCAL GAIN / LIVE DROP** | Locally superior on winter cohort, but inferior to summer-tuned LightGBM on the test window. |

---

## 4. How to Reproduce Any Archived Experiment

All archived scripts reside in `scripts/archive/`. To reproduce any archived run:

```powershell
# Reproduce multi-family committee (v18):
uv run python scripts/archive/build_v18_multifamily_committee.py

# Reproduce two-stage hurdle experiment (v16):
uv run python scripts/archive/build_sub_v16_hurdle_blend.py

# Reproduce hypothesis Experiment A (Cascaded Spend):
uv run python scripts/archive/experiment_a_cascaded_spend.py
```
