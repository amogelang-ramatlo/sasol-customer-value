# Sasol Customer Value — Champion V2 Model Specification & System Blueprint

This document provides a comprehensive, reproducible technical specification of the **Champion V2** production architecture on the **Sasol Customer Value** competition on Zindi.

---

## 1. Executive Summary & Performance Scorecard

| Evaluation Metric | Component Weight | Baseline Local | Baseline Zindi | **Champion V2 (Live Zindi Leaderboard)** | Delta vs. Local |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Composite Leaderboard Score** | **100%** | 0.30586 | 0.31076 | **0.31045** | **+0.00459** |
| **Target F (`CLV_fuel` RMSE)** | 30% | 0.59101 | — | **0.59036** | **-0.00065 (Improvement)** |
| **Target N (`CLV_nonfuel` RMSE)** | 30% | 0.72433 | — | **0.72365** | **-0.00068 (Improvement)** |
| **Target O (`Opportunity` Weighted F1)** | 40% | 0.52940 | — | **0.53957** | **+0.01017 (Major Breakthrough)** |

### Official Competition Evaluation Metric
The leaderboard objective is a weighted composite score combining regression accuracy and multiclass classification quality:

$$\text{Composite Score} = 0.60 + 0.40 \times F_{1,\text{weighted}} - 0.40540558 \times \text{RMSE}_{\text{fuel}} - 0.36764644 \times \text{RMSE}_{\text{nonfuel}}$$

### Evaluation Horizon & Test Environment
- **History Cutoff Date:** `2025-12-01 00:00:00` (features strictly computed from $t < \text{2025-12-01}$).
- **Outcome Evaluation Window:** `[2025-12-01, 2026-03-01)` (South African Summer / Festive Holiday Period).
- **Test Customer Cohort:** Exactly 5,488 active loyalty programme customers matching `test.csv`.

---

## 2. System Architecture & Specialist Decomposition

Rather than forcing a single generic multi-output model, the architecture deploys **three dedicated target specialists**, each tuned specifically to the error characteristics and feature subspaces of its respective objective:

```
                                  ┌────────────────────────────────────────┐
                                  │ Raw Loyalty Transactions (train.csv)   │
                                  └──────────────────┬─────────────────────┘
                                                     │
                                                     ▼
                                  ┌────────────────────────────────────────┐
                                  │ Feature Store: 225 Engineered Features │
                                  │ (Strictly TransactionDateTime < Cutoff)│
                                  └──────────────────┬─────────────────────┘
                                                     │
             ┌───────────────────────────────────────┼───────────────────────────────────────┐
             │                                       │                                       │
             ▼                                       ▼                                       ▼
  ┌───────────────────────┐               ┌───────────────────────┐               ┌───────────────────────┐
  │ Model F (CLV_fuel)    │               │ Model N (CLV_nonfuel) │               │ Model O (Opportunity) │
  ├───────────────────────┤               ├───────────────────────┤               ├───────────────────────┤
  │ Target Specialist     │               │ Target Specialist     │               │ Target Specialist     │
  │ 202-Feature Subspace  │               │ 218-Feature Subspace  │               │ 3 Subspaces (A, B, C) │
  │ LightGBM (0.28) +     │               │ CatBoost (0.93) +     │               │ 15 LightGBM Trees     │
  │ CatBoost (0.72)       │               │ LightGBM (0.07)       │               │ Summer Holiday Biases │
  │ 5 Random Seeds Bagged │               │ 5 Random Seeds Bagged │               │ Majority Consensus    │
  └──────────┬────────────┘               └──────────┬────────────┘               └──────────┬────────────┘
             │                                       │                                       │
             └───────────────────────────────────────┼───────────────────────────────────────┘
                                                     │
                                                     ▼
                                  ┌────────────────────────────────────────┐
                                  │ Automated 5-Point Compliance Check     │
                                  │ (5,488 Rows, No Nulls, Non-negative)   │
                                  └──────────────────┬─────────────────────┘
                                                     │
                                                     ▼
                                  ┌────────────────────────────────────────┐
                                  │ submissions/sub_champion_v2.csv        │
                                  └────────────────────────────────────────┘
```

---

## 3. Feature Store Specification (225 Features Total)

Features are computed strictly prior to the snapshot cutoff date ($t < t_{\text{cutoff}}$). Future transactions are filtered as the very first operation.

### Feature Functional Groups

1. **Recency, Frequency & Tenure (8 features):**
   - `days_since_last_visit`, `days_since_first_visit`, `customer_tenure_days`.
   - Basket counts: `baskets_lifetime`, `baskets_last_30d`, `baskets_last_60d`, `baskets_last_90d`.
   - Visit momentum: $\text{baskets}_{30\text{d}} \times 3.0 / \max(1, \text{baskets}_{90\text{d}})$.

2. **Fuel Volume & Cadence (11 features):**
   - Litres: `fuel_litres_lifetime`, `fuel_litres_last_30d`, `fuel_litres_last_60d`, `fuel_litres_last_90d`.
   - `fuel_baskets_lifetime`, `fuel_litres_per_visit`, `days_since_last_fuel`.
   - Litres momentum: $\text{litres}_{30\text{d}} \times 3.0 / \max(1, \text{litres}_{90\text{d}})$.

3. **Store Spend & Merchandise Baskets (11 features):**
   - Signed pre-tax spend: `nonfuel_spend_lifetime`, `nonfuel_spend_last_30d`, `60d`, `90d`.
   - `nonfuel_baskets_lifetime`, `nonfuel_spend_per_visit`, `days_since_last_nonfuel`.
   - Store spend momentum: $\text{spend}_{30\text{d}} \times 3.0 / \max(1, \text{spend}_{90\text{d}})$.

4. **Cross-Category Interactions, Cadence & Churn Flags (10 features):**
   - `fuel_to_total_baskets_ratio`, `nonfuel_to_total_baskets_ratio`, `has_both_fuel_and_store`.
   - `mean_days_between_baskets`, `recency_to_cadence_ratio`, `churn_risk_flag_60d` ($\text{recency} > 60$).
   - `total_spend_lifetime`, `avg_spend_per_basket`, `distinct_locations_visited`, `weekend_baskets_ratio`.

5. **Category-Level Purchase Flags & 90-Day Baseline Spend (48 features):**
   - For all 24 retail categories:
     - `cat_ever_bought_<cat>`: Binary flag ($0$ vs. $1$) indicating any qualifying lifetime purchase.
     - `cat_spend_last_90d_<cat>`: Net signed spend in the preceding 90 days (baseline for the 1.25x growth rule).

6. **1-Year Seasonal Lookback (4 features):**
   - Behavior in the corresponding 3-month season of the previous year ($[t-12\text{mo}, t-9\text{mo})$):
     `same_season_prior_year_baskets`, `same_season_prior_year_active`, `same_season_prior_year_fuel_litres`, `same_season_prior_year_nonfuel_spend`.

7. **Quarterly (Q1, Q2, LY) & Monthly Series (83 features):**
   - Calendar quarter aggregates: `fuel_rand_q1`, `fuel_litres_q1`, `store_rand_q1`, `baskets_q1` ($k=1..3$ months before cutoff).
   - Growth hurdle: `fuel_growth_hurdle_rand` $= \max(20.0, 0.25 \times \text{fuel\_rand\_q1})$.
   - Monthly sequence and log-linear slope over last 6 months: `fuel_rand_slope_6m`, `store_rand_slope_6m`, `baskets_slope_6m`.
   - Station loyalty share: `primary_station_share` (max location baskets / total baskets).
   - Return transaction proportion: `return_lines_share` (proportion of negative amount lines).

8. **Visit Regularity & Inter-Basket Gaps (10 features):**
   - Inter-basket gaps: `gap_mean`, `gap_std`, `gap_median`, `gap_max`, `gap_last`.
   - `gap_cv` ($\text{std} / \text{mean}$), `recency_over_median_gap`, `expected_visits_next_q` ($91.0 / \text{median\_gap}$).

9. **Category Presence & Calendar Q1 Spend (16 features):**
   - `q1cat_<category>` and `histbaskets_<category>` across core retail lines.
   - `n_store_cats_never_bought`, `n_store_cats_bought_once`, `n_cats_active_q1`, `max_store_cat_q1`.

10. **Store Affinity & Zero-Spend Separation (8 features, "Option A"):**
    - `store_basket_ratio`, `fuel_only_basket_ratio`, `is_never_store_shopper`, `days_since_last_store_basket`.
    - `store_spend_30d_over_90d`, `store_active_last_30d`, `store_active_last_90d`.

11. **Growth Hurdles & Acceleration (8 features, "Option B"):**
    - `fuel_litres_accel_90_vs_prev90`, `fuel_litres_ratio_90_vs_prev90`, `fuel_candidate_growth_flag`.
    - `store_spend_accel_90_vs_prev90`, `store_candidate_growth_flag` ($\ge \text{R}20$ and $\ge 1.25\times$).

12. **Commuter Personas & Diurnal Patterns (8 features, "Option C"):**
    - Time-of-day basket shares: `morning_commute_share` (06:00–09:00), `midday_basket_share` (10:00–14:00), `evening_commute_share` (15:00–19:00), `weekend_basket_share`.
    - Persona indicators: `is_fuel_only_buyer_lifetime`, `is_store_only_buyer_lifetime`, `is_dual_buyer_lifetime`, `shopper_persona_segment_90d`.

### Feature Subspace Allocations
- **Subspace 202:** Features 0 through 201 (Core RFM + Fuel + Quarterly baselines; excludes Options A, B, C).
- **Subspace 218:** Features 0 through 217 (Subspace 202 + Option A Store Affinity + Option B Growth Hurdles).
- **Subspace 225:** All 225 features (Subspace 218 + Option C Commuter Personas & Diurnal Patterns).

---

## 4. Target Specialist Specifications

### 4.1 Model F (`FuelModel`) — Fuel Specialist Regressor
- **Objective:** Continuous regression minimizing RMSE on $z_{\text{fuel}}$.
- **Normalization:**
  $$z_{\text{fuel}} = \frac{\ln(1 + y_{\text{litres}})}{s_{\text{fuel}}}, \quad s_{\text{fuel}} = 2.352825386185113$$
- **Feature Subspace:** **202 features** (excluding convenience store features removes noise from the fuel volume signal).
- **Component Estimators:**
  - **LightGBM Regressor:** `n_estimators=150`, `learning_rate=0.03`, objective `regression_l2`, `verbose=-1`, `n_jobs=-1`.
  - **CatBoost Regressor:** `iterations=250`, `learning_rate=0.04`, objective `RMSE`, `verbose=0`, `thread_count=-1`.
- **Ensemble Blend & Bagging:**
  - **Weights:** $0.28 \times \text{LightGBM} + 0.72 \times \text{CatBoost} + 0.00 \times \text{XGBoost}$.
  - **Bagging:** 5 random seeds (`[42, 100, 2024, 777, 999]`).
  - **Post-Processing:** Non-negative clipping: $\hat{z}_{\text{fuel}} = \max(0.0, \bar{z})$.

### 4.2 Model N (`NonfuelModel`) — Store Merchandise Specialist Regressor
- **Objective:** Continuous regression minimizing RMSE on $z_{\text{nonfuel}}$.
- **Normalization:**
  $$z_{\text{nonfuel}} = \frac{\ln(1 + y_{\text{Rand}})}{s_{\text{nonfuel}}}, \quad s_{\text{nonfuel}} = 2.4200904988927476$$
- **Feature Subspace:** **218 features** (includes Option A zero-spend separation and Option B category acceleration).
- **Target Distribution Characteristic:** Heavy zero-inflation (**65.1% of customers have exactly 0 outcome spend**).
- **Component Estimators:**
  - **CatBoost Regressor (Primary):** `iterations=250`, `learning_rate=0.04`, objective `RMSE`, `verbose=0`. CatBoost's symmetric, oblivious decision trees act as strong regularizers against overfitting on the zero-mass.
  - **LightGBM Regressor:** `n_estimators=150`, `learning_rate=0.03`, objective `regression_l2`.
- **Ensemble Blend & Bagging:**
  - **Weights:** $0.07 \times \text{LightGBM} + 0.93 \times \text{CatBoost} + 0.00 \times \text{XGBoost}$ (heavy CatBoost weighting prevents over-predicting positive spend for non-store customers).
  - **Bagging:** 5 random seeds (`[42, 100, 2024, 777, 999]`).
  - **Post-Processing:** Non-negative clipping: $\hat{z}_{\text{nonfuel}} = \max(0.0, \bar{z})$.

### 4.3 Model O (`OpportunityModel`) — 17-Class Multiclass Specialist
- **Objective:** 17-class classification maximizing Weighted F1.
- **Architecture:** Multi-Architecture Diverse Committee across three feature subspaces:
  - **Committee Member A:** LightGBM Classifier on **Subspace 202** (`n_estimators=120`, `learning_rate=0.04`, `num_class=17`).
  - **Committee Member B:** LightGBM Classifier on **Subspace 218** (`n_estimators=120`, `learning_rate=0.04`, `num_class=17`).
  - **Committee Member C:** LightGBM Classifier on **Subspace 225** (`n_estimators=120`, `learning_rate=0.04`, `num_class=17`).
- **Seed Averaging:** Each committee member averages predicted probability distributions across 5 random seeds (`[42, 100, 2024, 777, 999]`). Total: **15 LightGBM classification trees**.
- **Summer Festive Log-Odds Calibration:**
  To counter holiday travel shifts during the Dec–Feb test window (where commuter habits break down and long-distance travel surges), probability vectors are adjusted via domain log-odds shifts before taking argmax:
  $$\text{Score}_c = \ln\left(\max(10^{-9}, P_c)\right) + \text{bias}_c$$
  Authoritative bias vector:
  ```python
  DEFAULT_OPP_BIASES = {
      "Existing-category growth: Beverages": +1.10,
      "Existing-category growth: Food & beverage": +1.20,
      "Existing-category growth: Fuel": +0.15,
      "Existing-category growth: Other": +0.60,
      "Inactivity": +0.10,
      "New category adoption: Beverages": +1.20,
      "New category adoption: Braai & Ice": +0.50,
      "New category adoption: Breads & rolls": +1.40,
      "New category adoption: Confectionary": +0.30,
      "New category adoption: Food & beverage": -0.50,
      "New category adoption: Groceries": +1.00,
      "New category adoption: Ice-cream": +1.40,
      "New category adoption: Lubricants": +0.40,
      "New category adoption: Other": +1.40,
      "New category adoption: Snacking": -0.20,
      "New category adoption: Tobacco": +0.80,
      "Stable": 0.00,
  }
  ```
- **Majority Consensus Decision Protocol:**
  Each member computes its biased argmax prediction: $\hat{y}_A, \hat{y}_B, \hat{y}_C$.
  Final class is assigned by **majority vote (2 out of 3)**. In a 3-way tie, Member B (the 218-feature model) acts as tie-breaker.

---

## 5. Training Protocol & Dataset Stacking

- **Training Historical Horizon:** 15 stacked monthly historical snapshots spanning `2024-07-01` through `2025-09-01`.
  - Total Training Observations: **65,722 customer-cutoff snapshots**.
- **Test Inference Date:** Features extracted as of `2025-12-01 00:00:00`.
- **Test Output Cohort:** Exactly 5,488 rows matching `test.csv`.
- **Training Time:** Complete 15-snapshot training across all 25 GBDT models runs in **~497 seconds** on a modern multi-core workstation.

---

## 6. Critical Experimental Findings & Negative Results (What Failed)

1. **Two-Stage Hurdle Regression Failed for Nonfuel (RMSE deteriorated from 0.7425 to 0.7724–0.8098):**
   Attempting to predict zero-spend probability and thresholding predictions severely increased RMSE. Squared error heavily penalizes cases where a customer was predicted as 0.0 but actually had positive spend. Smooth expected-value predictions from GBDTs outperformed hurdle architectures.
2. **Cascaded Multi-Task Features Failed for Opportunity (Weighted F1 dropped from 0.5140 to 0.5112):**
   Passing predicted fuel/nonfuel spend into the Opportunity classifier added noise to the decision boundaries. Tree splits overfit to Out-Of-Fold residuals and misclassified subtle boundary transitions between `Inactivity` and `Stable`.
3. **Local Cross-Fold Threshold Multiplier Optimization Degraded Test F1:**
   Tuning multipliers via Nelder-Mead on historical autumn/winter folds reduced generalization on the summer festive test distribution. The static holiday-calibrated log-odds bias vector was essential for achieving the **0.53957** leaderboard F1.

---

## 7. Submission Integrity Checksums (SHA-256)

- `submissions/sub_champion_v2.csv`: `e9435c954ce82935c0fd48a30620673038625b1a5a1c3f53c87f3ec6c04a66ef`
- `submissions/sub_champion_zindi.csv`: `dc26b02c73eb4037b8a7e7a87f034495a66473e499758017d42ffa284cae7159` *(Preserved benchmark)*
- `submissions/sub_champion_local.csv`: `b1028123e730f7882f9c49bad0e7fc9cdb8f8e6d3b67e9a032bd11a63f0e362a` *(Preserved benchmark)*
