# Sasol Loyalty Customer Value & Opportunity Prediction — Master Competition Handoff

> **Document Status:** Authoritative Master Handoff Specification  
> **Repository:** `sasol-customer-value`  
> **Target Audience:** Incoming Principal AI / Claude Systems Engineer  
> **Current Champion Submission:** `submissions/sub_v3_seasonal.csv` (Local OOT Fuel RMSE: `0.60189`, Non-Fuel RMSE: `0.74288`, Opp Weighted F1: `46.71%`)  
> **Execution Environment:** Windows PowerShell, Python `3.12.x` pinned via `.python-version`, managed via `uv`  

---

## 1. Executive Summary & Objective for the Next Agent

This document is the **Single Source of Truth (SSOT)** for the Sasol Loyalty Customer Value & Opportunity Prediction competition on Zindi. It is designed to allow a top-tier Claude coding agent to take over immediately, understand the entire data science challenge, inspect all empirical findings, avoid failed avenues, and execute high-impact improvements autonomously.

### Core Mandate
1. **Understand the Mechanics:** Absorb the mathematical formulations, multi-metric objective function, receipt-level schema traps, and domain business logic.
2. **Respect the Anti-Leakage Contract:** Maintain strict temporal integrity ($t < t_{\text{cutoff}}$). Cross-sectional random shuffling across time is strictly prohibited.
3. **Build upon Verified Evidence:** Do not guess or repeat failed experiments (e.g., hurdle zero-inflation, hard zero-thresholding, naive probability reweighting, standalone neural nets).
4. **Push the Performance Frontier:** Leverage proper Out-Of-Fold (OOF) stacking, advanced feature engineering, station-level behavioral graph mining, and direct metric-targeted threshold optimization to drive the composite score towards the top of the leaderboard (~`0.30 - 0.32`).

---

## 2. Competition Overview, Constraints & Mathematical Formulation

### 2.1 The Business Problem
Sasol operates extensive retail fuel service stations and convenience forecourts (Sasol Delight) across South Africa. Loyalty customers scan digital or physical identifiers at checkout till slips.

We are provided with historical, receipt-level Point-Of-Sale (POS) transaction streams (`data/raw/train.csv`, ~467,880 rows from March 2024 to November 2025). For **5,488 target customers** listed in `data/raw/test.csv`, we must forecast customer activity over a future **3-month outcome window** (December 1, 2025 – February 28, 2026).

### 2.2 The Three Prediction Targets

| Target | Nature | Scope / Business Definition | Scaling & Normalization | Evaluation Metric | Weight |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`CLV_fuel`** | Continuous Regression | Total litres of fuel purchased (`LineItemType == 'Sale_Fuel'` AND `ItemCategoryLevel2 == 'Fuel'`). | $z_{\text{fuel}} = \frac{\ln(1 + y_{\text{litres}})}{s_{\text{fuel}}}$, where $s_{\text{fuel}} = 2.352825386185113$ | Root Mean Squared Error (RMSE) on $z_{\text{fuel}}$ | **0.30** |
| **`CLV_nonfuel`** | Continuous Regression | Total store merchandise spend in ZAR (`TotalExclAmount`), excluding `Fuel`, `Carrier Bags`, `Print media`, and `Value added services`. | $z_{\text{nonfuel}} = \frac{\ln(1 + y_{\text{rands}})}{s_{\text{nonfuel}}}$, where $s_{\text{nonfuel}} = 2.4200904988927476$ | Root Mean Squared Error (RMSE) on $z_{\text{nonfuel}}$ | **0.30** |
| **`Opportunity`** | 17-Class Mutually Exclusive Classification | Forecasts the customer's behavioral transition across 17 distinct classes (2 non-growth, 4 growth, 11 adoption). | Discrete label string (exact string matching required). | Multi-class Weighted F1 Score ($F_{1,\text{weighted}}$) | **0.40** |

### 2.3 Composite Objective Function
The official competition evaluation metric is:
$$\text{Score} = 0.30 \times \text{RMSE}\left(z_{\text{fuel}}, \hat{z}_{\text{fuel}}\right) + 0.30 \times \text{RMSE}\left(z_{\text{nonfuel}}, \hat{z}_{\text{nonfuel}}\right) + 0.40 \times F_{1,\text{weighted}}\left(y_{\text{opp}}, \hat{y}_{\text{opp}}\right)$$

*Note on Metric Optimization:* RMSE is minimized (lower is better), while Weighted F1 is maximized (higher is better). For loss minimization, internal optimization tracks:
$$\mathcal{L}_{\text{comp}} = 0.30 \times \text{RMSE}_{\text{fuel}} + 0.30 \times \text{RMSE}_{\text{nonfuel}} + 0.40 \times \left(1 - F_{1,\text{weighted}}\right)$$

### 2.4 The 17 Authoritative Opportunity Classes
The 17 labels configured in `docs/competition/label_config.json` are:
1. `Inactivity` (No qualifying in-scope purchases during the outcome window)
2. `Stable` (Qualifying activity present, but no category qualifies for growth or adoption; includes declining spend)
3. **Existing-Category Growth (4 classes):**
   * `Existing-category growth: Beverages`
   * `Existing-category growth: Food & beverage`
   * `Existing-category growth: Fuel`
   * `Existing-category growth: Other`
4. **New-Category Adoption (11 classes):**
   * `New category adoption: Beverages`
   * `New category adoption: Braai & Ice`
   * `New category adoption: Breads & rolls`
   * `New category adoption: Confectionary`
   * `New category adoption: Food & beverage`
   * `New category adoption: Groceries`
   * `New category adoption: Ice-cream`
   * `New category adoption: Lubricants`
   * `New category adoption: Other`
   * `New category adoption: Snacking`
   * `New category adoption: Tobacco`

---

## 3. Critical Domain Rules & Data Integrity Traps

Violation of any of these domain rules will corrupt the ground truth labels and guarantee local-vs-leaderboard divergence.

```
                          ┌──────────────────────────────────────┐
                          │   Raw Till Line Item Transaction     │
                          └──────────────────┬───────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ Excluded Items ]                           [ In-Scope Lines ]
      • Carrier Bags                               • Fuel: LineItemType == 'Sale_Fuel'
      • Print media                                • Store: Pre-tax TotalExclAmount
      • Value added services                       • Category 'NA' strictly preserved
      • Promo/Rounding Adjustments
```

### 3.1 The Literal `"NA"` String Trap (DO NOT PARSE AS NAN)
* In `ItemCategoryLevel2`, the string `"NA"` denotes a legitimate store department (general merchandise).
* **Rule:** You must load transactions with `keep_default_na=False, na_values=["", "#N/A", "null"]` (already implemented in `sasol_customer_value.data.loader.load_transactions`). Spending under `"NA"` contributes to `CLV_nonfuel` and maps to `Other`.

### 3.2 The Repeated Basket Total Trap
* `TransactionTotalInclAmount` represents the total basket amount and is duplicated across every line item of that `TransactionId`.
* **Rule:** **NEVER sum `TransactionTotalInclAmount`.** Customer spend is strictly calculated from signed line-level `TotalExclAmount`.

### 3.3 Eligibility Threshold
* A customer is eligible for opportunity scoring and target calculation if and only if they have accumulated **$\ge 3$ distinct historical `TransactionId` values** strictly prior to the snapshot cutoff ($t < t_{\text{cutoff}}$).

### 3.4 Opportunity Resolution Hierarchy & Tie-Breaking
1. **Growth Criteria:**
   * Customer must have at least one purchase in that category prior to cutoff.
   * At least one purchase in the outcome window.
   * Net spend increase $\ge \text{R}20.00$ over the preceding 3-month baseline window.
   * Outcome window spend $\ge 1.25 \times$ baseline window spend.
2. **Adoption Criteria:**
   * Zero purchases in that category across the customer's **entire observed history** prior to cutoff.
   * At least one purchase in the outcome window with strictly positive spend ($> 0$).
3. **Winner Resolution:**
   * Candidates compete based on **Net Rand Increment** (for Fuel, increment is computed from `TotalExclAmount`, not litres).
   * Highest increment wins.
   * **Tie-Breaker 1:** Adoption beats Growth if increments are identical.
   * **Tie-Breaker 2:** Alphabetical ordering by original raw category name.
   * **Mapping:** Categories compete individually under raw names before unmapped classes collapse into `Other`.
   * **Fixed-Point Precision:** Calculations enforce 6-decimal fixed-point precision (`Decimal` or scaled integers `SCALE = 1_000_000`) to guarantee deterministic tie-breaking without floating-point IEEE-754 drift.

### 3.5 Competition Restrictions
* **AutoML Ban:** Auto-Sklearn, TPOT, H2O Driverless, AutoKeras, etc., are strictly prohibited by Zindi rules. All pipelines, feature interactions, and ensembles must be explicitly written.
* **Submission Format:** Exactly 5,488 rows matching `test.csv` IDs in order, four columns (`ID`, `CLV_fuel`, `CLV_nonfuel`, `Opportunity`), zero nulls, non-negative values.

---

## 4. Repository Architecture & Codebase Map

The project adheres to a clean, tested package layout:

```text
sasol-customer-value/
│
├── .python-version                 # Python 3.12 pinned
├── pyproject.toml                  # Central dependencies & tool configs (uv)
├── uv.lock                         # Cryptographic lockfile
├── README.md                       # High-level overview
├── GEMINI.md                       # Architecture & philosophy guidelines
├── COMPETITION_HANDOFF.md          # THIS FILE (Authoritative master handoff)
│
├── docs/
│   ├── PLAIN_ENGLISH_GUIDE.md      # Layman guide to retail receipt formulations
│   └── competition/                # Immutable competition assets (read-only SSOT)
│       ├── DATA_DICTIONARY.md
│       ├── LABEL_RULES.md
│       ├── challenge_overview_and_getting_started_guide.md
│       ├── dataset_and_task_overview.md
│       ├── label_config.json       # Authoritative configuration JSON
│       └── label_rules.py          # Reference label implementation
│
├── data/
│   ├── raw/                        # Immutable inputs: train.csv (71.8MB), test.csv, SampleSubmission.csv
│   └── processed/                  # Derived artifacts (val_labels_20250901.parquet)
│
├── notebooks/                      # Scratchpads & visual EDA ONLY (no production logic)
│   ├── 01_data_audit_and_validation.ipynb
│   ├── 02_target_analysis.ipynb
│   └── 03_in_depth_eda_and_model_insights.ipynb  # Comprehensive EDA with visual figures
│
├── reports/
│   └── figures/                    # High-resolution persisted diagnostic plots (PNG)
│       ├── fig1_target_distributions.png
│       ├── fig2_opportunity_class_imbalance.png
│       ├── fig3_temporal_trends_and_seasonality.png
│       ├── fig4_fuel_vs_nonfuel_joint_dynamics.png
│       └── fig5_test_predictions_diagnostic.png
│
├── src/
│   └── sasol_customer_value/       # Typed, tested production Python package
│       ├── __init__.py
│       ├── config.py               # SSOT loader: parses label_config.json, constants & paths
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py           # Safe CSV reading ('NA' preserved, typed columns)
│       │   └── labels.py           # Fixed-point ground truth target generator
│       ├── features/
│       │   ├── __init__.py
│       │   └── rfm.py              # 88 leak-free RFM, cadence, & 1-year seasonal features
│       ├── models/
│       │   ├── __init__.py
│       │   ├── baseline.py         # Baseline LightGBM pipeline
│       │   ├── ensemble.py         # Stacked Triple-GBDT (LGBM + CatBoost + XGBoost)
│       │   ├── metrics.py          # Exact competition loss evaluator
│       │   └── predict.py          # CLI training, evaluation & submission generator
│       └── utils/
│           ├── __init__.py
│           └── helpers.py          # Seeding, format validation & submission verifier
│
├── tests/                          # Automated test suite (11 passing tests)
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_labels.py
│   ├── test_metrics.py
│   └── test_smoke_imports.py
│
└── submissions/                    # Generated submission CSVs (100% compliant)
    ├── sub_baseline.csv            # Baseline single-snapshot LightGBM
    ├── sub_ensemble_v2.csv         # Multi-model stacked ensemble
    └── sub_v3_seasonal.csv         # CURRENT CHAMPION: Ensemble + Seasonal Lookback
```

---

## 5. Summary of Everything Attempted & Results Log

### 5.1 Progression of Validated Benchmarks (Local OOT Validation at Cutoff `2025-09-01`)
*Cohort size: 5,238 active eligible customers.*

| Iteration / Model Strategy | Fuel RMSE (30%) ↓ | Non-Fuel RMSE (30%) ↓ | Opportunity F1 (40%) ↑ | Official Composite Metric | True Loss ($1 - F_1$) ↓ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0. Dummy Baseline (All Zeros + "Stable")** | `1.83449` | `1.22406` | `11.33%` | `0.96289` | `1.2722` | Reference Floor |
| **1. Mean Value Baseline** | `1.06202` | `1.00055` | `11.33%` | `0.66410` | `0.9735` | Reference Floor |
| **2. Single-Snapshot LightGBM Baseline** | `0.61645` | `0.75241` | `46.38%` | `0.59836` | `0.6251` | Phase 3 Baseline |
| **3. Stacked 3-Snapshot Multi-Model Ensemble** | `0.60221` | `0.74264` | `46.38%` | `0.58898` | `0.6179` | Multi-Model Stack |
| **4. Ensemble + 1-Year Seasonal Lookback** | **`0.60189`** | **`0.74288`** | **`46.71%`** | **`0.59029`** | **`0.6162`** | **CURRENT CHAMPION (`sub_v3_seasonal.csv`)** |

*Note on Metric Reading:* In the official metric formula ($0.3\text{RMSE} + 0.3\text{RMSE} + 0.4F_1$), an increase in F1 from 46.38% to 46.71% raises the metric by $+0.0013$. Under the true loss minimization formulation ($0.3\text{RMSE} + 0.3\text{RMSE} + 0.4(1 - F_1)$), Iteration 4 improves the loss from `0.6179` to `0.6162`.

---

### 5.2 Negative Experiments Log (What Failed & Why — DO NOT REPEAT)

To avoid wasted compute and dead ends, the following approaches were rigorously evaluated and found to degrade performance:

#### 1. Two-Stage Hurdle (Zero-Inflation) Models ❌ FAILED
* **Hypothesis:** Split regression into a binary classifier (*"Will customer spend?"*) multiplied by a conditional regressor (*"If active, how much?"*).
* **Result:** Increased fuel RMSE from `0.602` to `0.606`.
* **Root Cause:** The target $z = \ln(1 + y) / s$ is already log-transformed, which naturally compresses zeros without massive right-tail distortion. GBDT regression trees minimizing MSE directly predict the mathematically optimal unconditional expectation $\mathbb{E}[z \mid X]$. Multiplying an uncalibrated classifier probability by a conditional regressor compounds estimation variance at the boundary.

#### 2. Post-Hoc Hard Zero-Thresholding ❌ FAILED
* **Hypothesis:** For customers with low predicted spend ($z < 0.10, 0.20$), snap predictions to exact zero.
* **Result:** Degraded non-fuel RMSE from `0.74732` to `0.75848`.
* **Root Cause:** Under squared error loss (RMSE), predicting a soft expectation (e.g., $0.20$) for an uncertain customer yields strictly lower expected loss than hard-snapping to $0.0$:
  $$\mathbb{E}[(0.2 - y)^2] = 0.8(0.04) + 0.2(0.64) = 0.160 \quad < \quad \mathbb{E}[(0 - y)^2] = 0.8(0) + 0.2(1.0) = 0.200$$

#### 3. Naive In-Sample Feature Cascading ❌ FAILED
* **Hypothesis:** Pass predicted fuel spend $\hat{z}_{\text{fuel}}$ directly as an input feature into the store spend and Opportunity models.
* **Result:** Opportunity F1 dropped from `46.71%` down to `45.83%`.
* **Root Cause:** Generating Stage 1 predictions in-sample (on the training data) introduced training label leakage and distribution shift when evaluated on out-of-sample data. Feature cascading *requires* strict K-Fold Out-Of-Fold (OOF) feature generation to avoid this bias.

#### 4. Post-Hoc Probability Inverse-Frequency Multipliers ❌ FAILED
* **Hypothesis:** Multiply predicted class probabilities by inverse class frequency powers ($1 / p_c^\alpha$) to boost rare adoption categories.
* **Result:** Weighted F1 plummeted from `45.83%` down to `41.66%`.
* **Root Cause:** LightGBM was already trained with `class_weight='balanced'`. Applying a secondary inverse-frequency multiplier double-counted the minority class weights, causing catastrophic false-positive spikes on rare classes (`Braai & Ice`, `Confectionary`) and devastating overall precision.

#### 5. Standalone Tabular Neural Networks (MLP) ❌ FAILED
* **Hypothesis:** A Multi-Layer Perceptron (MLP) with RobustScaler and early stopping might capture non-linear patterns missed by trees.
* **Result:** Standalone Fuel RMSE was `0.69094` (~15% worse than GBDT `0.60187`). Blending even a 5%–10% weight of MLP into GBDT degraded the ensemble score (`0.60187` $\to$ `0.60248`).
* **Root Cause:** Tabular retail receipt data features sharp orthogonal step boundaries (minimum visit thresholds, spend limits) that trees split effortlessly, whereas smooth neural hyperplanes struggle without vast sample sizes.

---

### 5.3 Key Empirical EDA Insights (`notebooks/03_in_depth_eda_and_model_insights.ipynb`)

1. **The Four Forecourt Behavioral Archetypes:**
   * **Fuel Only:** **37.6%** (1,972 customers) — visit only the forecourt; never buy store merchandise.
   * **Dual Shoppers (Fuel + Store):** **33.1%** (1,734 customers) — buy both fuel and store items.
   * **Completely Inactive:** **27.4%** (1,436 customers) — zero purchases across both departments over the 3-month window.
   * **Store Only:** Only **1.8%** (96 customers) — almost no one visits a Sasol station *only* for convenience store merchandise without purchasing fuel.
   * *Strategic Implication:* Fuel is the master driver of foot traffic. If a customer has zero predicted fuel spend, store spend is almost certainly zero.
2. **Opportunity Class Distribution Imbalance:**
   * Three classes account for **81.6%** of all customer outcomes: `Inactivity` (27.4%), `Fuel Growth` (27.4%), and `Stable` (26.8%).
   * Single-model LightGBM over-predicted `Stable` at 33.3% (playing it safe), while our 3-model ensemble restored accurate balance across minority growth classes.
3. **Macro Seasonality:**
   * February exhibits a massive annual spike in convenience store spend (exceeding R8.3M).
   * Incorporating 1-year same-season lookback features captures annual holiday and back-to-school travel routines.

---

## 6. Current State & Immediate Execution Commands

### 6.1 Environment Sync & Verification
The project uses `uv` for deterministic, cryptographic package management:

```powershell
# 1. Sync virtual environment
uv sync --all-groups

# 2. Run automated test suite (Must report 11 passed)
uv run pytest -v tests/

# 3. Check code style and formatting
uv run ruff check src/ tests/

# 4. Run static type checking
uv run mypy src/
```

### 6.2 Reproducing the Champion Pipeline
To train the champion multi-model ensemble on historical snapshots (`2024-12-01`, `2025-03-01`, `2025-06-01`), evaluate on the OOT validation cohort (`2025-09-01`), and generate a compliant submission:

```powershell
# Run full ensemble pipeline
uv run python -m sasol_customer_value.models.predict --pipeline ensemble --output submissions/sub_v3_seasonal.csv

# Verify submission compliance (Checks row count 5488, columns, nulls, value boundaries)
uv run python -m sasol_customer_value.utils.helpers --verify-submission submissions/sub_v3_seasonal.csv
```

---

## 7. Instructions for the Next Claude Model: In-Depth Improvement Roadmap

The next Claude model must not simply tweak hyperparameters blindly. It should conduct a structured, hypothesis-driven investigation.

### 7.1 High-ROI Improvement Backlog

#### Priority 1: Proper Out-Of-Fold (OOF) Stacking & Cascading
* **The Opportunity:** Only 1.8% of customers buy store goods without fuel. Fuel volume is a powerful predictor for both store spend and opportunity class.
* **The Method:**
  1. Implement a 5-Fold temporal cross-validation loop on the stacked training snapshots.
  2. For each out-of-fold slice, predict $\hat{z}_{\text{fuel}}$.
  3. Feed the **OOF $\hat{z}_{\text{fuel}}$** as an input feature into the Stage 2 Non-Fuel models and Stage 3 Opportunity classifier.
  4. At test time, use the ensemble average of the 5 fold models. This completely prevents the in-sample distribution shift that doomed our earlier naive attempt!

#### Priority 2: Direct Threshold Optimization for Weighted F1 via Nelder-Mead / Optuna
* **The Opportunity:** The Opportunity classifier predicts 17 class probabilities $p_1, \dots, p_{17}$. Standard `argmax` picks the class with highest probability. Because classes have different base rates and different contributions to Weighted F1, an optimal threshold vector $w_1, \dots, w_{17}$ exists such that:
  $$\hat{y} = \arg\max_c \left( w_c \cdot p_c \right)$$
* **The Method:**
  1. Extract predicted probability matrix $P \in \mathbb{R}^{5238 \times 17}$ on the validation set.
  2. Use Scipy's `minimize(method='Nelder-Mead')` or Optuna to optimize the 17 weights $w$ to directly maximize `f1_score(y_true, y_pred, average='weighted')`.
  3. Validate on out-of-fold data to ensure threshold weights do not overfit.

#### Priority 3: Deep Forecourt Behavioral Feature Engineering
* **Fill-up Regularity & Inter-purchase Variance:**
  - Compute coefficient of variation of days between fill-ups: $\sigma_{\Delta t} / \mu_{\Delta t}$. Commuters have low variance; erratic drivers have high variance.
* **Fuel Volume per Visit:**
  - Litres per fill-up: $\text{Quantity}_{\text{fuel}} / \text{baskets}_{\text{fuel}}$. Differentiates small vehicle tanks (30L) from large SUVs/bakkies/taxis (75L+).
* **Station / Forecourt Loyalty:**
  - Percentage of visits at customer's primary station ID: $\max(\text{visits}_{\text{station}}) / \text{total\_visits}$. High loyalty customers churn less.
* **Pre-Holiday Spend Acceleration:**
  - Ratio of spend in the last 30 days vs preceding 60 days: $\text{spend}_{30\text{d}} / (\text{spend}_{90\text{d}} - \text{spend}_{30\text{d}} + \epsilon)$.

#### Priority 4: Objective Function Customization in LightGBM & CatBoost
* **Fuel / Non-fuel Custom Loss:**
  - Test Huber loss or Fair loss on $z$ to see if down-weighting extreme outliers in log-space improves RMSE on test data.
* **Opportunity Focal Loss / Multi-Class Log-Loss Tuning:**
  - Tune CatBoost's `MultiClass` with custom class priors.

---

## 8. Guardrails, Invariants & Submission Checklist

Every experiment and final submission deliverable must satisfy these non-negotiable checks:

- [ ] **Temporal Integrity:** Every feature engineering step must accept `cutoff_date` and execute `df[df['_time'] < cutoff_date]` as the very first line.
- [ ] **Literal `"NA"` Preservation:** Verify that `ItemCategoryLevel2 == 'NA'` is preserved as a string and mapped to `Other`.
- [ ] **No AutoML:** Do not introduce Auto-Sklearn, TPOT, AutoGluon, or cloud AutoML.
- [ ] **Deterministic Seeding:** Seed all GBDTs and NumPy operations using `sasol_customer_value.config.RANDOM_SEED = 42`.
- [ ] **Submission Schema Verification:**
  - Exactly **5,488 rows** in identical order as `data/raw/test.csv`.
  - Four columns: `ID`, `CLV_fuel`, `CLV_nonfuel`, `Opportunity`.
  - Zero `NaN`, `null`, or `inf` values.
  - `CLV_fuel >= 0.0` and `CLV_nonfuel >= 0.0`.
  - `Opportunity` values belong strictly to the 17 defined classes.
  - Run `uv run python -m sasol_customer_value.utils.helpers --verify-submission <path_to_submission>`.
- [ ] **No Direct Git Commits:** The user manages all git commits personally; do NOT execute `git commit`.

---

## 9. Quick Reference: Key Paths & Files

| Component | Absolute Path | Description |
| :--- | :--- | :--- |
| **SSOT Config** | `docs/competition/label_config.json` | Authoritative competition parameters & labels |
| **Config Loader** | `src/sasol_customer_value/config.py` | Python module exposing constants & paths |
| **Data Loader** | `src/sasol_customer_value/data/loader.py` | Safe CSV readers with "NA" preservation |
| **Ground Truth Generator** | `src/sasol_customer_value/data/labels.py` | Generates official OOT validation targets |
| **Feature Store** | `src/sasol_customer_value/features/rfm.py` | 88 temporal RFM, cadence & seasonal features |
| **Ensemble Model** | `src/sasol_customer_value/models/ensemble.py` | LightGBM + CatBoost + XGBoost pipeline |
| **Metrics Engine** | `src/sasol_customer_value/models/metrics.py` | Official competition metric calculator |
| **Inference CLI** | `src/sasol_customer_value/models/predict.py` | Train, evaluate, and generate submissions |
| **Submission Validator** | `src/sasol_customer_value/utils/helpers.py` | 6-point submission compliance verification |
| **Validation Parquet** | `data/processed/val_labels_20250901.parquet` | 5,238 ground-truth targets for OOT validation |
| **EDA Notebook** | `notebooks/03_in_depth_eda_and_model_insights.ipynb` | Comprehensive visual diagnostic notebook |
| **Champion Submission** | `submissions/sub_v3_seasonal.csv` | Best verified competition submission |
