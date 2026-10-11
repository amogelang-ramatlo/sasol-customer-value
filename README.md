# Sasol Loyalty Customer Value & Opportunity Prediction Platform

> **Project Identity:** Sasol Loyalty Customer Value & Opportunity Forecasting Engine  
> **Repository:** `sasol-customer-value`  
> **Package:** `sasol_customer_value` (`src/sasol_customer_value/`)  
> **Target Horizon:** 3-Month Forward Rolling Evaluation Window  
> **Evaluation Mode:** Local Out-Of-Time (OOT) Multi-Fold Rolling-Origin Validation  

---

## 1. Executive Summary & Business Context

Sasol operates retail fuel forecourts and convenience stores (Sasol Delight) across South Africa. Loyalty members present physical or digital identifiers at point-of-sale (POS) till slips across both forecourt fuel pumps and convenience store checkouts.

This repository implements an end-to-end, leak-free machine learning architecture to forecast customer value metrics and behavioral transitions over a forward 3-month window from granular, receipt-level point-of-sale transaction streams.

### Prediction Targets & Formulations
For each customer, the pipeline forecasts three distinct targets across the outcome horizon:

1. **`CLV_fuel` (Continuous Regression):**  
   Total litres of fuel purchased (`LineItemType == 'Sale_Fuel'` and `ItemCategoryLevel2 == 'Fuel'`).  
   Normalized target: $z_{\text{fuel}} = \frac{\ln(1 + y_{\text{litres}})}{s_{\text{fuel}}}$ where $s_{\text{fuel}} = 2.352825386185113$.  
   Evaluation Metric: Root Mean Squared Error (RMSE). Weight: **0.30**.

2. **`CLV_nonfuel` (Continuous Regression):**  
   Total pre-tax convenience store spend in ZAR (`TotalExclAmount`), excluding Fuel and non-merchandise categories (`Carrier Bags`, `Print media`, `Value added services`).  
   Normalized target: $z_{\text{nonfuel}} = \frac{\ln(1 + y_{\text{rands}})}{s_{\text{nonfuel}}}$ where $s_{\text{nonfuel}} = 2.4200904988927476$.  
   Evaluation Metric: Root Mean Squared Error (RMSE). Weight: **0.30**.

3. **`Opportunity` (17-Class Mutually Exclusive Classification):**  
   Categorizes customer retail progression or lapse during the horizon into exactly one of 17 mutually exclusive states:
   - Non-growth states: `Inactivity`, `Stable`
   - Existing-category growth (4 classes): `Beverages`, `Food & beverage`, `Fuel`, `Other`
   - New-category adoption (11 classes): `Beverages`, `Braai & Ice`, `Breads & rolls`, `Confectionary`, `Food & beverage`, `Groceries`, `Ice-cream`, `Lubricants`, `Other`, `Snacking`, `Tobacco`  
   Evaluation Metric: Multiclass Weighted F1 Score ($F_{1,\text{weighted}}$). Weight: **0.40**.

### Composite Objective Function
$$\text{Score} = 0.60 + 0.40 \times F_{1,\text{weighted}}(\text{Opportunity}) - 0.40540558 \times \text{RMSE}(\text{Fuel}) - 0.36764644 \times \text{RMSE}(\text{Nonfuel})$$

---

## 2. Model Architecture & Target Specialists (F, N, O)

Because the competition evaluation targets are mutually independent, the platform implements a clean **Target-Decoupled Architecture** powered by three dedicated target specialists:
- **Model F (`FuelModel`):** Specialist continuous regressor for `CLV_fuel` (normalized log fuel volume).
- **Model N (`NonfuelModel`):** Specialist continuous regressor for `CLV_nonfuel` (normalized log convenience store spend).
- **Model O (`OpportunityModel`):** Specialist multi-class committee classifier for `Opportunity` (17 mutually exclusive behavioral states).

### Champion Solution Profiles

#### 1. Zindi Leaderboard Champion: `sub_champion_zindi.csv`
* **Model F (Fuel Specialist):** Weighted blend (0.50 LightGBM / 0.30 CatBoost / 0.20 XGBoost) trained on the clean 202-feature forecourt subspace (Seed 42).
* **Model N (Store Spend Specialist):** Bagged ensemble (0.40 LightGBM / 0.60 CatBoost) trained on 218 features across 5 random seeds (`[42, 100, 2024, 777, 999]`). Multi-seed CatBoost regularization minimizes store spend residual error.
* **Model O (Opportunity Specialist):** 5-seed committee majority consensus across three distinct feature representations (Model A on 202 features, Model B on 218 features, Model C on 225 features) calibrated with South African summer holiday class log-odds biases.
* **Live Public Leaderboard Score:** **`0.31076` – `0.31144`** (Current Rank: **#26**).
* **Primary Output:** `submissions/sub_champion_zindi.csv` *(alias: `sub_model_best1.csv`)*.

#### 2. Local Validation Champion: `sub_champion_local.csv`
* **Model F (Fuel Specialist):** High-CatBoost blend (0.20 LightGBM / 0.55 CatBoost / 0.25 XGBoost) trained on 218 features strictly anchored to Seed 42.
* **Model N (Store Spend Specialist):** 0.40 LightGBM / 0.60 CatBoost regressor on 218 features strictly anchored to Seed 42 with median imputation.
* **Model O (Opportunity Specialist):** Pure single-seed (Seed 42) consensus majority vote across Model A (202), Model B (218), and Model C (225) with summer holiday log-odds biases.
* **Local Validation Score:** **`0.28926`** on Sep–Nov 2025 cohort / **`0.27737`** 3-fold mean. Live Zindi LB: `0.30735` (Rank **#46**).
* **Primary Output:** `submissions/sub_champion_local.csv` *(alias: `sub_model_best2.csv`)*.

---

## 3. Repository Architecture

```text
sasol-customer-value/
│
├── README.md                              # Master project overview & reproduction guide
├── pyproject.toml                         # Project metadata, dependencies & tool configs
├── uv.lock                                # Deterministic cryptographic lockfile
│
├── docs/                                  # Documentation
│   ├── PLAIN_ENGLISH_GUIDE.md             # Guide to till slips, basket totals & labels
│   └── competition/                       # Canonical competition reference files (read-only)
│       ├── label_config.json              # Canonical single source of truth (SSOT)
│       ├── DATA_DICTIONARY.md             # Dataset schema specifications
│       ├── LABEL_RULES.md                 # Fixed-point target calculation rules
│       ├── label_rules.py                 # Reference label generation logic
│       ├── dataset_and_task_overview.md   # Task description
│       └── challenge_overview_and_getting_started_guide.md
│
├── data/
│   ├── raw/                               # Immutable raw transaction streams (train, test)
│   └── processed/                         # Processed snapshot parquets (15 monthly snapshots)
│
├── src/
│   └── sasol_customer_value/              # Tested, typed Python production package
│       ├── config.py                      # SSOT configuration & category mappings
│       ├── data/                          # Data loading, label calculation & snapshots
│       │   ├── loader.py                  # NA literal preservation & string ID handling
│       │   ├── labels.py                  # Fixed-point currency opportunity logic
│       │   └── snapshots.py               # Leak-free monthly snapshot manager
│       ├── features/                      # 100% leak-free feature extraction
│       │   ├── rfm.py                     # Recency, frequency, monetary & fuel volumes (68 feats)
│       │   └── behavioral.py              # Inter-purchase cadence, station loyalty, personas (157 feats)
│       ├── models/                        # Target specialist estimators & validation
│       │   ├── fuel.py                    # Model F Specialist (FuelModel)
│       │   ├── nonfuel.py                 # Model N Specialist (NonfuelModel)
│       │   ├── opportunity.py             # Model O Specialist (OpportunityModel)
│       │   ├── ensemble.py                # Core multi-model ensemble definitions & bias dicts
│       │   ├── metrics.py                 # Exact composite evaluation formula
│       │   ├── validation.py              # Strict 3-month embargoed rolling-origin validator
│       │   └── predict.py                 # Standard CLI inference pipeline
│       └── utils/
│           └── helpers.py                 # Automated 5-point submission compliance checker
│
├── scripts/                               # Production execution entrypoints (DRY & SSOT)
│   ├── evaluate.py                        # Out-of-time (OOT) rolling-origin validation harness
│   ├── train.py                           # Full historical dataset training across all snapshots
│   └── assemble.py                        # Unified end-to-end submission assembler & verifier
│
├── tests/                                 # Automated pytest test suite (15 passed)
└── submissions/                           # Final verified competition submission candidates
    ├── sub_champion_zindi.csv             # Proven Leaderboard Champion (LB 0.31076 - 0.31144, #26)
    ├── sub_champion_local.csv             # Local Validation Benchmark (LB 0.30735, #46)
    ├── sub_model_best1.csv                # Reference alias for sub_champion_zindi.csv
    └── sub_model_best2.csv                # Reference alias for sub_champion_local.csv
```

---

## 4. Setup & Environment Synchronization

The project standardizes on Python `3.12` and [`uv`](https://astral.sh/uv/) for fast, deterministic dependency resolution.

```powershell
# 1. Synchronize environment and install dependencies from lockfile
uv sync --all-groups

# 2. Run the automated test suite
uv run pytest -v tests/

# 3. Verify static typing and code style
uv run mypy src/
uv run ruff check src/ tests/ scripts/
```

---

## 5. Model Execution & Reproduction Instructions

The platform provides unified, DRY execution scripts decoupled across the three competition targets (**Model F**, **Model N**, and **Model O**).

### Generate Competition Submissions (`scripts/assemble.py`)

To assemble and verify submission files from scratch across all 15 historical monthly snapshots (65k rows):

```powershell
# Generate Zindi Leaderboard Champion (#26: submissions/sub_champion_zindi.csv)
uv run python scripts/assemble.py --config best_zindi

# Generate Local Validation Champion (#46: submissions/sub_champion_local.csv)
uv run python scripts/assemble.py --config best_local
```

### Modular Training by Target Specialist (`scripts/train.py`)

To train individual target specialists independently:

```powershell
# Train only Model F (Fuel Specialist)
uv run python scripts/train.py --config best_zindi --target F

# Train only Model N (Non-fuel Specialist)
uv run python scripts/train.py --config best_zindi --target N

# Train only Model O (Opportunity Specialist)
uv run python scripts/train.py --config best_zindi --target O

# Train all specialists
uv run python scripts/train.py --config best_zindi --target all
```

### Validate Output Compliance
Every generated prediction file is automatically verified by `scripts/assemble.py`, or can be checked manually:
```powershell
uv run python -m sasol_customer_value.utils.helpers --verify-submission submissions/sub_champion_zindi.csv
uv run python -m sasol_customer_value.utils.helpers --verify-submission submissions/sub_champion_local.csv
```
Verification checks:
1. Exactly 5,488 rows matching target customer IDs in identical order.
2. Complete schema: `ID`, `CLV_fuel`, `CLV_nonfuel`, `Opportunity`.
3. Zero null, NaN, or infinite values.
4. Non-negative continuous spend ($z \ge 0.0$).
5. Opportunity classes belong strictly to the 17 defined categories.

---

## 6. Out-Of-Time (OOT) Validation Protocol (`scripts/evaluate.py`)

To guard against temporal target leakage, models are validated using a **Rolling-Origin Out-Of-Time Validation Harness with a Strict 3-Month Outcome Embargo**:

```
Historical Training Snapshots                          Val Cutoff      3-Month Outcome Window
───────────────────────────────────────────────────────────┼─────────────────────────────────►
[=================== Features (t < Cutoff) ===============)│[======= Ground Truth Labels ====)
                                                           │
Quarterly Fold 1:                                          2025-03-01   ->   2025-06-01
Quarterly Fold 2:                                          2025-06-01   ->   2025-09-01
Quarterly Fold 3:                                          2025-09-01   ->   2025-12-01
```

To run the complete rolling-origin validation harness across all quarterly folds:
```powershell
# Evaluate all models across the 3 embargoed folds
uv run python scripts/evaluate.py --config all

# Or evaluate a specific configuration
uv run python scripts/evaluate.py --config best_zindi
uv run python scripts/evaluate.py --config best_local
```
