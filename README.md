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

## 2. Model Architecture & Selected Solutions

Because the evaluation targets are evaluated independently, the platform implements a **Target Decoupled Architecture** pairing specialized regressors with calibrated consensus classification:

### Primary Solution: Target-Decoupled Ensemble (`sub_model_best1.csv`)
* **Fuel Spend Regressor (Model A):** Specialized 3-model weighted blend (0.50 LightGBM / 0.30 CatBoost / 0.20 XGBoost) trained on 202 features strictly anchored to Seed 42. Produces the lowest fuel spend error.
* **Store Spend Regressor (Model B):** Bagged ensemble (0.40 LightGBM / 0.60 CatBoost) trained on 218 features across 5 random seeds (`[42, 100, 2024, 777, 999]`). CatBoost's depth regularizer minimizes non-fuel residual error.
* **Opportunity Classification (Committee Consensus):** Discrete majority consensus among three distinct feature representations (Model A with 202 features, Model B with 218 features, and Model C with 225 features) with summer-calibrated class log-odds biases. Operates without artificial heuristic overrides, relying directly on the learned tree probabilities and committee majority vote.
* **Output Path:** `submissions/sub_model_best1.csv` *(retained reference: `submissions/sub_v14_all_star_synthesis.csv`)*.

### Secondary Solution: Pure Single-Seed Committee (`sub_model_best2.csv`)
* **Strict Seed 42 Execution:** Fully reproducible single-seed counterpart operating without multi-seed averaging.
* **Mature 225-Feature Store:** Incorporates complete inter-purchase cadence (gap statistics, gap volatility), station loyalty concentration, and shopper personas.
* **Committee Decision Voting:** Pure consensus majority vote across three feature models, serving as an un-bagged, sharp-boundary alternative.
* **Output Path:** `submissions/sub_model_best2.csv` *(retained reference: `submissions/sub_v13_single_seed.csv`)*.

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
│   ├── competition/                       # Canonical competition reference files (read-only)
│   │   ├── label_config.json              # Canonical single source of truth (SSOT)
│   │   ├── DATA_DICTIONARY.md             # Dataset schema specifications
│   │   ├── LABEL_RULES.md                 # Fixed-point target calculation rules
│   │   └── label_rules.py                 # Reference label generation logic
│   └── experiment/                        # Historical research archive
│       ├── archived_experiments.md        # Comprehensive ledger of all 14 exploratory models
│       ├── SUBMISSIONS_MANIFEST.md        # Historical submission log & benchmarks
│       ├── COMPETITION_HANDOFF.md         # Internal development briefings
│       ├── GEMINI.md                      # Engineering design history
│       ├── notebooks/                     # Exploratory analysis & audit notebooks
│       ├── scripts/                       # Archived exploration & ablation scripts
│       ├── src/                           # Experimental model modules (hurdle regressor)
│       └── submissions/                   # Historical intermediate outputs
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
│       ├── features/                      # 100% leak-free, non-fragmented feature extraction
│       │   ├── rfm.py                     # Recency, frequency, monetary & fuel volumes (68 feats)
│       │   └── behavioral.py              # Inter-purchase cadence, station loyalty, personas (157 feats)
│       ├── models/                        # Estimators, ensembles, metrics & validation
│       │   ├── ensemble.py                # Core multi-model ensemble (LightGBM + CatBoost + XGBoost)
│       │   ├── metrics.py                 # Exact composite evaluation formula
│       │   ├── validation.py              # Strict 3-month embargoed rolling-origin validator
│       │   └── predict.py                 # Standard CLI inference pipeline
│       └── utils/
│           └── helpers.py                 # Automated 5-point submission compliance checker
│
├── scripts/                               # Production execution entrypoints
│   ├── build_sub_model_best1.py           # Builds/trains Primary Flagship Model (sub_model_best1.csv)
│   ├── build_sub_model_best2.py           # Builds/trains Secondary Single-Seed Model (sub_model_best2.csv)
│   └── benchmark_embargoed_folds.py        # Runs multi-fold OOT rolling-origin validation
│
├── tests/                                 # Automated pytest test suite (12 passed)
└── submissions/                           # Final verified model outputs
    ├── sub_model_best1.csv                # Primary solution output (from v14 target-decoupled)
    ├── sub_model_best2.csv                # Secondary solution output (from v13 single-seed)
    ├── sub_v14_all_star_synthesis.csv     # Retained reference baseline
    └── sub_v13_single_seed.csv            # Retained reference baseline
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

### Generate the Primary Flagship Model
To generate the primary target-decoupled model from verified components:
```powershell
uv run python scripts/build_sub_model_best1.py
```
*(To train all underlying models completely from scratch across all 15 historical monthly snapshots, pass `--train-from-scratch`)*.

### Generate the Secondary Single-Seed Model
To train and evaluate the pure single-seed model strictly on **Seed 42**:
```powershell
uv run python scripts/build_sub_model_best2.py --seed 42
```

### Validate Output Compliance
Every generated prediction file is automatically verified against the 5 schema constraints:
```powershell
uv run python -m sasol_customer_value.utils.helpers --verify-submission submissions/sub_model_best1.csv
uv run python -m sasol_customer_value.utils.helpers --verify-submission submissions/sub_model_best2.csv
```
Verification checks:
1. Exactly 5,488 rows matching target customer IDs in identical order.
2. Complete schema: `ID`, `CLV_fuel`, `CLV_nonfuel`, `Opportunity`.
3. Zero null, NaN, or infinite values.
4. Non-negative continuous spend ($z \ge 0.0$).
5. Opportunity classes belong strictly to the 17 defined categories.

---

## 6. Out-Of-Time (OOT) Validation Protocol

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
uv run python scripts/benchmark_embargoed_folds.py
```

---

## 7. Historical Research Archive

For complete documentation, hyperparameters, and lessons learned from the experimental iterations explored during model development, refer to:
👉 [`docs/experiment/archived_experiments.md`](file:///docs/experiment/archived_experiments.md)
