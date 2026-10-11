# GEMINI.md — Sasol Customer Value & Opportunity Platform

> **Project Identity:** Sasol Loyalty Customer Value & Opportunity Prediction  
> **Repository:** `sasol-customer-value`  
> **Python Package:** `sasol_customer_value` (`src/sasol_customer_value/`)  
> **Target Horizon:** 3-Month Forward Rolling Window (`CLV_fuel`, `CLV_nonfuel`, `Opportunity`)  
> **Evaluation Mode:** Local Out-Of-Time (OOT) Snapshot Validation + Multi-Metric Leaderboard  

---

## 1. Executive Summary & Guiding Philosophy

Sasol operates extensive retail fuel forecourts and convenience stores (Sasol Delight) across South Africa. Customers participate by presenting digital or physical loyalty identifiers at checkout. This repository implements an end-to-end, leak-free machine learning architecture to forecast customer value metrics and behavioural transitions over a future 3-month window from granular, receipt-level point-of-sale (POS) transaction streams.

### Core Philosophy
> **"A rigorous, leakage-free, reproducible competition solution built like production software, not messy ad-hoc scripts."**

* **Separation of Concerns:** Jupyter notebooks (`notebooks/`) are restricted to exploratory data analysis (EDA), diagnostic plotting, and visual error analysis. All reusable pipelines, feature engineering, schemas, model classes, and inference loops must live inside the typed, tested package `src/sasol_customer_value/`.
* **Zero Premature Bloat:** Avoid heavyweight orchestration engines (Airflow, Kubeflow) or external cloud infrastructure. Maintain a lean, high-velocity codebase using Python 3.12, `uv`, Parquet storage, and vectorized operations.
* **Strict Temporal Integrity:** Time-series customer loyalty data is prone to catastrophic target leakage. All feature representations, aggregations, and normalization statistics are strictly anchored to explicit snapshot cutoffs ($t < t_{\text{cutoff}}$).
* **Single Source of Truth (SSOT):** No hardcoded competition magic numbers or category lists scattered across scripts. All configuration flows directly from `docs/competition/label_config.json` via `sasol_customer_value.config`.

---

## 2. Technical Stack & Environment Standards

The project standardizes on modern Python tooling with deterministic lockfiles.

| Component | Specification | Purpose |
| :--- | :--- | :--- |
| **Runtime** | Python `3.12.x` | Base language runtime pinned via `.python-version` |
| **Package Manager** | `uv` (`pyproject.toml`, `uv.lock`) | Fast, deterministic dependency resolution and PEP 735 dependency groups |
| **Data Engine** | `pandas >= 2.2.0`, `pyarrow >= 15.0.0` | Ingestion, columnar storage, and tabular transformations |
| **Numeric & Scientific** | `numpy >= 1.26.0`, `scipy >= 1.12.0` | Vectorized math, statistical distributions, and fixed precision |
| **ML Frameworks** | `lightgbm >= 4.3.0`, `xgboost >= 2.0.0`, `catboost >= 1.2.0` | Gradient-boosted decision trees for tabular classification/regression |
| **Scikit-Learn** | `scikit-learn >= 1.4.0` | Metric implementations, cross-validation splitters, and preprocessing |
| **Code Quality** | `ruff >= 0.3.0`, `mypy >= 1.9.0` | Linting, code formatting, and static type checking |
| **Test Suite** | `pytest >= 8.0.0` | Unit, integration, and regression testing |
| **Visual Diagnostics** | `matplotlib >= 3.8.0`, `seaborn >= 0.13.0` | Feature distributions, residual plots, and confusion matrices |

---

## 3. Directory Architecture & Boundary Contracts

The project adheres to a clean hexagonal layout. Code flows strictly inward: raw data is immutable, processed artifacts are transient and git-ignored, and package code is strictly tested.

```text
sasol-customer-value/
│
├── .gitignore                      # Ignore .venv, __pycache__, data/raw/*.csv, data/processed/*, submissions/*
├── .python-version                 # Pinned to 3.12
├── pyproject.toml                  # Central project configuration, metadata, and dependencies
├── uv.lock                         # Cryptographic dependency lockfile
├── README.md                       # High-level overview and getting-started guide
├── GEMINI.md                       # Master engineering and architectural specification (SSOT)
│
├── docs/                           # Project and competition documentation
│   ├── PLAIN_ENGLISH_GUIDE.md      # Layman guide to retail till slips and target formulations
│   └── competition/                # Immutable competition assets provided by Zindi / Sasol
│       ├── challenge_overview_and_getting_started_guide.md
│       ├── dataset_and_task_overview.md
│       ├── DATA_DICTIONARY.md
│       ├── LABEL_RULES.md
│       ├── label_config.json       # Canonical configuration JSON (read-only SSOT)
│       └── label_rules.py          # Reference label generation implementation
│
├── data/
│   ├── raw/                        # Immutable raw inputs (train.csv, test.csv, SampleSubmission.csv)
│   └── processed/                  # Derived feature stores and OOT validation parquets (git-ignored)
│
├── notebooks/                      # Scratchpads and diagnostics ONLY (no production business logic)
│   ├── 01_data_audit_and_validation.ipynb
│   └── 02_target_analysis.ipynb
│
├── src/
│   └── sasol_customer_value/       # Tested, importable production Python package
│       ├── __init__.py
│       ├── config.py               # SSOT loader: parses label_config.json, exposes paths & constants
│       ├── data/                   # Ingestion and ground-truth label generators
│       │   ├── __init__.py
│       │   ├── loader.py           # Safe CSV reading (NA literal preservation, string IDs)
│       │   └── labels.py           # Official label generator implementing fixed-point logic
│       ├── features/               # Leak-free feature transformation pipelines
│       │   ├── __init__.py
│       │   └── rfm.py              # Recency, frequency, monetary, and category transition features
│       ├── models/                 # Model architectures, trainers, and loss functions
│       │   ├── __init__.py
│       │   └── metrics.py          # Exact competition loss function and individual target metrics
│       └── utils/                  # Deterministic helpers and submission checkers
│           ├── __init__.py
│           └── helpers.py          # Seeding, submission verification, and file I/O
│
├── tests/                          # Automated test suite executed on every commit
│   ├── __init__.py
│   ├── test_smoke_imports.py       # Validates package imports and path resolution
│   ├── test_data_loader.py         # Enforces literal 'NA' category preservation and schema types
│   ├── test_labels.py              # Validates ground-truth label derivation and edge cases
│   └── test_metrics.py             # Validates exact competition composite metric calculation
│
├── reports/
│   └── figures/                    # Persisted high-resolution EDA and diagnostic figures (.gitkeep)
│
└── submissions/                    # Generated submission CSVs with timestamped logs (.gitkeep)
```

---

## 4. Prediction Targets & Mathematical Formulation

For each of the **5,488 target loyalty customers** in `test.csv`, the model must forecast three distinct targets over the forward 3-month evaluation window:

### 4.1 Target Specifications

1. **`CLV_fuel` (Continuous Regression):**
   * **Scope:** Total litres of fuel purchased (`LineItemType == 'Sale_Fuel'` AND `ItemCategoryLevel2 == 'Fuel'`).
   * **Normalized Target ($z$):**
     $$z_{\text{fuel}} = \frac{\ln(1 + y_{\text{litres}})}{s_{\text{fuel}}} \quad \text{where } s_{\text{fuel}} = 2.352825386185113$$
   * **Evaluation Metric:** Root Mean Squared Error (RMSE) on submitted $z$. Weight: **0.30**.
   * **Raw Litres Inversion:** $\hat{y}_{\text{litres}} = \exp(s_{\text{fuel}} \times z) - 1$.

2. **`CLV_nonfuel` (Continuous Regression):**
   * **Scope:** Total store merchandise spend in ZAR (`TotalExclAmount`), excluding `Fuel`, `Carrier Bags`, `Print media`, and `Value added services`.
   * **Normalized Target ($z$):**
     $$z_{\text{nonfuel}} = \frac{\ln(1 + y_{\text{rands}})}{s_{\text{nonfuel}}} \quad \text{where } s_{\text{nonfuel}} = 2.4200904988927476$$
   * **Evaluation Metric:** Root Mean Squared Error (RMSE) on submitted $z$. Weight: **0.30**.
   * **Raw Rand Inversion:** $\hat{y}_{\text{rands}} = \exp(s_{\text{nonfuel}} \times z) - 1$.

3. **`Opportunity` (17-Class Mutually Exclusive Classification):**
   * **Scope:** Categorizes customer retail progression or lapse during the 3-month horizon.
   * **Classes (Exactly 17):**
     * **Non-growth States (2):** `Inactivity`, `Stable` (includes declining spend).
     * **Existing-Category Growth (4):** `Beverages`, `Food & beverage`, `Fuel`, `Other`.
     * **New-Category Adoption (11):** `Beverages`, `Braai & Ice`, `Breads & rolls`, `Confectionary`, `Food & beverage`, `Groceries`, `Ice-cream`, `Lubricants`, `Other`, `Snacking`, `Tobacco`.
   * **Evaluation Metric:** Multiclass Weighted F1 Score ($F_{1,\text{weighted}}$). Weight: **0.40**.

### 4.2 Composite Competition Objective Function
$$\mathcal{L}_{\text{comp}} = 0.30 \times \text{RMSE}\left(z_{\text{fuel}}, \hat{z}_{\text{fuel}}\right) + 0.30 \times \text{RMSE}\left(z_{\text{nonfuel}}, \hat{z}_{\text{nonfuel}}\right) + 0.40 \times \left(1 - F_{1,\text{weighted}}\left(y_{\text{opp}}, \hat{y}_{\text{opp}}\right)\right)$$

*(Note: While the competition leaderboard reports $0.3 \times \text{RMSE} + 0.3 \times \text{RMSE} + 0.4 \times F_1$, internal optimization treats metric loss minimization as minimizing the negative or inverted $F_1$ component).*

---

## 5. Domain Business Rules & Data Integrity Traps

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

### 5.1 The Literal `"NA"` String Trap
* In `ItemCategoryLevel2`, the string literal `"NA"` denotes a legitimate merchandise department (e.g., specific non-categorized store merchandise).
* **Mandatory Ingestion Rule:** You must disable standard Pandas NA parsing for this column:
  ```python
  # CORRECT: Never allow Pandas to convert "NA" to np.nan
  df = pd.read_csv(filepath, keep_default_na=False, na_values=["", "#N/A", "null"])
  ```
* Spending under `"NA"` contributes to `CLV_nonfuel` and resolves to the opportunity category `Other`.

### 5.2 The Repeated Basket Total Trap
* `TransactionTotalInclAmount` represents the grand basket total and is duplicated across every line-item row belonging to that `TransactionId`.
* **Mandatory Rule:** **NEVER** compute customer spend by summing `TransactionTotalInclAmount`. Customer spend is strictly calculated from signed line-level `TotalExclAmount`.

### 5.3 Customer Eligibility Threshold
* A customer is eligible for opportunity scoring and target calculation if and only if they have accumulated **$\ge 3$ distinct historical `TransactionId` records** strictly prior to the snapshot cutoff timestamp ($t < t_{\text{cutoff}}$).

### 5.4 Opportunity Classification Logic & Tie-Breaking
1. **Growth Candidate Criteria:**
   * Requires historical purchase in category prior to cutoff.
   * Requires $\ge 1$ purchase in the outcome window.
   * Net spend increase $\ge \text{R}20.00$ over the preceding 3-month baseline window.
   * Outcome window spend $\ge 1.25 \times$ baseline window spend.
2. **Adoption Candidate Criteria:**
   * Zero purchases in that category across the customer's **entire observed history** prior to cutoff.
   * $\ge 1$ purchase in the outcome window with strictly positive spend ($> 0$).
3. **Winner Resolution Hierarchy:**
   * Candidates compete based on **Net Rand Increment** (for Fuel, rand increment is computed from `TotalExclAmount`, not litres).
   * Highest increment wins.
   * **Tie-Breaker 1:** Adoption beats Growth if increments are identical.
   * **Tie-Breaker 2:** Alphabetical ordering by original raw category name.
   * **Mapping:** Categories compete individually under raw names before unmapped classes collapse into `Other`.
4. **Fixed-Point Precision Rule:**
   * Currency calculations must enforce 6-decimal fixed-point precision (`Decimal` or scaled integers `SCALE = 1_000_000`) to guarantee deterministic tie-breaking without floating-point IEEE-754 drift.

---

## 6. Leak-Free Validation Strategy

Local Out-Of-Time (OOT) validation is the primary benchmark. Public leaderboard submissions serve only as an external smoke test.

```
Historical Data Window                               Cutoff        3-Month Outcome Window
───────────────────────────────────────────────────────┼─────────────────────────────────►
[============ Training & Feature Extraction ==========)│[====== Ground Truth Labels =====)
                                                       │
Snapshot A (Local Validation):                         2025-09-01   ->   2025-12-01
Snapshot B (Final Competition Test):                   2025-12-01   ->   2026-03-01
```

### 6.1 Validation Protocol
1. **Validation Cutoff:** `2025-09-01 00:00:00`
   * Feature generation window: All data strictly $< \text{2025-09-01}$.
   * Evaluation window: $[\text{2025-09-01}, \text{2025-12-01})$.
   * Ground truth target file: `data/processed/val_labels_20250901.parquet`.
2. **Test Cutoff:** `2025-12-01 00:00:00`
   * Feature generation window: All data strictly $< \text{2025-12-01}$.
   * Evaluation window: $[\text{2025-12-01}, \text{2026-03-01})$ (corresponds to `test.csv` customers).

### 6.2 Temporal Anti-Leakage Guardrails
* Feature engineering functions must accept an explicit `cutoff_date: pd.Timestamp` argument.
* Filtering `df[df['DateTimeZA'] < cutoff_date]` must occur as the **first operation** before any aggregation or state computation.
* Cross-validation splitters that randomly shuffle customers or line items across time are strictly prohibited.

---

## 7. Single Source of Truth (SSOT) Architecture

All domain constants, category mappings, and scaling factors must originate from `docs/competition/label_config.json`. Hardcoded constants in model files or notebooks are considered lint errors.

### Implementation Pattern (`src/sasol_customer_value/config.py`)
```python
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_CONFIG_PATH = REPO_ROOT / "docs" / "competition" / "label_config.json"

def load_competition_config() -> Dict[str, Any]:
    if not LABEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing authoritative label config at {LABEL_CONFIG_PATH}")
    with open(LABEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_CONFIG = load_competition_config()

S_FUEL: float = float(_CONFIG["normalization"]["CLV_fuel"])
S_NONFUEL: float = float(_CONFIG["normalization"]["CLV_nonfuel"])
EXCLUDED_CATEGORIES: List[str] = [
    cat for cat, props in _CONFIG["category_mapping"].items() if not props.get("included", True)
]
ADJUSTMENT_ITEMS: List[str] = _CONFIG["adjustment_items"]
OPPORTUNITY_CLASSES: List[str] = _CONFIG["opportunity_labels"]
RANDOM_SEED: int = 42
```

---

## 8. Development & Verification Workflows

All contributors and AI pairing sessions must verify changes using the standard project toolchain.

### 8.1 Setup & Environment Sync
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize and sync virtual environment with all dependency groups
uv sync --all-groups
```

### 8.2 Code Quality, Typing & Testing
```bash
# Run code linting and formatting checks
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Execute static type checking
uv run mypy src/

# Run complete automated test suite
uv run pytest -v tests/
```

### 8.3 Submission Generation & Verification
```bash
# Run baseline inference pipeline
uv run python -m sasol_customer_value.models.predict --cutoff 2025-12-01 --output submissions/sub_baseline.csv

# Verify submission schema, row count (5488), null absence, and boundary constraints
uv run python -m sasol_customer_value.utils.helpers --verify-submission submissions/sub_baseline.csv
```

---

## 9. Mandatory Engineering & Coding Rules

1. **Deterministic Execution:** Always seed pseudorandom number generators across NumPy, Python standard library, LightGBM, and XGBoost using `sasol_customer_value.config.RANDOM_SEED`.
2. **Type Annotations:** All production functions in `src/` must include complete PEP 484 type hints (`pd.DataFrame`, `pd.Timestamp`, `float`, `int`, etc.).
3. **Immutability of Raw Inputs:** Files inside `data/raw/` must never be modified, overwritten, or renamed by any automated script.
4. **AutoML Ban:** In accordance with competition rules, automated machine learning tools (TPOT, Auto-Sklearn, H2O Driverless, AutoKeras) are prohibited. All pipeline steps and feature interactions must be explicitly crafted.
5. **Probability Submission Standard:** Multiclass predictions must generate calibrated class probabilities. Never hard-threshold intermediate probabilities before final decision assignment.
6. **Submissions Validation:** Every output file destined for Zindi must verify:
   * Exactly 5,488 rows matching `test.csv` IDs.
   * Four columns: `ID`, `CLV_fuel`, `CLV_nonfuel`, `Opportunity`.
   * Zero null, NaN, or infinite values.
   * `CLV_fuel >= 0.0` and `CLV_nonfuel >= 0.0`.
   * `Opportunity` values belong strictly to the 17 defined classes.

---

## 10. Phased Experimental Roadmap

Development proceeds along structured, evidence-based milestones:

```
Phase 1: Foundation ──► Phase 2: Feature Store ──► Phase 3: Model Baselines ──► Phase 4: Ensembling & Polish
• Repo Scaffold          • RFM Aggregations        • LightGBM Regressors         • Multi-model Stacking
• OOT Ground Truth       • Inter-purchase Delays   • Calibrated Classifier       • Metric Threshold Tuning
• Test Suite             • Category Transitions    • 3-Task Local Evaluation     • Final Submission Check
```

1. **Phase 1: Foundation & Ground Truth Generation**
   * Establish immutable SSOT config parser.
   * Implement `labels.py` to generate accurate ground truth for `2025-09-01` OOT validation.
   * Validate unit tests for literal `"NA"` and 17-class mapping.
2. **Phase 2: Temporal Feature Store**
   * Build baseline RFM (Recency, Frequency, Monetary) features strictly parameterized by cutoff date.
   * Extract fuel-specific metrics (litres per fill-up, transaction frequency, fuel grade preferences).
   * Extract store-specific metrics (basket variety, top categories, brand loyalty, category co-occurrence).
3. **Phase 3: Model Baselines & Local Evaluation**
   * Train LightGBM regressors with RMSE objective for $z_{\text{fuel}}$ and $z_{\text{nonfuel}}$.
   * Train multi-class LightGBM classifier for the 17 `Opportunity` classes with class-balanced weighting.
   * Run full evaluation on `val_labels_20250901.parquet` to establish the baseline composite score.
4. **Phase 4: Advanced Iterations & Hypotheses**
   * Test two-stage hurdle models (zero-spend classification followed by log-normal regression).
   * Test multi-task cascaded features (passing predicted CLV spend into the opportunity classifier).
   * Post-processing optimization: Optimize class prediction thresholds specifically for Weighted F1.
