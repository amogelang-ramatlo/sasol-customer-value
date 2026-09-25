"""Authoritative configuration and constants for the Sasol Customer Value project.

All target rules, normalization factors, and class specifications match
the competition rules documented in docs/label_config.json and docs/LABEL_RULES.md.
"""
from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# Project Environment Paths
# -----------------------------------------------------------------------------
# src/sasol_customer_value/config.py -> parent x 3 = PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
TESTS_DIR = PROJECT_ROOT / "tests"

RAW_TRAIN_PATH = RAW_DATA_DIR / "train.csv"
RAW_TEST_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "SampleSubmission.csv"
LABEL_CONFIG_PATH = DOCS_DIR / "label_config.json"

# -----------------------------------------------------------------------------
# Competition Settings & Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42

# Column identifiers
ID_COL: str = "ID"
TRANSACTION_ID_COL: str = "TransactionId"
DATETIME_COL: str = "DateTimeZA"

# Primary submission targets
TARGET_FUEL: str = "CLV_fuel"
TARGET_NONFUEL: str = "CLV_nonfuel"
TARGET_OPPORTUNITY: str = "Opportunity"
SUBMISSION_COLUMNS: List[str] = [ID_COL, TARGET_FUEL, TARGET_NONFUEL, TARGET_OPPORTUNITY]

# -----------------------------------------------------------------------------
# Evaluation Metric Weights
# Final Score = 0.30 * RMSE(CLV_fuel) + 0.30 * RMSE(CLV_nonfuel) + 0.40 * WeightedF1(Opportunity)
# -----------------------------------------------------------------------------
METRIC_WEIGHTS: Dict[str, float] = {
    TARGET_FUEL: 0.30,
    TARGET_NONFUEL: 0.30,
    TARGET_OPPORTUNITY: 0.40,
}

# -----------------------------------------------------------------------------
# Authoritative Target Normalization & Rules (docs/label_config.json)
# Formula: z = ln(1 + y) / s
# -----------------------------------------------------------------------------
NORMALIZATION_CONSTANTS: Dict[str, float] = {
    TARGET_FUEL: 2.352825386185113,
    TARGET_NONFUEL: 2.4200904988927476,
}

# Precision scaling factor used in label_rules.py for exact currency calculation
SCALE: int = 1_000_000

# Minimum historical baskets required strictly before cutoff for customer eligibility
MINIMUM_HISTORICAL_TRANSACTIONS: int = 3

# Temporal Cutoffs
PUBLIC_HISTORY_END_EXCLUSIVE: str = "2025-12-01"
OUTCOME_END_EXCLUSIVE: str = "2026-03-01"

# Historical quarterly cutoffs available in training history
CALIBRATION_CUTOFFS: List[str] = [
    "2024-09-01",
    "2024-12-01",
    "2025-03-01",
    "2025-06-01",
    "2025-09-01",
]

# Primary Out-Of-Time (OOT) local validation cutoff (mirrors the 3-month test window)
DEFAULT_VAL_CUTOFF: str = "2025-09-01"

# -----------------------------------------------------------------------------
# Exclusions & Special Items
# -----------------------------------------------------------------------------
ADJUSTMENT_ITEMS: List[str] = [
    "TOTAL/ROUNDING DISCO",
    "PENSIONER DISCOUNT",
    "WOW_PROMO",
]

EXCLUDED_CATEGORIES: List[str] = [
    "Value added services",
    "Print media",
    "Carrier Bags",
]

# -----------------------------------------------------------------------------
# Authoritative Opportunity Classes (Exactly 17 classes)
# -----------------------------------------------------------------------------
OPPORTUNITY_LABELS: List[str] = [
    "Existing-category growth: Beverages",
    "Existing-category growth: Food & beverage",
    "Existing-category growth: Fuel",
    "Existing-category growth: Other",
    "Inactivity",
    "New category adoption: Beverages",
    "New category adoption: Braai & Ice",
    "New category adoption: Breads & rolls",
    "New category adoption: Confectionary",
    "New category adoption: Food & beverage",
    "New category adoption: Groceries",
    "New category adoption: Ice-cream",
    "New category adoption: Lubricants",
    "New category adoption: Other",
    "New category adoption: Snacking",
    "New category adoption: Tobacco",
    "Stable",
]
