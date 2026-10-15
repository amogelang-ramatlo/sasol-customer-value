"""Authoritative configuration and constants for the Sasol Customer Value project.

All competition rules, normalization factors, calibration cutoffs, and opportunity classes
are dynamically loaded from docs/competition/label_config.json to ensure a single source of truth.
"""

import json
from pathlib import Path
from typing import Any, Literal

ModelVariant = Literal[
    "best_zindi",
    "best_local",
    "champion_v2",
    "champion_v3",
    "champion_v4",
    "champion_v5",
    "champion_v6",
    "champion_v5_reactivated",
    "champion_triple_crown",
]

# -----------------------------------------------------------------------------
# Project Environment Paths
# -----------------------------------------------------------------------------
# src/sasol_customer_value/config.py -> parent x 3 = PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
COMPETITION_DOCS_DIR = DOCS_DIR / "competition"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
TESTS_DIR = PROJECT_ROOT / "tests"

RAW_TRAIN_PATH = RAW_DATA_DIR / "train.csv"
RAW_TEST_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "SampleSubmission.csv"
LABEL_CONFIG_PATH = COMPETITION_DOCS_DIR / "label_config.json"


# -----------------------------------------------------------------------------
# Dynamic Competition Rules Loader (docs/competition/label_config.json)
# -----------------------------------------------------------------------------
def load_competition_json() -> dict[str, Any]:
    """Load competition configuration from authoritative JSON file."""
    if not LABEL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Authoritative competition config not found at: {LABEL_CONFIG_PATH}"
        )
    with open(LABEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_COMP_CONFIG = load_competition_json()

# Authoritative Normalization Constants: z = ln(1 + y) / s
NORMALIZATION_CONSTANTS: dict[str, float] = _COMP_CONFIG["normalization"]

# Authoritative 17 Opportunity Labels
OPPORTUNITY_LABELS: list[str] = _COMP_CONFIG["opportunity_labels"]

# Historical quarterly snapshot cutoffs available in training history
CALIBRATION_CUTOFFS: list[str] = _COMP_CONFIG["calibration_cutoffs"]

# Minimum historical transactions strictly before cutoff for customer eligibility
MINIMUM_HISTORICAL_TRANSACTIONS: int = _COMP_CONFIG["minimum_historical_transactions"]

# Temporal Cutoffs
PUBLIC_HISTORY_END_EXCLUSIVE: str = _COMP_CONFIG["public_history_end_exclusive"]
OUTCOME_END_EXCLUSIVE: str = _COMP_CONFIG["outcome_end_exclusive"]

# Adjustment items that do not contribute to targets or purchase activity
ADJUSTMENT_ITEMS: list[str] = _COMP_CONFIG["adjustment_items"]

# Category mapping dictionary
CATEGORY_MAPPING: dict[str, Any] = _COMP_CONFIG["category_mapping"]

# Explicit scaling constants for target normalizations
S_FUEL: float = float(NORMALIZATION_CONSTANTS["CLV_fuel"])
S_NONFUEL: float = float(NORMALIZATION_CONSTANTS["CLV_nonfuel"])

# Opportunity deterministic rule thresholds
GROWTH_MIN_RAND: float = float(_COMP_CONFIG.get("growth_min_rand", 20.0))
GROWTH_MIN_RELATIVE: float = float(_COMP_CONFIG.get("growth_min_relative", 0.25))
GROWTH_MIN_PURCHASES: int = int(_COMP_CONFIG.get("growth_min_purchases", 1))
ADOPTION_MIN_RAND: float = float(_COMP_CONFIG.get("adoption_min_rand", 0.0))
ADOPTION_MIN_PURCHASES: int = int(_COMP_CONFIG.get("adoption_min_purchases", 1))

# -----------------------------------------------------------------------------
# Competition Settings & Column Identifiers
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42

ID_COL: str = "ID"
TRANSACTION_ID_COL: str = "TransactionId"
DATETIME_COL: str = "DateTimeZA"

# Primary submission targets
TARGET_FUEL: str = "CLV_fuel"
TARGET_NONFUEL: str = "CLV_nonfuel"
TARGET_OPPORTUNITY: str = "Opportunity"
SUBMISSION_COLUMNS: list[str] = [
    ID_COL,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
]

# -----------------------------------------------------------------------------
# Evaluation Metric Weights
# Final Score = 0.30 * RMSE(CLV_fuel) + 0.30 * RMSE(CLV_nonfuel) + 0.40 * WeightedF1(Opportunity)
# -----------------------------------------------------------------------------
METRIC_WEIGHTS: dict[str, float] = {
    TARGET_FUEL: 0.30,
    TARGET_NONFUEL: 0.30,
    TARGET_OPPORTUNITY: 0.40,
}

# Bagging seeds and Opportunity class counts for specialist models
DEFAULT_BAGGING_SEEDS: list[int] = [42, 100, 2024, 777, 999]
N_OPPORTUNITY_CLASSES: int = len(OPPORTUNITY_LABELS)

# Official competition leaderboard composite score parameters:
# Composite = COMPOSITE_OFFSET + COMPOSITE_W_OPP * F1 - COMPOSITE_COEFF_FUEL * RMSE_fuel - COMPOSITE_COEFF_NONFUEL * RMSE_nonfuel
COMPOSITE_OFFSET: float = 0.60
COMPOSITE_W_OPP: float = 0.40
COMPOSITE_COEFF_FUEL: float = 0.40540558
COMPOSITE_COEFF_NONFUEL: float = 0.36764644

# Precision scaling factor used in label_rules.py for exact currency calculation
SCALE: int = 1_000_000

# Primary Out-Of-Time (OOT) local validation cutoff (mirrors the 3-month test window)
DEFAULT_VAL_CUTOFF: str = "2025-09-01"

# Excluded categories that do not count as activity under label rules
EXCLUDED_CATEGORIES: list[str] = [
    cat for cat, props in CATEGORY_MAPPING.items() if not props.get("included", True)
]

# All available historical monthly training snapshots in released dataset
HISTORICAL_SNAPSHOT_CUTOFFS: list[str] = [
    "2024-07-01",
    "2024-08-01",
    "2024-09-01",
    "2024-10-01",
    "2024-11-01",
    "2024-12-01",
    "2025-01-01",
    "2025-02-01",
    "2025-03-01",
    "2025-04-01",
    "2025-05-01",
    "2025-06-01",
    "2025-07-01",
    "2025-08-01",
    "2025-09-01",
]

# Mature historical snapshots where cohort churn has stabilized (excludes early immature 2024-07 to 2024-09 snapshots)
MATURE_SNAPSHOT_CUTOFFS: list[str] = [
    c for c in HISTORICAL_SNAPSHOT_CUTOFFS if c >= "2024-10-01"
]

# Authoritative summer festive calibrated log-odds biases for 17-class Opportunity
DEFAULT_OPP_BIASES: dict[str, float] = {
    "Existing-category growth: Beverages": 1.1,
    "Existing-category growth: Food & beverage": 1.2,
    "Existing-category growth: Fuel": 0.15,
    "Existing-category growth: Other": 0.6,
    "Inactivity": 0.1,
    "New category adoption: Beverages": 1.2,
    "New category adoption: Braai & Ice": 0.5,
    "New category adoption: Breads & rolls": 1.4,
    "New category adoption: Confectionary": 0.3,
    "New category adoption: Food & beverage": -0.5,
    "New category adoption: Groceries": 1.0,
    "New category adoption: Ice-cream": 1.4,
    "New category adoption: Lubricants": 0.4,
    "New category adoption: Other": 1.4,
    "New category adoption: Snacking": -0.2,
    "New category adoption: Tobacco": 0.8,
    "Stable": 0.0,
}


def sanitize_category_name(cat: str) -> str:
    """Consistently sanitize raw category names for feature column naming across modules."""
    return cat.replace(" ", "_").replace("&", "and").replace("-", "_")
