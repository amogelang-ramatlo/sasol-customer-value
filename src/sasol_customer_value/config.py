"""Authoritative configuration and constants for the Sasol Customer Value project.

All competition rules, normalization factors, calibration cutoffs, and opportunity classes
are dynamically loaded from docs/competition/label_config.json to ensure a single source of truth.
"""

import json
from pathlib import Path
from typing import Any, Literal

ModelVariant = Literal["best_zindi", "best_local", "champion_v2"]

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

# Precision scaling factor used in label_rules.py for exact currency calculation
SCALE: int = 1_000_000

# Primary Out-Of-Time (OOT) local validation cutoff (mirrors the 3-month test window)
DEFAULT_VAL_CUTOFF: str = "2025-09-01"

# Excluded categories that do not count as activity under label rules
EXCLUDED_CATEGORIES: list[str] = [
    cat for cat, props in CATEGORY_MAPPING.items() if not props.get("included", True)
]
