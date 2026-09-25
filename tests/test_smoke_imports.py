"""Smoke test to verify package installation, imports, and config integrity."""
import pytest
from sasol_customer_value import config
from sasol_customer_value.utils.helpers import set_seed


def test_package_imports():
    """Verify package imports and basic utilities work."""
    set_seed(42)


def test_config_paths():
    """Verify critical project paths and files exist."""
    assert config.PROJECT_ROOT.exists(), "PROJECT_ROOT must exist"
    assert config.RAW_DATA_DIR.exists(), "RAW_DATA_DIR must exist"
    assert config.DOCS_DIR.exists(), "DOCS_DIR must exist"
    assert config.RAW_TRAIN_PATH.exists(), f"Missing train.csv at {config.RAW_TRAIN_PATH}"
    assert config.RAW_TEST_PATH.exists(), f"Missing test.csv at {config.RAW_TEST_PATH}"
    assert config.SAMPLE_SUBMISSION_PATH.exists(), f"Missing SampleSubmission.csv at {config.SAMPLE_SUBMISSION_PATH}"
    assert config.LABEL_CONFIG_PATH.exists(), f"Missing label_config.json at {config.LABEL_CONFIG_PATH}"


def test_competition_constants():
    """Verify authoritative competition constants."""
    assert len(config.OPPORTUNITY_LABELS) == 17, "Must have exactly 17 Opportunity classes"
    assert "Inactivity" in config.OPPORTUNITY_LABELS
    assert "Stable" in config.OPPORTUNITY_LABELS
    assert config.METRIC_WEIGHTS["CLV_fuel"] == 0.30
    assert config.METRIC_WEIGHTS["CLV_nonfuel"] == 0.30
    assert config.METRIC_WEIGHTS["Opportunity"] == 0.40
    assert config.NORMALIZATION_CONSTANTS["CLV_fuel"] == 2.352825386185113
    assert config.NORMALIZATION_CONSTANTS["CLV_nonfuel"] == 2.4200904988927476
