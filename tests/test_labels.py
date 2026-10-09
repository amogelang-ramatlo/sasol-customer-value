"""Unit tests for ground-truth label generation and schema verification."""

import pandas as pd

from sasol_customer_value.config import (
    OPPORTUNITY_LABELS,
    PROCESSED_DATA_DIR,
    SUBMISSION_COLUMNS,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)


def test_validation_label_schema():
    """Verify that generated validation labels match submission requirements."""
    parquet_path = PROCESSED_DATA_DIR / "val_labels_20250901.parquet"
    assert parquet_path.exists(), "val_labels_20250901.parquet should exist"

    df = pd.read_parquet(parquet_path)
    # Check columns
    assert list(df.columns) == SUBMISSION_COLUMNS

    # Check non-empty
    assert len(df) > 5000

    # Check no missing values
    assert not df.isna().any().any()

    # Check non-negative regression targets
    assert (df[TARGET_FUEL] >= 0).all()
    assert (df[TARGET_NONFUEL] >= 0).all()

    # Check all classes belong to authoritative 17 classes
    unique_classes = set(df[TARGET_OPPORTUNITY])
    assert unique_classes.issubset(set(OPPORTUNITY_LABELS))
