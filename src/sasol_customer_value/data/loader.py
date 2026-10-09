"""Reusable data loading utilities for Sasol loyalty transaction data.

CRITICAL: The dataset uses literal string 'NA' in ItemCategoryLevel2 to represent
a legitimate retail category. CSV loaders must never parse 'NA' as null/NaN.
"""

from pathlib import Path

import pandas as pd

from sasol_customer_value.config import (
    DATETIME_COL,
    ID_COL,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    SAMPLE_SUBMISSION_PATH,
)


def load_transactions(
    path: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load raw historical line-item transaction records.

    Ensures string 'NA' is treated as a valid category string rather than NaN,
    and parses DateTimeZA into datetime objects.

    Args:
        path: Path to transaction CSV file. Defaults to config.RAW_TRAIN_PATH.
        nrows: Optional number of rows to read (useful for testing/fast audits).

    Returns:
        DataFrame containing cleaned transaction records.
    """
    file_path = Path(path) if path is not None else RAW_TRAIN_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Transaction file not found: {file_path}")

    # keep_default_na=False preserves "NA" as literal string
    df = pd.read_csv(
        file_path,
        nrows=nrows,
        keep_default_na=False,
        dtype={
            ID_COL: str,
            "TransactionId": str,
            "LocationId": str,
            "StockItemId": str,
            "LineItemType": str,
            "ItemName": str,
            "ItemCategoryLevel1": str,
            "ItemCategoryLevel2": str,
            "ItemCategoryLevel3": str,
        },
    )

    # Parse timestamps
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], format="mixed", errors="raise")
    df["_time"] = df[DATETIME_COL]

    # Numeric columns
    numeric_cols = [
        "LineItemSequence",
        "TransactionTotalInclAmount",
        "Quantity",
        "TotalExclAmount",
        "TotalModifierAmount",
        "UnitSellingExclPrice",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_test_customers(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load customer IDs requiring predictions in test period.

    Args:
        path: Path to test.csv. Defaults to config.RAW_TEST_PATH.

    Returns:
        DataFrame with column 'ID' as string.
    """
    file_path = Path(path) if path is not None else RAW_TEST_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Test customer file not found: {file_path}")

    df = pd.read_csv(file_path, dtype={ID_COL: str})
    return df


def load_sample_submission(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load reference sample submission file.

    Args:
        path: Path to SampleSubmission.csv. Defaults to config.SAMPLE_SUBMISSION_PATH.

    Returns:
        DataFrame matching the required submission schema.
    """
    file_path = Path(path) if path is not None else SAMPLE_SUBMISSION_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Sample submission file not found: {file_path}")

    return pd.read_csv(file_path, dtype={ID_COL: str, "Opportunity": str})
