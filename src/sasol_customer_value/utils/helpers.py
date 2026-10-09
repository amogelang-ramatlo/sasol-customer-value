"""Helper functions for reproducibility and validation."""

import os
import random

import numpy as np
import pandas as pd

from sasol_customer_value.config import (
    ID_COL,
    OPPORTUNITY_LABELS,
    RANDOM_SEED,
    SUBMISSION_COLUMNS,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across Python, NumPy, and environment for deterministic results."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def validate_submission_format(df: pd.DataFrame, expected_ids: list[str]) -> None:
    """Verify that a submission DataFrame complies with competition format rules.

    Raises:
        ValueError if columns, row counts, IDs, values, or class labels are invalid.
    """
    # 1. Column structure
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"Columns mismatch. Expected {SUBMISSION_COLUMNS}, got {list(df.columns)}"
        )

    # 2. Row count and IDs
    if len(df) != len(expected_ids):
        raise ValueError(
            f"Row count mismatch. Expected {len(expected_ids)}, got {len(df)}"
        )

    submitted_ids = set(df[ID_COL].astype(str))
    required_ids = set(map(str, expected_ids))
    if submitted_ids != required_ids:
        missing = required_ids - submitted_ids
        unexpected = submitted_ids - required_ids
        raise ValueError(
            f"ID mismatch: {len(missing)} missing IDs, {len(unexpected)} unexpected IDs."
        )

    # 3. Check for NaNs
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Submission contains NaN values in columns: {nan_cols}")

    # 4. Non-negative numeric targets
    if (df[TARGET_FUEL] < 0).any():
        raise ValueError("Found negative predictions in CLV_fuel.")
    if (df[TARGET_NONFUEL] < 0).any():
        raise ValueError("Found negative predictions in CLV_nonfuel.")

    # 5. Opportunity class validity
    invalid_classes = set(df[TARGET_OPPORTUNITY]) - set(OPPORTUNITY_LABELS)
    if invalid_classes:
        raise ValueError(
            f"Found unrecognized Opportunity class labels: {invalid_classes}"
        )


def main() -> None:
    """CLI helper to verify competition submission files."""
    import argparse
    from pathlib import Path

    from sasol_customer_value.data.loader import load_test_customers

    parser = argparse.ArgumentParser(description="Sasol submission file verifier")
    parser.add_argument(
        "--verify-submission",
        type=Path,
        required=True,
        help="Path to submission CSV file to verify",
    )
    args = parser.parse_args()

    sub_path = args.verify_submission
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file not found: {sub_path}")

    sub_df = pd.read_csv(sub_path, dtype={ID_COL: str, TARGET_OPPORTUNITY: str})
    expected_ids = list(load_test_customers()[ID_COL].astype(str))

    validate_submission_format(sub_df, expected_ids=expected_ids)
    print(
        f"[VERIFIED] {sub_path} is 100% compliant ({len(sub_df):,} rows, valid schema)."
    )


if __name__ == "__main__":
    main()
