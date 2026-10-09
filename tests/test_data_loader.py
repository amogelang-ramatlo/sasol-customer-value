"""Tests for data loading integrity, specifically string 'NA' preservation."""

import pandas as pd

from sasol_customer_value.config import ID_COL, SUBMISSION_COLUMNS
from sasol_customer_value.data.loader import (
    load_sample_submission,
    load_test_customers,
    load_transactions,
)


def test_load_test_customers():
    """Verify test customers load properly with string ID column."""
    df_test = load_test_customers()
    assert ID_COL in df_test.columns
    assert len(df_test) > 0
    assert df_test[ID_COL].dtype == object or isinstance(df_test[ID_COL].iloc[0], str)


def test_load_sample_submission():
    """Verify SampleSubmission.csv loads with required columns."""
    df_sample = load_sample_submission()
    assert list(df_sample.columns) == SUBMISSION_COLUMNS
    assert len(df_sample) > 0


def test_transaction_loader_na_preservation():
    """Verify that literal 'NA' category values are loaded as strings and not NaN."""
    # Test reading the first 500 rows of train.csv
    df_sample = load_transactions(nrows=500)
    assert len(df_sample) == 500
    assert "ItemCategoryLevel2" in df_sample.columns
    # Check that ItemCategoryLevel2 column has string dtype
    assert pd.api.types.is_string_dtype(df_sample["ItemCategoryLevel2"])
