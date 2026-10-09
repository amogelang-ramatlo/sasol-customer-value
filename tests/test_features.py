"""Tests for leakage-free feature extraction."""

import pandas as pd

from sasol_customer_value.data.loader import load_transactions
from sasol_customer_value.features.rfm import extract_rfm_features


def test_rfm_feature_extraction_anti_leakage():
    """Verify that transactions occurring after cutoff date have zero impact on features."""
    # Load small slice of transactions
    df = load_transactions(nrows=1000)
    cutoff = "2024-06-01"

    feats_orig = extract_rfm_features(df, cutoff_date=cutoff)

    # Inject fake transactions in late 2025 (well after cutoff)
    df_tampered = df.copy()
    future_rows = df.iloc[:10].copy()
    future_rows["DateTimeZA"] = pd.Timestamp("2025-11-01")
    future_rows["_time"] = pd.Timestamp("2025-11-01")
    future_rows["Quantity"] = 999999
    future_rows["TotalExclAmount"] = 999999
    df_tampered = pd.concat([df_tampered, future_rows], ignore_index=True)

    feats_tampered = extract_rfm_features(
        df_tampered, cutoff_date=cutoff, customer_ids=feats_orig.index
    )

    # Features must be identical despite future injected rows
    pd.testing.assert_frame_equal(feats_orig, feats_tampered)


def test_rfm_feature_shape_and_customers():
    """Verify customer alignment and non-empty feature generation."""
    df = load_transactions(nrows=1000)
    cutoff = "2024-06-01"
    cust_ids = pd.Index(df["ID"].unique()[:10])

    feats = extract_rfm_features(df, cutoff_date=cutoff, customer_ids=cust_ids)

    assert len(feats) == len(cust_ids)
    assert "days_since_last_visit" in feats.columns
    assert "fuel_litres_last_30d" in feats.columns
    assert "nonfuel_spend_last_90d" in feats.columns
    assert (feats["days_since_last_visit"] >= 0).all()
