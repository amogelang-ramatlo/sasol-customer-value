"""Tests for official competition metric calculation."""

import numpy as np
import pandas as pd
import pytest

from sasol_customer_value.config import TARGET_FUEL, TARGET_NONFUEL, TARGET_OPPORTUNITY
from sasol_customer_value.models.metrics import compute_competition_metrics


def test_perfect_predictions():
    """Verify that identical predictions yield 0 RMSE and 1.0 F1 score."""
    df = pd.DataFrame(
        {
            TARGET_FUEL: [0.0, 1.5, 2.3],
            TARGET_NONFUEL: [0.0, 0.8, 1.2],
            TARGET_OPPORTUNITY: [
                "Stable",
                "Inactivity",
                "Existing-category growth: Fuel",
            ],
        }
    )
    metrics = compute_competition_metrics(df, df)
    assert np.isclose(metrics["rmse_fuel"], 0.0)
    assert np.isclose(metrics["rmse_nonfuel"], 0.0)
    assert np.isclose(metrics["weighted_f1_opportunity"], 1.0)
    # Score = 0.3 * 0 + 0.3 * 0 + 0.4 * 1.0 = 0.40
    assert np.isclose(metrics["final_score"], 0.40)


def test_missing_column_raises():
    """Verify KeyError is raised if a required target column is missing."""
    df_true = pd.DataFrame(
        {TARGET_FUEL: [1.0], TARGET_NONFUEL: [2.0], TARGET_OPPORTUNITY: ["Stable"]}
    )
    df_pred = pd.DataFrame({TARGET_FUEL: [1.0], TARGET_NONFUEL: [2.0]})
    with pytest.raises(KeyError):
        compute_competition_metrics(df_true, df_pred)
