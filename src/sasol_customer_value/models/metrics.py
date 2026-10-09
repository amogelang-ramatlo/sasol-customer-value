"""Authoritative evaluation metric for the Sasol Customer Value competition.

Metric:
    Final Score = 0.30 * RMSE(CLV_fuel) + 0.30 * RMSE(CLV_nonfuel) + 0.40 * Weighted_F1(Opportunity)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error

from sasol_customer_value.config import (
    METRIC_WEIGHTS,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)


def compute_competition_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> dict[str, float]:
    """Calculate the official competition loss and component metrics.

    Args:
        y_true: DataFrame containing ground truth columns [CLV_fuel, CLV_nonfuel, Opportunity].
        y_pred: DataFrame containing predicted columns [CLV_fuel, CLV_nonfuel, Opportunity].

    Returns:
        Dict with keys: 'final_score', 'rmse_fuel', 'rmse_nonfuel', 'weighted_f1_opportunity'.
    """
    # Verify required targets are present
    for target in [TARGET_FUEL, TARGET_NONFUEL, TARGET_OPPORTUNITY]:
        if target not in y_true.columns:
            raise KeyError(f"Missing required target column '{target}' in y_true")
        if target not in y_pred.columns:
            raise KeyError(f"Missing required target column '{target}' in y_pred")

    # 1. RMSE CLV_fuel
    true_fuel = y_true[TARGET_FUEL].to_numpy(dtype=float)
    pred_fuel = y_pred[TARGET_FUEL].to_numpy(dtype=float)
    rmse_fuel = float(np.sqrt(mean_squared_error(true_fuel, pred_fuel)))

    # 2. RMSE CLV_nonfuel
    true_nonfuel = y_true[TARGET_NONFUEL].to_numpy(dtype=float)
    pred_nonfuel = y_pred[TARGET_NONFUEL].to_numpy(dtype=float)
    rmse_nonfuel = float(np.sqrt(mean_squared_error(true_nonfuel, pred_nonfuel)))

    # 3. Weighted F1 Opportunity
    true_opp = y_true[TARGET_OPPORTUNITY].astype(str).to_numpy()
    pred_opp = y_pred[TARGET_OPPORTUNITY].astype(str).to_numpy()
    f1_opp = float(
        f1_score(
            true_opp,
            pred_opp,
            average="weighted",
            zero_division=0,
        )
    )

    # Combined score
    w_fuel = METRIC_WEIGHTS[TARGET_FUEL]
    w_nonfuel = METRIC_WEIGHTS[TARGET_NONFUEL]
    w_opp = METRIC_WEIGHTS[TARGET_OPPORTUNITY]

    final_score = (w_fuel * rmse_fuel) + (w_nonfuel * rmse_nonfuel) + (w_opp * f1_opp)

    return {
        "final_score": float(final_score),
        "rmse_fuel": rmse_fuel,
        "rmse_nonfuel": rmse_nonfuel,
        "weighted_f1_opportunity": f1_opp,
    }
