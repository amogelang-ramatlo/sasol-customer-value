"""Baseline LightGBM models for CLV fuel, CLV non-fuel, and Opportunity prediction.

Provides three baseline models conforming to the competition multi-task objective:
1. LGBMRegressor for CLV_fuel (RMSE on z)
2. LGBMRegressor for CLV_nonfuel (RMSE on z)
3. LGBMClassifier for Opportunity (Weighted F1 on 17 classes)
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from sasol_customer_value.config import (
    ID_COL,
    RANDOM_SEED,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.data.labels import generate_cutoff_labels
from sasol_customer_value.features.rfm import extract_rfm_features
from sasol_customer_value.models.metrics import compute_competition_metrics


@dataclass
class BaselinePipeline:
    """End-to-end container managing the three LightGBM baseline models."""

    random_seed: int = RANDOM_SEED
    fuel_model: lgb.LGBMRegressor = field(init=False)
    nonfuel_model: lgb.LGBMRegressor = field(init=False)
    opp_model: lgb.LGBMClassifier = field(init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.fuel_model = lgb.LGBMRegressor(
            random_state=self.random_seed,
            objective="regression",
            metric="rmse",
            n_estimators=100,
            learning_rate=0.05,
            verbose=-1,
        )
        self.nonfuel_model = lgb.LGBMRegressor(
            random_state=self.random_seed,
            objective="regression",
            metric="rmse",
            n_estimators=100,
            learning_rate=0.05,
            verbose=-1,
        )
        self.opp_model = lgb.LGBMClassifier(
            random_state=self.random_seed,
            objective="multiclass",
            num_class=17,
            class_weight=None,
            n_estimators=100,
            learning_rate=0.05,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "BaselinePipeline":
        """Fit all three baseline models on feature matrix X and target matrix y.

        Args:
            X: Feature matrix indexed by customer ID.
            y: Target DataFrame with columns [CLV_fuel, CLV_nonfuel, Opportunity].
        """
        # Align indices
        common_ids = X.index.intersection(
            y[ID_COL].astype(str) if ID_COL in y.columns else y.index.astype(str)
        )
        if len(common_ids) == 0:
            raise ValueError(
                "No matching customer IDs between features X and targets y."
            )

        X_aligned = X.loc[common_ids].copy()
        if ID_COL in y.columns:
            y_indexed = y.set_index(y[ID_COL].astype(str))
        else:
            y_indexed = y.copy()
            y_indexed.index = y_indexed.index.astype(str)
        y_aligned = y_indexed.loc[common_ids]

        self.feature_names_ = list(X_aligned.columns)

        # 1. Fit CLV_fuel
        self.fuel_model.fit(
            X_aligned,
            y_aligned[TARGET_FUEL].to_numpy(dtype=float),
        )

        # 2. Fit CLV_nonfuel
        self.nonfuel_model.fit(
            X_aligned,
            y_aligned[TARGET_NONFUEL].to_numpy(dtype=float),
        )

        # 3. Fit Opportunity
        self.opp_model.fit(
            X_aligned,
            y_aligned[TARGET_OPPORTUNITY].to_numpy(dtype=str),
        )

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions complying with competition constraints.

        Args:
            X: Feature matrix indexed by customer ID.

        Returns:
            DataFrame with columns: [ID, CLV_fuel, CLV_nonfuel, Opportunity].
        """
        X_eval = X[self.feature_names_].copy()

        # Continuous predictions clipped at 0.0 (non-negative constraint)
        pred_fuel = np.clip(self.fuel_model.predict(X_eval), 0.0, None)
        pred_nonfuel = np.clip(self.nonfuel_model.predict(X_eval), 0.0, None)

        # Multiclass prediction
        pred_opp = self.opp_model.predict(X_eval)

        out = pd.DataFrame(
            {
                ID_COL: X.index.astype(str),
                TARGET_FUEL: pred_fuel,
                TARGET_NONFUEL: pred_nonfuel,
                TARGET_OPPORTUNITY: pred_opp,
            }
        )
        return out

    def evaluate(self, X: pd.DataFrame, y_true: pd.DataFrame) -> dict[str, float]:
        """Generate predictions and compute the official competition metrics."""
        y_pred = self.predict(X)
        return compute_competition_metrics(y_true, y_pred)


def prepare_snapshot(
    transactions_df: pd.DataFrame,
    cutoff_date: str,
    use_behavioral: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both ground-truth targets and leak-free features for a cutoff date.

    Args:
        transactions_df: Loaded transaction DataFrame.
        cutoff_date: Snapshot cutoff (e.g. '2025-06-01' or '2025-09-01').
        use_behavioral: If True, uses cached/built snapshot with RFM + behavioral features.

    Returns:
        Tuple of (X_features, y_targets).
    """
    if use_behavioral:
        from sasol_customer_value.data.snapshots import build_snapshot

        return build_snapshot(transactions_df, cutoff=cutoff_date)

    _, y_norm = generate_cutoff_labels(transactions_df, cutoff_date)
    eligible_ids = pd.Index(y_norm[ID_COL].astype(str))

    X_feats = extract_rfm_features(
        transactions_df,
        cutoff_date=cutoff_date,
        customer_ids=eligible_ids,
    )
    return X_feats, y_norm
