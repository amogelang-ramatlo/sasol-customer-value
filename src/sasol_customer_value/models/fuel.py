"""Specialist model for CLV_fuel (Target F).

Continuous regression predicting normalized log-fuel volume:
    z_fuel = ln(1 + y_litres) / s_fuel
"""

from dataclasses import dataclass
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


@dataclass
class FuelModel:
    """Specialist estimator for CLV_fuel continuous regression."""

    variant: Literal["best_zindi", "best_local"] = "best_zindi"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.variant == "best_zindi":
            # Proven Zindi Leaderboard Champion: 0.50 LGB / 0.30 CB / 0.20 XGB on 202 features
            self.weights = (0.50, 0.30, 0.20)
            self.n_features = 202
            self.m_lgb = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.03,
                random_state=self.seed,
                verbose=-1,
                n_jobs=-1,
            )
            self.m_cb = CatBoostRegressor(
                iterations=250,
                learning_rate=0.04,
                random_seed=self.seed,
                verbose=0,
                thread_count=-1,
            )
            self.m_xgb = XGBRegressor(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=5,
                random_state=self.seed,
                n_jobs=-1,
                verbosity=0,
            )
        elif self.variant == "best_local":
            # Original Local Validation Benchmark: 0.20 LGB / 0.55 CB / 0.25 XGB on 218 features
            self.weights = (0.20, 0.55, 0.25)
            self.n_features = 218
            self.m_lgb = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.03,
                random_state=self.seed,
                verbose=-1,
                n_jobs=-1,
            )
            self.m_cb = CatBoostRegressor(
                iterations=250,
                learning_rate=0.04,
                random_seed=self.seed,
                verbose=0,
                thread_count=-1,
            )
            self.m_xgb = XGBRegressor(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=5,
                random_state=self.seed,
                n_jobs=-1,
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown FuelModel variant: {self.variant}")

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = list(X.columns[: self.n_features])
        return X[cols]

    def fit(self, X: pd.DataFrame, y_fuel: np.ndarray) -> "FuelModel":
        """Fit all component regressors on training features and fuel target."""
        X_sel = self._select_features(X)
        self.m_lgb.fit(X_sel, y_fuel)
        self.m_cb.fit(X_sel, y_fuel)
        self.m_xgb.fit(X_sel, y_fuel)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate blended, non-negative continuous fuel predictions."""
        X_sel = self._select_features(X)
        p_lgb = np.clip(self.m_lgb.predict(X_sel), 0.0, None)
        p_cb = np.clip(self.m_cb.predict(X_sel), 0.0, None)
        p_xgb = np.clip(self.m_xgb.predict(X_sel), 0.0, None)

        w_lgb, w_cb, w_xgb = self.weights
        return w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb
