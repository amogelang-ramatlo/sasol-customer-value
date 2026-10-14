"""Specialist model for CLV_fuel (Target F).

Continuous regression predicting normalized log-fuel volume:
    z_fuel = ln(1 + y_litres) / s_fuel
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sasol_customer_value.config import DEFAULT_BAGGING_SEEDS, ModelVariant


@dataclass
class FuelModel:
    """Specialist estimator for CLV_fuel continuous regression."""

    variant: ModelVariant = "best_zindi"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.variant == "best_zindi":
            # Proven Zindi Leaderboard Champion: 0.50 LGB / 0.30 CB / 0.20 XGB on 202 features
            self.weights = (0.50, 0.30, 0.20)
            self.n_features = 202
            self.seeds = [self.seed]
        elif self.variant == "best_local":
            # Original Local Validation Benchmark: 0.20 LGB / 0.55 CB / 0.25 XGB on 218 features
            self.weights = (0.20, 0.55, 0.25)
            self.n_features = 218
            self.seeds = [self.seed]
        elif self.variant in ["champion_v2", "champion_v3"]:
            # Optimized 202-feature subspace with tuned weights (0.28 LGB, 0.72 CB) and 5-seed bagging
            self.weights = (0.28, 0.72, 0.0)
            self.n_features = 202
            self.seeds = DEFAULT_BAGGING_SEEDS
        elif self.variant in ["champion_v4", "champion_v5", "champion_v6"]:
            # Champion V4, V5 & V6: 5-seed bagged tri-family blend (0.40 LGB, 0.45 CB, 0.15 XGB) on 202 features
            self.weights = (0.40, 0.45, 0.15)
            self.n_features = 202
            self.seeds = DEFAULT_BAGGING_SEEDS
        else:
            raise ValueError(f"Unknown FuelModel variant: {self.variant}")

        self.models_: list[
            tuple[lgb.LGBMRegressor, CatBoostRegressor, XGBRegressor | None]
        ] = []

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = list(X.columns[: self.n_features])
        return X[cols]

    def fit(
        self,
        X: pd.DataFrame,
        y_fuel: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "FuelModel":
        """Fit all component regressors across configured seeds on training features and fuel target."""
        X_sel = self._select_features(X)
        self.models_ = []

        for s in self.seeds:
            m_lgb = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.03,
                random_state=s,
                verbose=-1,
                n_jobs=-1,
            )
            m_cb = CatBoostRegressor(
                iterations=250,
                learning_rate=0.04,
                random_seed=s,
                verbose=0,
                thread_count=-1,
            )
            m_xgb = None
            if self.weights[2] > 0.0:
                m_xgb = XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.03,
                    max_depth=5,
                    random_state=s,
                    n_jobs=-1,
                    verbosity=0,
                )

            m_lgb.fit(X_sel, y_fuel, sample_weight=sample_weight)
            m_cb.fit(X_sel, y_fuel, sample_weight=sample_weight)
            if m_xgb is not None:
                m_xgb.fit(X_sel, y_fuel, sample_weight=sample_weight)
            self.models_.append((m_lgb, m_cb, m_xgb))

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate blended, non-negative continuous fuel predictions."""
        X_sel = self._select_features(X)
        seed_preds = []

        w_lgb, w_cb, w_xgb = self.weights
        for m_lgb, m_cb, m_xgb in self.models_:
            p_lgb = np.clip(m_lgb.predict(X_sel), 0.0, None)
            p_cb = np.clip(m_cb.predict(X_sel), 0.0, None)
            if m_xgb is not None and w_xgb > 0.0:
                p_xgb = np.clip(m_xgb.predict(X_sel), 0.0, None)
                seed_preds.append(w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb)
            else:
                seed_preds.append(w_lgb * p_lgb + w_cb * p_cb)

        return np.mean(seed_preds, axis=0)
