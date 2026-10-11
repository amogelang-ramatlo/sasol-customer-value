"""Specialist model for CLV_nonfuel (Target N).

Continuous regression predicting normalized log-store merchandise spend:
    z_nonfuel = ln(1 + y_rands) / s_nonfuel
"""

from dataclasses import dataclass, field
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


@dataclass
class NonfuelModel:
    """Specialist estimator for CLV_nonfuel continuous regression."""

    variant: Literal["best_zindi", "best_local"] = "best_zindi"
    seeds: list[int] = field(default_factory=lambda: [42])

    def __post_init__(self) -> None:
        self.weights = (0.40, 0.60)  # Optimal blend: 0.40 LGBM + 0.60 CatBoost
        self.n_features = 218

        if self.variant == "best_zindi":
            # Proven Zindi Leaderboard Champion: 5-seed bagged ensemble
            self.seeds = [42, 100, 2024, 777, 999]
        elif self.variant == "best_local":
            # Local Validation Champion: Single Seed 42
            self.seeds = [42]

        self.models_: list[tuple[lgb.LGBMRegressor, CatBoostRegressor]] = []

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = list(X.columns[: self.n_features])
        return X[cols]

    def fit(self, X: pd.DataFrame, y_nonfuel: np.ndarray) -> "NonfuelModel":
        """Fit bagged regressors across configured random seeds."""
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
            m_lgb.fit(X_sel, y_nonfuel)
            m_cb.fit(X_sel, y_nonfuel)
            self.models_.append((m_lgb, m_cb))

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate bagged, non-negative continuous non-fuel predictions."""
        X_sel = self._select_features(X)
        seed_preds = []

        w_lgb, w_cb = self.weights
        for m_lgb, m_cb in self.models_:
            p_lgb = np.clip(m_lgb.predict(X_sel), 0.0, None)
            p_cb = np.clip(m_cb.predict(X_sel), 0.0, None)
            seed_preds.append(w_lgb * p_lgb + w_cb * p_cb)

        return np.mean(seed_preds, axis=0)
