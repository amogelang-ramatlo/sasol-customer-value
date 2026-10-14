"""Specialist model for CLV_nonfuel (Target N).

Continuous regression predicting normalized log-store merchandise spend:
    z_nonfuel = ln(1 + y_rands) / s_nonfuel
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sasol_customer_value.config import DEFAULT_BAGGING_SEEDS, ModelVariant


@dataclass
class NonfuelModel:
    """Specialist estimator for CLV_nonfuel continuous regression."""

    variant: ModelVariant = "best_zindi"
    seeds: list[int] = field(default_factory=lambda: [42])

    def __post_init__(self) -> None:
        if self.variant == "best_zindi":
            # Proven Zindi Leaderboard Champion: 5-seed bagged 0.40 LGB / 0.60 CB on 218 features
            self.weights = (0.40, 0.60, 0.0)
            self.n_features = 218
            self.seeds = DEFAULT_BAGGING_SEEDS
        elif self.variant == "best_local":
            # Original Local Validation Benchmark: 0.40 LGB / 0.60 CB on 218 features (Seed 42)
            self.weights = (0.40, 0.60, 0.0)
            self.n_features = 218
            self.seeds = [42]
        elif self.variant in ["champion_v2", "champion_v3"]:
            # Optimized 218-feature subspace with tuned weights (0.07 LGB, 0.93 CB) and 5-seed bagging
            self.weights = (0.07, 0.93, 0.0)
            self.n_features = 218
            self.seeds = DEFAULT_BAGGING_SEEDS
        elif self.variant == "champion_v4":
            # Champion V4: Balanced 5-seed bagged blend (0.25 LGB, 0.75 CB) on 218 features
            self.weights = (0.25, 0.75, 0.0)
            self.n_features = 218
            self.seeds = DEFAULT_BAGGING_SEEDS
        elif self.variant in ["champion_v5", "champion_v6"]:
            # Champion V5 & V6: CatBoost-dominated 5-seed bagged blend (0.05 LGB, 0.95 CB) on 218 features
            # Strongly regularizes against variance spikes on 65% zero-inflated store spend.
            self.weights = (0.05, 0.95, 0.0)
            self.n_features = 218
            self.seeds = DEFAULT_BAGGING_SEEDS
        else:
            raise ValueError(f"Unknown NonfuelModel variant: {self.variant}")

        self.models_: list[
            tuple[lgb.LGBMRegressor, CatBoostRegressor, XGBRegressor | None]
        ] = []

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = list(X.columns[: self.n_features])
        return X[cols]

    def fit(
        self,
        X: pd.DataFrame,
        y_nonfuel: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "NonfuelModel":
        """Fit regressors across configured random seeds."""
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
                m_xgb.fit(X_sel, y_nonfuel, sample_weight=sample_weight)

            m_lgb.fit(X_sel, y_nonfuel, sample_weight=sample_weight)
            m_cb.fit(X_sel, y_nonfuel, sample_weight=sample_weight)
            self.models_.append((m_lgb, m_cb, m_xgb))

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate bagged, non-negative continuous non-fuel predictions."""
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
