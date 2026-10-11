"""Two-Stage Hurdle Regressor for Zero-Inflated Customer Spend Targets.

Mathematical Formulation:
Let z in [0, inf) be the normalized target spend (CLV_fuel or CLV_nonfuel).
Because 29.2% of fuel and 65.1% of store spend targets are exact zeros,
the optimal L2 MSE estimate decomposes into:
    E[z | x] = P(z > 0 | x) * E[z | z > 0, x]

Architecture:
1. Stage 1 (Classification Gate):
   Calibrated binary LightGBM/CatBoost classifier estimating P(z > 0 | x).
2. Stage 2 (Conditional Regressor):
   L2 Regression model trained strictly on positive spend samples (z > 0).
3. Hurdle Prediction:
   z_hurdle = p_gate * z_pos
4. Ensemble Blend:
   z_final = alpha * z_direct + (1 - alpha) * z_hurdle
   where alpha in [0, 1] is tuned out-of-fold.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


@dataclass
class HurdleRegressor:
    """Two-stage hurdle model for zero-inflated continuous targets."""

    model_type: str = "lgbm"  # "lgbm", "catboost", "blend"
    random_state: int = 42
    threshold_zero: float = 1e-6

    def __post_init__(self):
        if self.model_type == "lgbm":
            self.clf = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.03,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            self.reg = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.03,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif self.model_type == "catboost":
            self.clf = CatBoostClassifier(
                iterations=200,
                learning_rate=0.04,
                depth=5,
                random_seed=self.random_state,
                verbose=0,
                thread_count=-1,
            )
            self.reg = CatBoostRegressor(
                iterations=250,
                learning_rate=0.04,
                depth=5,
                random_seed=self.random_state,
                verbose=0,
                thread_count=-1,
            )

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "HurdleRegressor":
        y_arr = np.asarray(y, dtype=float)
        y_binary = (y_arr > self.threshold_zero).astype(int)

        # 1. Fit Stage 1 (Binary Zero/Non-Zero Gate)
        self.clf.fit(X, y_binary)

        # 2. Fit Stage 2 (Conditional Positive Spend Regressor)
        pos_mask = y_binary == 1
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]

        if len(y_pos) > 0:
            self.reg.fit(X_pos, y_pos)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.asarray(self.clf.predict_proba(X))
        p_active = probs[:, 1]
        pred_pos = np.clip(self.reg.predict(X), 0.0, None)
        return p_active * pred_pos


@dataclass
class BaggedHurdleRegressor:
    """Multi-seed bagged hurdle regressor combining GBDT architectures."""

    seeds: tuple[int, ...] = (42, 100, 2024, 777, 999)

    def fit(
        self, X: pd.DataFrame, y: np.ndarray | pd.Series
    ) -> "BaggedHurdleRegressor":
        y_arr = np.asarray(y, dtype=float)
        self.models_ = []
        for s in self.seeds:
            # 1. LightGBM hurdle
            m_lgb = HurdleRegressor(model_type="lgbm", random_state=s)
            m_lgb.fit(X, y_arr)

            # 2. CatBoost hurdle
            m_cb = HurdleRegressor(model_type="catboost", random_state=s)
            m_cb.fit(X, y_arr)

            self.models_.append((m_lgb, m_cb))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        seed_preds = []
        for m_lgb, m_cb in self.models_:
            p_lgb = m_lgb.predict(X)
            p_cb = m_cb.predict(X)
            seed_preds.append(0.50 * p_lgb + 0.50 * p_cb)
        return np.mean(seed_preds, axis=0)
