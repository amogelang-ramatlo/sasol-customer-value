"""Multi-model ensemble pipeline combining LightGBM, CatBoost, and XGBoost.

Features:
1. Blended gradient boosting for CLV_fuel and CLV_nonfuel (LightGBM + CatBoost + XGBoost).
2. Frequency-calibrated multiclass probability reweighting for Weighted F1 optimization.
3. Behavioral consistency post-processing (inactivity alignment).
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sasol_customer_value.config import (
    ID_COL,
    RANDOM_SEED,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.models.metrics import compute_competition_metrics

DEFAULT_OPP_BIASES: dict[str, float] = {
    "Existing-category growth: Beverages": 1.1,
    "Existing-category growth: Food & beverage": 1.2,
    "Existing-category growth: Fuel": 0.15,
    "Existing-category growth: Other": 0.6,
    "Inactivity": 0.1,
    "New category adoption: Beverages": 1.2,
    "New category adoption: Braai & Ice": 0.5,
    "New category adoption: Breads & rolls": 1.4,
    "New category adoption: Confectionary": 0.3,
    "New category adoption: Food & beverage": -0.5,
    "New category adoption: Groceries": 1.0,
    "New category adoption: Ice-cream": 1.4,
    "New category adoption: Lubricants": 0.4,
    "New category adoption: Other": 1.4,
    "New category adoption: Snacking": -0.2,
    "New category adoption: Tobacco": 0.8,
    "Stable": 0.0,
}


@dataclass
class EnsemblePipeline:
    """Multi-model ensemble pipeline for continuous spend and calibrated opportunity classification."""

    random_seed: int = RANDOM_SEED
    class_weight: str | None = None
    class_biases: dict[str, float] | None = field(
        default_factory=lambda: dict(DEFAULT_OPP_BIASES)
    )

    # Models
    lgb_fuel: lgb.LGBMRegressor = field(init=False)
    cb_fuel: CatBoostRegressor = field(init=False)
    xgb_fuel: XGBRegressor = field(init=False)

    lgb_nonfuel: lgb.LGBMRegressor = field(init=False)
    cb_nonfuel: CatBoostRegressor = field(init=False)
    xgb_nonfuel: XGBRegressor = field(init=False)

    opp_model: lgb.LGBMClassifier = field(init=False)

    feature_names_: list[str] = field(default_factory=list, init=False)
    class_weights_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    classes_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)

    def __post_init__(self) -> None:
        # Fuel Regressors
        self.lgb_fuel = lgb.LGBMRegressor(
            random_state=self.random_seed,
            n_estimators=150,
            learning_rate=0.03,
            verbose=-1,
        )
        self.cb_fuel = CatBoostRegressor(
            random_seed=self.random_seed,
            iterations=250,
            learning_rate=0.04,
            verbose=0,
        )
        self.xgb_fuel = XGBRegressor(
            random_state=self.random_seed,
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
        )

        # Non-Fuel Regressors
        self.lgb_nonfuel = lgb.LGBMRegressor(
            random_state=self.random_seed,
            n_estimators=150,
            learning_rate=0.03,
            verbose=-1,
        )
        self.cb_nonfuel = CatBoostRegressor(
            random_seed=self.random_seed,
            iterations=250,
            learning_rate=0.04,
            verbose=0,
        )
        self.xgb_nonfuel = XGBRegressor(
            random_state=self.random_seed,
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
        )

        # Multiclass Opportunity Classifier
        self.opp_model = lgb.LGBMClassifier(
            random_state=self.random_seed,
            num_class=17,
            class_weight=self.class_weight,
            n_estimators=120,
            learning_rate=0.04,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "EnsemblePipeline":
        """Fit all ensemble component models."""
        self.feature_names_ = list(X.columns)

        y_fuel = y[TARGET_FUEL].to_numpy(dtype=float)
        y_nonfuel = y[TARGET_NONFUEL].to_numpy(dtype=float)
        y_opp = y[TARGET_OPPORTUNITY].to_numpy(dtype=str)

        # 1. Fit Fuel models
        self.lgb_fuel.fit(X, y_fuel)
        self.cb_fuel.fit(X, y_fuel)
        self.xgb_fuel.fit(X, y_fuel)

        # 2. Fit Non-fuel models
        self.lgb_nonfuel.fit(X, y_nonfuel)
        self.cb_nonfuel.fit(X, y_nonfuel)
        self.xgb_nonfuel.fit(X, y_nonfuel)

        # 3. Fit Opportunity classifier
        self.opp_model.fit(X, y_opp)
        self.classes_ = self.opp_model.classes_

        # 4. Calibrate class threshold weights (power law inverse frequency)
        class_counts = (
            pd.Series(y_opp).value_counts().reindex(self.classes_).fillna(1).values
        )
        weights = 1.0 / (class_counts**0.18)
        self.class_weights_ = weights / weights.mean()

        return self

    def predict_components(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw (pred_fuel, pred_nonfuel, opp_probs) before bias and postprocessing."""
        X_eval = X[self.feature_names_].copy()

        # 1. Blend Fuel predictions (optimal weights: CatBoost 0.55, XGBoost 0.25, LightGBM 0.20)
        p_lgb_f = np.clip(self.lgb_fuel.predict(X_eval), 0.0, None)
        p_cb_f = np.clip(self.cb_fuel.predict(X_eval), 0.0, None)
        p_xgb_f = np.clip(self.xgb_fuel.predict(X_eval), 0.0, None)
        pred_fuel = 0.20 * p_lgb_f + 0.55 * p_cb_f + 0.25 * p_xgb_f

        # 2. Blend Non-Fuel predictions (optimal weights: CatBoost 0.60, LightGBM 0.40)
        p_lgb_nf = np.clip(self.lgb_nonfuel.predict(X_eval), 0.0, None)
        p_cb_nf = np.clip(self.cb_nonfuel.predict(X_eval), 0.0, None)
        pred_nonfuel = 0.40 * p_lgb_nf + 0.60 * p_cb_nf

        # 3. Opportunity probabilities
        probs = np.asarray(self.opp_model.predict_proba(X_eval))
        return pred_fuel, pred_nonfuel, probs

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate ensembled, calibrated predictions."""
        X_eval = X[self.feature_names_].copy()
        pred_fuel, pred_nonfuel, probs = self.predict_components(X)

        if self.class_biases:
            bias_vec = np.array(
                [self.class_biases.get(c, 0.0) for c in self.classes_], dtype=float
            )
            log_probs = np.log(np.clip(probs, 1e-9, 1.0)) + bias_vec
            pred_opp = self.classes_[log_probs.argmax(axis=1)].copy()
        else:
            pred_opp = self.classes_[probs.argmax(axis=1)].copy()

        # Inactivity post-processing consistency
        if "days_since_last_visit" in X_eval.columns:
            inactive_mask = (
                (pred_fuel < 0.10)
                & (pred_nonfuel < 0.10)
                & (X_eval["days_since_last_visit"] > 60.0)
            )
            pred_opp[inactive_mask] = "Inactivity"

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
        """Evaluate ensemble predictions against ground truth."""
        y_pred = self.predict(X)
        return compute_competition_metrics(y_true, y_pred)
