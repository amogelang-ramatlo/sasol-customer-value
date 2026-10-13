"""Specialist model for Opportunity Classification (Target O).

17-Class mutually exclusive classifier combining LightGBM estimators,
summer holiday calibrated log-odds biases, and majority committee voting.
"""

from collections import Counter
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from sasol_customer_value.config import ModelVariant
from sasol_customer_value.models.ensemble import DEFAULT_OPP_BIASES


@dataclass
class OpportunityModel:
    """Specialist committee classifier for the 17-class Opportunity target."""

    variant: ModelVariant = "best_zindi"
    seeds: list[int] = field(default_factory=lambda: [42])

    def __post_init__(self) -> None:
        if self.variant in ["best_zindi", "champion_v2"]:
            # 5-Seed multi-architecture committee
            self.seeds_A = [42, 100, 2024, 777, 999]
            self.seeds_B = [42, 100, 2024, 777, 999]
            self.seeds_C = [42, 100, 2024, 777, 999]
        elif self.variant == "best_local":
            # Pure single-seed committee (Seed 42)
            self.seeds_A = [42]
            self.seeds_B = [42]
            self.seeds_C = [42]

        self.models_A: list[lgb.LGBMClassifier] = []
        self.models_B: list[lgb.LGBMClassifier] = []
        self.models_C: list[lgb.LGBMClassifier] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y_opp: np.ndarray) -> "OpportunityModel":
        """Fit all committee members across their respective feature subspaces."""
        cols_202 = list(X.columns[:202])
        cols_218 = list(X.columns[:218])
        cols_225 = list(X.columns[:225])

        self.models_A = []
        for s in self.seeds_A:
            clf = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.04,
                num_class=17,
                random_state=s,
                verbose=-1,
                n_jobs=-1,
            )
            clf.fit(X[cols_202], y_opp)
            self.models_A.append(clf)

        self.models_B = []
        for s in self.seeds_B:
            clf = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.04,
                num_class=17,
                random_state=s,
                verbose=-1,
                n_jobs=-1,
            )
            clf.fit(X[cols_218], y_opp)
            self.models_B.append(clf)

        self.models_C = []
        for s in self.seeds_C:
            clf = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.04,
                num_class=17,
                random_state=s,
                verbose=-1,
                n_jobs=-1,
            )
            clf.fit(X[cols_225], y_opp)
            self.models_C.append(clf)

        self.classes_ = self.models_A[0].classes_
        return self

    def _predict_component(
        self, models: list[lgb.LGBMClassifier], X_sub: pd.DataFrame
    ) -> np.ndarray:
        """Average probabilities across models and apply summer log-odds biases."""
        if self.classes_ is None:
            raise RuntimeError(
                "OpportunityModel must be fitted before making predictions."
            )
        probs = np.mean([m.predict_proba(X_sub) for m in models], axis=0)
        bias_vec = np.array(
            [float(DEFAULT_OPP_BIASES.get(str(c), 0.0)) for c in self.classes_],
            dtype=np.float64,
        )
        log_probs = np.log(np.clip(probs, 1e-9, 1.0)) + bias_vec
        return np.asarray(self.classes_[log_probs.argmax(axis=1)])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Form committee majority consensus across Model A, B, and C."""
        if self.classes_ is None:
            raise RuntimeError(
                "OpportunityModel must be fitted before calling predict."
            )

        cols_202 = list(X.columns[:202])
        cols_218 = list(X.columns[:218])
        cols_225 = list(X.columns[:225])

        pred_A = self._predict_component(self.models_A, X[cols_202])
        pred_B = self._predict_component(self.models_B, X[cols_218])
        pred_C = self._predict_component(self.models_C, X[cols_225])

        consensus = []
        for i in range(len(X)):
            counts = Counter([pred_A[i], pred_B[i], pred_C[i]])
            top_cand, top_count = counts.most_common(1)[0]
            # If 2 or 3 agree, pick majority. On 3-way tie, default to Model B.
            consensus.append(top_cand if top_count >= 2 else pred_B[i])

        return np.array(consensus)
