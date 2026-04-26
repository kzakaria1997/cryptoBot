from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from app.ml.trainer import FEATURE_COLS


class MLPredictor:
    def __init__(self, model_path: str = "app/ml/model.pkl"):
        self.model = joblib.load(model_path)

    def predict_proba(self, row: pd.Series) -> float:
        """Retourne la probabilité que ce soit un bon trade (TP atteint avant SL)."""
        X = row[FEATURE_COLS].values.reshape(1, -1).astype(float)
        return float(self.model.predict_proba(X)[0, 1])

    def predict_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Score toutes les lignes d'un DataFrame. Retourne une Series de probabilités."""
        X = df[FEATURE_COLS].astype(float)
        valid_mask = X.notna().all(axis=1)
        scores = pd.Series(0.0, index=df.index, name="ml_score")
        if valid_mask.any():
            scores[valid_mask] = self.model.predict_proba(X[valid_mask])[:, 1]
        return scores
