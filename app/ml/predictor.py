from __future__ import annotations

import joblib
import pandas as pd

# Import lazy pour eviter crash si lightgbm non dispo sur le serveur
def _get_feature_cols():
    from app.ml.trainer import FEATURE_COLS
    return FEATURE_COLS


class MLPredictor:
    def __init__(self, model_path: str = "app/ml/model.pkl"):
        self.model = joblib.load(model_path)
        self._feature_cols = _get_feature_cols()

    def predict_proba(self, row: pd.Series) -> float:
        X = row[self._feature_cols].values.reshape(1, -1).astype(float)
        return float(self.model.predict_proba(X)[0, 1])

    def predict_dataframe(self, df: pd.DataFrame) -> pd.Series:
        X = df[self._feature_cols].astype(float)
        valid_mask = X.notna().all(axis=1)
        scores = pd.Series(0.0, index=df.index, name="ml_score")
        if valid_mask.any():
            scores[valid_mask] = self.model.predict_proba(X[valid_mask])[:, 1]
        return scores
