from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report

FEATURE_COLS = [
    # RSI court terme et lags
    "rsi", "rsi_lag1", "rsi_lag2", "rsi_lag3",
    # RSI moyen terme (regime momentum)
    "rsi_slow",
    # Volume
    "rel_volume", "vol_ratio_5",
    # Retours court terme et lags
    "return_1", "return_3", "return_6",
    "return_1_lag1", "return_1_lag2", "return_1_lag3",
    # Retours multi-timeframes (REGIME)
    "return_1d", "return_7d", "return_30d",
    # Volatilite court/long terme + regime vol
    "volatility_10", "vol_long", "vol_regime",
    "range_pct", "atr_pct",
    # MACD normalise
    "macd_pct", "macd_signal_pct", "macd_hist_pct",
    # Bollinger Bands
    "bb_width", "bb_pct",
    # Stochastique
    "stoch_k", "stoch_d",
    # Distances EMA (%)
    "dist_ema_fast_pct", "dist_ema_slow_pct", "ema_gap_pct", "trend_gap_pct",
    # Position dans le range 20 jours (REGIME)
    "dist_high_20d", "dist_low_20d", "pos_in_range_20d",
    # Flag bull/bear macro (REGIME)
    "regime_bull",
    # Temps
    "hour", "dayofweek",
]


class MLTrainer:
    def __init__(self, model_path: str = "app/ml/model.pkl", n_splits: int = 5):
        self.model_path = model_path
        self.n_splits = n_splits
        self.model: LGBMClassifier | None = None

    def train(self, df: pd.DataFrame) -> None:
        df = df[df["label"] != -1].dropna(subset=FEATURE_COLS + ["label"])

        X = df[FEATURE_COLS].astype(float)
        y = df["label"].astype(int)

        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        print(f"Dataset : {len(df):,} lignes | Positifs (TP): {n_pos:,} ({n_pos/len(y)*100:.1f}%) | Négatifs (SL): {n_neg:,}")

        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        auc_scores = []

        print(f"\n--- Walk-forward CV ({self.n_splits} folds) ---")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.02,
                num_leaves=63,
                min_child_samples=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                metric="auc",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(-1)],
            )

            proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            auc_scores.append(auc)

            # Seuil adaptatif : top 23% (proportion de positifs dans le dataset)
            threshold = np.percentile(proba, 100 * (1 - n_pos / (n_pos + n_neg)))
            preds = (proba >= threshold).astype(int)
            report = classification_report(y_val, preds, output_dict=True, zero_division=0)
            precision = report["1"]["precision"]
            recall = report["1"]["recall"]
            print(f"Fold {fold} | AUC: {auc:.4f} | Precision(1): {precision:.3f} | Recall(1): {recall:.3f} | best_iter: {model.best_iteration_}")

        print(f"\nAUC moyen : {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

        # Entraînement final sur tout le dataset
        print("\n--- Entraînement final sur 100% des données ---")
        self.model = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            metric="auc",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(X, y)

        joblib.dump(self.model, self.model_path)
        print(f"Modèle sauvegardé : {self.model_path}")

        importances = pd.Series(
            self.model.feature_importances_,
            index=FEATURE_COLS,
        ).sort_values(ascending=False)
        print("\n--- Top 15 features ---")
        print(importances.head(15).to_string())
