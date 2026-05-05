from __future__ import annotations

import numpy as np
import pandas as pd

from app.config.settings import Settings
from app.data.market_data_service import MarketDataService
from app.features.feature_engine import FeatureEngine
from app.ml.label_generator import LabelGenerator
from app.ml.trainer import MLTrainer

TRAIN_END = pd.Timestamp("2025-09-01")   # train 2013 -> aout 2025
OOS_START = pd.Timestamp("2025-09-01")   # OOS  sept-dec 2025


def run_ml_training() -> None:
    print("=== Chargement des données ===")
    ds = MarketDataService(
        use_csv=Settings.USE_CSV,
        csv_filepath=Settings.CSV_FILEPATH,
        trading_pair=Settings.TRADING_PAIR,
        timeframe_minutes=Settings.TIMEFRAME_MINUTES,
    )
    df = ds.load_ohlc()
    print(f"{len(df):,} bougies  ({df['timestamp'].min().date()} -> {df['timestamp'].max().date()})")

    print("\n=== Features ===")
    df = FeatureEngine.add_features(
        df,
        fast_ema=Settings.FAST_EMA,
        slow_ema=Settings.SLOW_EMA,
        rsi_period=Settings.RSI_PERIOD,
    )

    print("\n=== Labels (SL/TP forward scan, 3 jours max) ===")
    # SL=1%, TP=4.5% (seuil minimum "bon trade"), fenetre 72h
    label_gen = LabelGenerator(
        stop_loss_pct=0.010,
        take_profit_pct=0.045,
        max_bars=288,
    )
    df["label"] = label_gen.generate(df)

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    print(f"  Positifs (TP atteint) : {n_pos:,}  ({n_pos/(n_pos+n_neg)*100:.1f}%)")
    print(f"  Négatifs (SL atteint) : {n_neg:,}  ({n_neg/(n_pos+n_neg)*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # Split temporel strict
    # ------------------------------------------------------------------ #
    df_train = df[df["timestamp"] < TRAIN_END].copy()
    df_oos   = df[df["timestamp"] >= OOS_START].copy()
    print(f"\n  Train : {len(df_train):,} bougies  ({df_train['timestamp'].min().date()} -> {TRAIN_END.date()})")
    print(f"  OOS   : {len(df_oos):,} bougies  ({OOS_START.date()} -> {df_oos['timestamp'].max().date()})")

    # ------------------------------------------------------------------ #
    # Entraînement
    # ------------------------------------------------------------------ #
    print("\n=== Entraînement sur 2013-2024 ===")
    trainer = MLTrainer(model_path="app/ml/model_oos.pkl", n_splits=5)
    trainer.train(df_train)

    # ------------------------------------------------------------------ #
    # Validation OOS 2025
    # ------------------------------------------------------------------ #
    print("\n=== Validation OOS 2025 ===")
    from app.ml.trainer import FEATURE_COLS
    df_oos_clean = df_oos[df_oos["label"] != -1].dropna(subset=FEATURE_COLS + ["label"])

    if len(df_oos_clean) == 0:
        print("  Pas assez de données OOS.")
        return

    X_oos = df_oos_clean[FEATURE_COLS].astype(float)
    y_oos = df_oos_clean["label"].astype(int)

    from sklearn.metrics import roc_auc_score, classification_report
    proba = trainer.model.predict_proba(X_oos)[:, 1]
    auc   = roc_auc_score(y_oos, proba)

    n_pos_oos = int(y_oos.sum())
    threshold = np.percentile(proba, 100 * (1 - n_pos_oos / len(y_oos)))
    preds = (proba >= threshold).astype(int)
    report = classification_report(y_oos, preds, output_dict=True, zero_division=0)

    print(f"  AUC OOS 2025   : {auc:.4f}")
    print(f"  Precision(win) : {report['1']['precision']:.3f}")
    print(f"  Recall(win)    : {report['1']['recall']:.3f}")
    print(f"  Seuil adaptatif: {threshold:.3f}")
    print(f"\n  Modèle sauvegardé : app/ml/model_oos.pkl")
    print("\n=== Terminé ===")


if __name__ == "__main__":
    run_ml_training()
