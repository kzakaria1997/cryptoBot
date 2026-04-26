from __future__ import annotations

from app.config.settings import Settings
from app.data.market_data_service import MarketDataService
from app.features.feature_engine import FeatureEngine
from app.ml.label_generator import LabelGenerator
from app.ml.trainer import MLTrainer


def run_ml_training() -> None:
    print("=== Chargement des données ===")
    data_service = MarketDataService(
        use_csv=Settings.USE_CSV,
        csv_filepath=Settings.CSV_FILEPATH,
        trading_pair=Settings.TRADING_PAIR,
        timeframe_minutes=Settings.TIMEFRAME_MINUTES,
    )
    df = data_service.load_ohlc()
    print(f"{len(df):,} bougies chargees ({df['timestamp'].min()} a {df['timestamp'].max()})")

    print("\n=== Calcul des features ===")
    df = FeatureEngine.add_features(
        df,
        fast_ema=Settings.FAST_EMA,
        slow_ema=Settings.SLOW_EMA,
        rsi_period=Settings.RSI_PERIOD,
    )

    print("\n=== Génération des labels (TP/SL forward scan) ===")
    print(f"Stop loss: {Settings.STOP_LOSS_PCT*100:.2f}% | Take profit: {Settings.TAKE_PROFIT_PCT*100:.2f}% | Max bars: {Settings.ML_MAX_BARS}")
    label_gen = LabelGenerator(
        stop_loss_pct=Settings.STOP_LOSS_PCT,
        take_profit_pct=Settings.TAKE_PROFIT_PCT,
        max_bars=Settings.ML_MAX_BARS,
    )
    df["label"] = label_gen.generate(df)
    print(f"Labels générés : {(df['label']==1).sum():,} positifs / {(df['label']==0).sum():,} négatifs")

    print("\n=== Entraînement du modèle ===")
    trainer = MLTrainer(model_path=Settings.ML_MODEL_PATH)
    trainer.train(df)

    print("\n=== Terminé ===")


if __name__ == "__main__":
    run_ml_training()
