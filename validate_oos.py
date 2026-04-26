from __future__ import annotations

import os
import pandas as pd

from app.backtest.engine import BacktestEngine
from app.backtest.report import BacktestReport
from app.config.settings import Settings
from app.data.market_data_service import MarketDataService
from app.features.feature_engine import FeatureEngine
from app.ml.label_generator import LabelGenerator
from app.ml.predictor import MLPredictor
from app.ml.trainer import MLTrainer
from app.strategy.momentum_strategy import MomentumStrategy
from app.strategy.pullback_strategy import PullbackStrategy

TRAIN_END = pd.Timestamp("2023-01-01")
MODEL_OOS_PATH = "app/ml/model_oos.pkl"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]


def run_oos_validation(strategy_name: str = "momentum") -> None:
    # ------------------------------------------------------------------ #
    # 1. Charger et featurizer sur TOUTES les donnees (warmup EMA correct)
    # ------------------------------------------------------------------ #
    print("=== Chargement des donnees ===")
    data_service = MarketDataService(
        use_csv=Settings.USE_CSV,
        csv_filepath=Settings.CSV_FILEPATH,
        trading_pair=Settings.TRADING_PAIR,
        timeframe_minutes=Settings.TIMEFRAME_MINUTES,
    )
    df = data_service.load_ohlc()
    df = FeatureEngine.add_features(
        df,
        fast_ema=Settings.FAST_EMA,
        slow_ema=Settings.SLOW_EMA,
        rsi_period=Settings.RSI_PERIOD,
    )
    print(f"Total : {len(df):,} bougies ({df['timestamp'].min().date()} a {df['timestamp'].max().date()})")

    # ------------------------------------------------------------------ #
    # 2. Generer les labels sur toute la serie
    # ------------------------------------------------------------------ #
    print("\n=== Generation des labels ===")
    label_gen = LabelGenerator(
        stop_loss_pct=Settings.STOP_LOSS_PCT,
        take_profit_pct=Settings.TAKE_PROFIT_PCT,
        max_bars=Settings.ML_MAX_BARS,
    )
    df["label"] = label_gen.generate(df)

    # ------------------------------------------------------------------ #
    # 3. Split train / test
    # ------------------------------------------------------------------ #
    df_train = df[df["timestamp"] < TRAIN_END].copy()
    df_test = df[df["timestamp"] >= TRAIN_END].copy()

    print(f"Train : {len(df_train):,} bougies ({df_train['timestamp'].min().date()} a {df_train['timestamp'].max().date()})")
    print(f"Test  : {len(df_test):,} bougies ({df_test['timestamp'].min().date()} a {df_test['timestamp'].max().date()})")

    n_pos_train = int((df_train["label"] == 1).sum())
    n_pos_test = int((df_test["label"] == 1).sum())
    print(f"Labels train : {n_pos_train:,} positifs / {(df_train['label']==0).sum():,} negatifs")
    print(f"Labels test  : {n_pos_test:,} positifs / {(df_test['label']==0).sum():,} negatifs")

    # ------------------------------------------------------------------ #
    # 4. Entrainer sur train uniquement -> model_oos.pkl
    # ------------------------------------------------------------------ #
    print("\n=== Entrainement sur donnees TRAIN (2013-2022) ===")
    trainer = MLTrainer(model_path=MODEL_OOS_PATH)
    trainer.train(df_train)

    # ------------------------------------------------------------------ #
    # 5. Backtest sur TEST uniquement
    # ------------------------------------------------------------------ #
    print(f"\n=== Backtest sur periode TEST (2023+) — {strategy_name} ===")

    if strategy_name == "pullback":
        strategy_no_ml = PullbackStrategy()
        strategy_ml = PullbackStrategy()
        use_exit = True
    else:
        strategy_no_ml = MomentumStrategy()
        strategy_ml = MomentumStrategy()
        use_exit = False

    common = dict(
        fee_rate=Settings.FEE_RATE,
        slippage_rate=Settings.SLIPPAGE_RATE,
        initial_cash=Settings.INITIAL_CASH,
        trade_size_usd=Settings.TRADE_SIZE_USD,
        stop_loss_pct=Settings.STOP_LOSS_PCT,
        take_profit_pct=Settings.TAKE_PROFIT_PCT,
        use_exit_signal=use_exit,
    )

    print("\n--- SANS filtre ML ---")
    result_base = BacktestEngine(strategy=strategy_no_ml, **common).run(df_test)
    BacktestReport.print_report(result_base)

    predictor = MLPredictor(model_path=MODEL_OOS_PATH)

    results = {"base": result_base}
    for threshold in THRESHOLDS:
        print(f"\n--- AVEC ML (threshold={threshold}) ---")
        if strategy_name == "pullback":
            strat = PullbackStrategy()
        else:
            strat = MomentumStrategy()
        result_ml = BacktestEngine(
            strategy=strat,
            predictor=predictor,
            ml_threshold=threshold,
            **common,
        ).run(df_test)
        BacktestReport.print_report(result_ml)
        results[threshold] = result_ml

    # ------------------------------------------------------------------ #
    # 6. Tableau comparatif final
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print(f"{'':20} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'Cash':>8} {'DD':>7}")
    print("=" * 65)

    base_s = result_base["summary"]
    print(
        f"{'Sans ML':20} {base_s['total_trades']:>7} "
        f"{base_s['win_rate']:>7.1f}% "
        f"{base_s['profit_factor']:>6.3f} "
        f"{result_base['final_cash']:>8.2f} "
        f"{result_base['max_drawdown']:>6.2f}%"
    )
    for threshold in THRESHOLDS:
        r = results[threshold]
        s = r["summary"]
        print(
            f"  ML>={threshold:<16} {s['total_trades']:>7} "
            f"{s['win_rate']:>7.1f}% "
            f"{s['profit_factor']:>6.3f} "
            f"{r['final_cash']:>8.2f} "
            f"{r['max_drawdown']:>6.2f}%"
        )
    print("=" * 65)
    print("Capital initial : $1000.00")


if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "momentum"
    run_oos_validation(strategy)
