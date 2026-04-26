from __future__ import annotations

import os

from app.backtest.engine import BacktestEngine
from app.backtest.report import BacktestReport
from app.config.settings import Settings
from app.data.market_data_service import MarketDataService
from app.features.feature_engine import FeatureEngine
from app.ml.predictor import MLPredictor
from app.strategy.momentum_strategy import MomentumStrategy
from app.strategy.pullback_strategy import PullbackStrategy


def run_momentum_backtest():
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

    strategy = MomentumStrategy()
    engine = BacktestEngine(
        strategy=strategy,
        fee_rate=Settings.FEE_RATE,
        slippage_rate=Settings.SLIPPAGE_RATE,
        initial_cash=Settings.INITIAL_CASH,
        trade_size_usd=Settings.TRADE_SIZE_USD,
        stop_loss_pct=Settings.STOP_LOSS_PCT,
        take_profit_pct=Settings.TAKE_PROFIT_PCT,
        use_exit_signal=False,
    )

    result = engine.run(df)
    BacktestReport.print_report(result)


def run_pullback_backtest():
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

    strategy = PullbackStrategy()
    engine = BacktestEngine(
        strategy=strategy,
        fee_rate=Settings.FEE_RATE,
        slippage_rate=Settings.SLIPPAGE_RATE,
        initial_cash=Settings.INITIAL_CASH,
        trade_size_usd=Settings.TRADE_SIZE_USD,
        stop_loss_pct=Settings.STOP_LOSS_PCT,
        take_profit_pct=Settings.TAKE_PROFIT_PCT,
        use_exit_signal=True,
    )

    result = engine.run(df)
    BacktestReport.print_report(result)


def run_ml_comparison(strategy_name: str = "momentum") -> None:
    """Compare backtest avec et sans filtre ML pour voir l'impact réel."""
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

    if strategy_name == "pullback":
        strategy = PullbackStrategy()
        use_exit = True
    else:
        strategy = MomentumStrategy()
        use_exit = False

    common_params = dict(
        strategy=strategy,
        fee_rate=Settings.FEE_RATE,
        slippage_rate=Settings.SLIPPAGE_RATE,
        initial_cash=Settings.INITIAL_CASH,
        trade_size_usd=Settings.TRADE_SIZE_USD,
        stop_loss_pct=Settings.STOP_LOSS_PCT,
        take_profit_pct=Settings.TAKE_PROFIT_PCT,
        use_exit_signal=use_exit,
    )

    print(f"=== SANS filtre ML ({strategy_name}) ===")
    result_base = BacktestEngine(**common_params).run(df)
    BacktestReport.print_report(result_base)

    if not os.path.exists(Settings.ML_MODEL_PATH):
        print(f"\nModele ML introuvable : {Settings.ML_MODEL_PATH}")
        print("Lancez d'abord : python train.py")
        return

    predictor = MLPredictor(model_path=Settings.ML_MODEL_PATH)

    for threshold in [0.55, 0.60, 0.65]:
        print(f"\n=== AVEC filtre ML (threshold={threshold}) ({strategy_name}) ===")
        result_ml = BacktestEngine(
            **common_params,
            predictor=predictor,
            ml_threshold=threshold,
        ).run(df)
        BacktestReport.print_report(result_ml)
        _print_delta(result_base, result_ml)


def _print_delta(base: dict, ml: dict) -> None:
    b, m = base["summary"], ml["summary"]
    delta_wr = m["win_rate"] - b["win_rate"]
    delta_pf = m["profit_factor"] - b["profit_factor"]
    delta_trades = m["total_trades"] - b["total_trades"]
    print(
        f"  Delta win rate: {delta_wr:+.2f}%  |  "
        f"Delta profit factor: {delta_pf:+.3f}  |  "
        f"Trades: {b['total_trades']} -> {m['total_trades']} ({delta_trades:+d})"
    )