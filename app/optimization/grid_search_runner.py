from __future__ import annotations

from itertools import product

from app.backtest.engine import BacktestEngine
from app.config.settings import Settings
from app.data.market_data_service import MarketDataService
from app.features.feature_engine import FeatureEngine
from app.strategy.momentum_strategy import MomentumStrategy


def run_grid_search():
    print("Loading market data...")
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

    param_grid = {
        "rsi_min": [52, 54],
        "rsi_max": [58, 60],
        "rel_volume_min": [1.2, 1.35],
        "return_3_min": [0.0015, 0.0025],
        "volatility_10_min": [0.001, 0.0015],
        "trend_gap_min": [0.0, 0.001],
    }

    risk_grid = {
        "stop_loss_pct": [0.005, 0.006],
        "take_profit_pct": [0.012, 0.015],
    }

    strategy_keys = list(param_grid.keys())
    strategy_values = list(product(*(param_grid[key] for key in strategy_keys)))

    risk_keys = list(risk_grid.keys())
    risk_values = list(product(*(risk_grid[key] for key in risk_keys)))

    all_results = []
    total_tests = len(strategy_values) * len(risk_values)
    current_test = 0

    print(f"Starting grid search: {total_tests} combinations")

    for strat_combo in strategy_values:
        strategy_params = dict(zip(strategy_keys, strat_combo))

        for risk_combo in risk_values:
            current_test += 1
            risk_params = dict(zip(risk_keys, risk_combo))

            strategy = MomentumStrategy(strategy_params)

            engine = BacktestEngine(
                strategy=strategy,
                fee_rate=Settings.FEE_RATE,
                slippage_rate=Settings.SLIPPAGE_RATE,
                initial_cash=Settings.INITIAL_CASH,
                trade_size_usd=Settings.TRADE_SIZE_USD,
                stop_loss_pct=risk_params["stop_loss_pct"],
                take_profit_pct=risk_params["take_profit_pct"],
                use_exit_signal=False,
            )

            result = engine.run(df)
            summary = result["summary"]

            row = {
                "test_number": current_test,
                "strategy_params": strategy_params,
                "risk_params": risk_params,
                "final_cash": result["final_cash"],
                "total_pnl": summary["total_pnl"],
                "profit_factor": summary["profit_factor"],
                "win_rate": summary["win_rate"],
                "total_trades": summary["total_trades"],
                "max_drawdown": result["max_drawdown"],
                "wins": summary["wins"],
                "losses": summary["losses"],
            }

            all_results.append(row)

            if current_test % 50 == 0:
                print(f"Progress: {current_test}/{total_tests}")

    # Tri sérieux:
    # 1. Profit factor élevé
    # 2. Cash final élevé
    # 3. Drawdown faible
    # 4. Nombre de trades pas ridicule
    ranked = sorted(
        all_results,
        key=lambda x: (
            x["profit_factor"],
            x["final_cash"],
            -x["max_drawdown"],
            x["total_trades"],
        ),
        reverse=True,
    )

    print("\n=== TOP 20 RESULTS ===")
    for i, row in enumerate(ranked[:20], start=1):
        print(f"\nRank #{i}")
        print(f"Test: {row['test_number']}")
        print(f"Strategy params: {row['strategy_params']}")
        print(f"Risk params: {row['risk_params']}")
        print(f"Final cash: {row['final_cash']:.2f}")
        print(f"PnL: {row['total_pnl']:.2f}")
        print(f"Profit factor: {row['profit_factor']:.2f}")
        print(f"Win rate: {row['win_rate']:.2f}%")
        print(f"Trades: {row['total_trades']}")
        print(f"Max drawdown: {row['max_drawdown']:.2f}%")
        print(f"Wins: {row['wins']}")
        print(f"Losses: {row['losses']}")

    profitable = [r for r in ranked if r["final_cash"] > Settings.INITIAL_CASH]
    pf_above_one = [r for r in ranked if r["profit_factor"] > 1.0]
    low_dd = [r for r in ranked if r["max_drawdown"] < 20.0]

    print("\n=== SUMMARY ===")
    print(f"Total tests run: {len(all_results)}")
    print(f"Configs with final cash > initial cash: {len(profitable)}")
    print(f"Configs with profit factor > 1.0: {len(pf_above_one)}")
    print(f"Configs with drawdown < 20%: {len(low_dd)}")

    return ranked


if __name__ == "__main__":
    run_grid_search()