"""
Optimise les parametres sur 2020-2023, valide les meilleurs configs sur 2023-2025.
Teste Momentum + Pullback, avec et sans filtre ML.

Usage : python optimize_recent.py
"""
from __future__ import annotations

import os
import sys
from itertools import product

import pandas as pd

from app.backtest.engine import BacktestEngine
from app.backtest.fast_engine import (
    momentum_signals, pullback_signals, run_fast,
)
from app.backtest.report import BacktestReport
from app.config.settings import Settings
from app.data.market_data_service import MarketDataService
from app.features.feature_engine import FeatureEngine
from app.ml.predictor import MLPredictor
from app.strategy.momentum_strategy import MomentumStrategy
from app.strategy.pullback_strategy import PullbackStrategy

_SIGNAL_FN = {
    MomentumStrategy: momentum_signals,
    PullbackStrategy: pullback_signals,
}

OPTIM_START = pd.Timestamp("2020-01-01")
OPTIM_END   = pd.Timestamp("2023-01-01")
TEST_START  = pd.Timestamp("2023-01-01")
TOP_N       = 5          # meilleurs configs a valider en OOS


# ------------------------------------------------------------------ #
# Grilles de parametres
# ------------------------------------------------------------------ #
MOMENTUM_GRID = {
    "rsi_min":          [50, 54, 58],
    "rsi_max":          [58, 62],
    "rel_volume_min":   [1.1, 1.3],
    "return_3_min":     [0.001, 0.002],
    "trend_gap_min":    [0.0, 0.001],
}

PULLBACK_GRID = {
    "rsi_min":           [44, 50],
    "rsi_max":           [55, 60],
    "rel_volume_min":    [1.0, 1.1],
    "trend_gap_min":     [0.001, 0.002],
    "use_exit_signal":   [True, False],
}

RISK_GRID = {
    "stop_loss_pct":   [0.006, 0.010, 0.015],
    "take_profit_pct": [0.018, 0.030, 0.045],
}


def _combo(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, v)) for v in product(*grid.values())]


def _score(r: dict) -> tuple:
    return (
        r["profit_factor"],
        r["total_pnl"],
        -r["max_drawdown"],
        r["total_trades"],
    )


def run_grid(df: pd.DataFrame, strategy_class, param_grid: dict,
             risk_grid: dict, label: str) -> list[dict]:
    signal_fn    = _SIGNAL_FN[strategy_class]
    strat_combos = _combo(param_grid)
    risk_combos  = _combo(risk_grid)
    total = len(strat_combos) * len(risk_combos)
    print(f"\n{label} — {total} combinaisons...")

    results = []
    for i, sp in enumerate(strat_combos):
        use_exit = bool(sp.get("use_exit_signal", False))
        params   = {k: v for k, v in sp.items() if k != "use_exit_signal"}

        for rp in risk_combos:
            result = run_fast(
                df,
                signal_fn,
                params,
                sl_pct=rp["stop_loss_pct"],
                tp_pct=rp["take_profit_pct"],
                fee_rate=Settings.FEE_RATE,
                trade_size_usd=Settings.TRADE_SIZE_USD,
                initial_cash=Settings.INITIAL_CASH,
                use_exit_signal=use_exit,
            )
            if result is None:
                continue
            if result["total_trades"] >= 15:
                results.append({
                    "strategy_params": {**params, "use_exit_signal": use_exit},
                    "risk_params":      rp,
                    "use_exit_signal":  use_exit,
                    "final_cash":       result["final_cash"],
                    "total_pnl":        result["total_pnl"],
                    "profit_factor":    result["profit_factor"],
                    "win_rate":         result["win_rate"],
                    "total_trades":     result["total_trades"],
                    "max_drawdown":     result["max_drawdown"],
                })

        if (i + 1) % 10 == 0:
            print(f"  {(i+1)*len(risk_combos)}/{total}...")

    results.sort(key=_score, reverse=True)
    return results


def _print_top(results: list[dict], n: int, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'#':>3}  {'PF':>6}  {'WR':>7}  {'Cash':>8}  {'DD':>7}  {'Trades':>7}  Exit")
    print(f"{'-'*70}")
    for i, r in enumerate(results[:n], 1):
        exit_flag = "SIGNAL" if r["use_exit_signal"] else "SL/TP"
        print(
            f"{i:>3}  {r['profit_factor']:>6.3f}  {r['win_rate']:>6.1f}%"
            f"  {r['final_cash']:>8.2f}  {r['max_drawdown']:>6.2f}%"
            f"  {r['total_trades']:>7}  {exit_flag}"
        )
        print(f"     strat={r['strategy_params']}  risk={r['risk_params']}")


def validate_on_test(df_test: pd.DataFrame, configs: list[dict],
                     strategy_class, label: str,
                     predictor: MLPredictor | None = None) -> None:
    print(f"\n{'='*70}")
    print(f"  VALIDATION TEST 2023-2025 — {label}")
    print(f"{'='*70}")

    for i, cfg in enumerate(configs[:TOP_N], 1):
        sp = cfg["strategy_params"].copy()
        use_exit = cfg["use_exit_signal"]
        rp = cfg["risk_params"]

        if strategy_class is PullbackStrategy:
            sp.pop("use_exit_signal", None)

        strategy = strategy_class(sp)

        # sans ML
        r_base = BacktestEngine(
            strategy=strategy_class(sp),
            fee_rate=Settings.FEE_RATE,
            slippage_rate=Settings.SLIPPAGE_RATE,
            initial_cash=Settings.INITIAL_CASH,
            trade_size_usd=Settings.TRADE_SIZE_USD,
            stop_loss_pct=rp["stop_loss_pct"],
            take_profit_pct=rp["take_profit_pct"],
            use_exit_signal=use_exit,
        ).run(df_test)

        sb = r_base["summary"]
        print(
            f"\n  Config #{i} | SL={rp['stop_loss_pct']*100:.1f}%  TP={rp['take_profit_pct']*100:.1f}%"
            f"  exit={'SIGNAL' if use_exit else 'SL/TP'}"
        )
        print(f"    Sans ML  : PF={sb['profit_factor']:.3f}  WR={sb['win_rate']:.1f}%"
              f"  Cash={r_base['final_cash']:.2f}  DD={r_base['max_drawdown']:.2f}%"
              f"  T={sb['total_trades']}")

        if predictor is not None:
            r_ml = BacktestEngine(
                strategy=strategy_class(sp),
                fee_rate=Settings.FEE_RATE,
                slippage_rate=Settings.SLIPPAGE_RATE,
                initial_cash=Settings.INITIAL_CASH,
                trade_size_usd=Settings.TRADE_SIZE_USD,
                stop_loss_pct=rp["stop_loss_pct"],
                take_profit_pct=rp["take_profit_pct"],
                use_exit_signal=use_exit,
                predictor=predictor,
                ml_threshold=0.52,
            ).run(df_test)
            sm = r_ml["summary"]
            print(f"    Avec ML  : PF={sm['profit_factor']:.3f}  WR={sm['win_rate']:.1f}%"
                  f"  Cash={r_ml['final_cash']:.2f}  DD={r_ml['max_drawdown']:.2f}%"
                  f"  T={sm['total_trades']}")


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Charger et featurizer
    # ------------------------------------------------------------------ #
    print("=== Chargement et features ===")
    ds = MarketDataService(
        use_csv=Settings.USE_CSV,
        csv_filepath=Settings.CSV_FILEPATH,
        trading_pair=Settings.TRADING_PAIR,
        timeframe_minutes=Settings.TIMEFRAME_MINUTES,
    )
    df = ds.load_ohlc()
    df = FeatureEngine.add_features(df, fast_ema=Settings.FAST_EMA,
                                    slow_ema=Settings.SLOW_EMA,
                                    rsi_period=Settings.RSI_PERIOD)

    df_optim = df[(df["timestamp"] >= OPTIM_START) & (df["timestamp"] < OPTIM_END)].copy()
    df_test  = df[df["timestamp"] >= TEST_START].copy()

    print(f"Optim : {len(df_optim):,} bougies ({OPTIM_START.date()} a {OPTIM_END.date()})")
    print(f"Test  : {len(df_test):,} bougies ({TEST_START.date()} a {df_test['timestamp'].max().date()})")

    # ------------------------------------------------------------------ #
    # 2. Grid search sur periode 2020-2023
    # ------------------------------------------------------------------ #
    mom_results = run_grid(df_optim, MomentumStrategy, MOMENTUM_GRID, RISK_GRID, "MOMENTUM")
    pbl_results = run_grid(df_optim, PullbackStrategy, PULLBACK_GRID, RISK_GRID, "PULLBACK")

    _print_top(mom_results, TOP_N, "TOP MOMENTUM (2020-2023)")
    _print_top(pbl_results, TOP_N, "TOP PULLBACK  (2020-2023)")

    # ------------------------------------------------------------------ #
    # 3. Validation OOS sur 2023-2025
    # ------------------------------------------------------------------ #
    predictor = None
    if os.path.exists(Settings.ML_MODEL_PATH):
        try:
            predictor = MLPredictor(model_path="app/ml/model_oos.pkl")
            print("\nModele ML charge : app/ml/model_oos.pkl")
        except Exception:
            pass

    validate_on_test(df_test, mom_results, MomentumStrategy, "MOMENTUM", predictor)
    validate_on_test(df_test, pbl_results, PullbackStrategy, "PULLBACK",  predictor)

    print("\n=== Termine ===")


if __name__ == "__main__":
    main()
