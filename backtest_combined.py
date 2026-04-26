"""
Backtest combiné : Momentum + Pullback, trailing stop, dynamic sizing.
Valide sur 2020-2023 (optim) et 2023-2025 (OOS).

Usage : python backtest_combined.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from app.backtest.fast_engine import run_combined
from app.config.settings import Settings
from app.data.market_data_service import MarketDataService
from app.features.feature_engine import FeatureEngine

# ------------------------------------------------------------------ #
# Meilleurs paramètres (issus de optimize_recent.py)
# ------------------------------------------------------------------ #
MOM_PARAMS = {
    "rsi_min":        58,
    "rsi_max":        62,
    "rel_volume_min": 1.3,
    "return_3_min":   0.001,
    "trend_gap_min":  0.001,
}

PB_PARAMS = {
    "rsi_min":        50,
    "rsi_max":        55,
    "rel_volume_min": 1.1,
    "trend_gap_min":  0.001,
}

SL_PCT            = 0.010   # stop loss fixe 1%
TRAIL_PCT         = 0.250   # trailing stop 25% depuis le peak
POSITION_FRACTION = 0.40    # 40% du capital par trade
INITIAL_CASH      = 1000.0


def _print_result(label: str, result: dict, n_years: float) -> None:
    r = result
    annual = ((r["final_cash"] / INITIAL_CASH) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Cash        : ${r['final_cash']:.2f}  (départ ${INITIAL_CASH:.0f})")
    print(f"  PnL total   : ${r['total_pnl']:+.2f}  ({r['total_pnl']/INITIAL_CASH*100:+.1f}%)")
    print(f"  Rendement/an: {annual:+.1f}%")
    print(f"  Trades      : {r['total_trades']}  (W:{r['wins']} / L:{r['losses']})")
    print(f"  Win rate    : {r['win_rate']:.1f}%")
    print(f"  Profit fact : {r['profit_factor']:.3f}")
    print(f"  Max DD      : {r['max_drawdown']:.1f}%")
    trades_per_year = r["total_trades"] / n_years if n_years > 0 else 0
    print(f"  Trades/an   : {trades_per_year:.0f}")


def _run_period(df_full: pd.DataFrame, start: str, end: str | None, label: str) -> None:
    tz = df_full["timestamp"].dt.tz
    ts_start = pd.Timestamp(start, tz=tz) if tz else pd.Timestamp(start)
    ts_end   = pd.Timestamp(end,   tz=tz) if (end and tz) else (pd.Timestamp(end) if end else None)
    mask = df_full["timestamp"] >= ts_start
    if ts_end is not None:
        mask &= df_full["timestamp"] < ts_end
    df = df_full[mask].copy()
    n_years = (df["timestamp"].max() - df["timestamp"].min()).days / 365.25

    result = run_combined(
        df,
        MOM_PARAMS, PB_PARAMS,
        sl_pct=SL_PCT,
        trail_pct=TRAIL_PCT,
        fee_rate=Settings.FEE_RATE,
        position_fraction=POSITION_FRACTION,
        initial_cash=INITIAL_CASH,
    )
    if result is None:
        print(f"{label}: données insuffisantes")
        return

    _print_result(label, result, n_years)

    # Détail par année
    if result["trades_log"] and n_years > 1:
        timestamps = df["timestamp"].values
        idx_map    = {i: ts for i, ts in enumerate(timestamps)}
        yearly: dict[int, list] = {}
        for (ei, ex_i, pnl) in result["trades_log"]:
            if ei < len(timestamps):
                yr = pd.Timestamp(timestamps[ei]).year
                yearly.setdefault(yr, []).append(pnl)

        print(f"\n  Détail par année :")
        print(f"  {'Année':>6}  {'Trades':>7}  {'PnL $':>8}  {'PnL %':>7}")
        running = INITIAL_CASH
        for yr in sorted(yearly):
            pnls  = yearly[yr]
            total = sum(pnls)
            pct   = total / running * 100
            running += total
            print(f"  {yr:>6}  {len(pnls):>7}  {total:>+8.2f}  {pct:>+6.1f}%")


def main() -> None:
    print("=== Chargement données ===")
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
    print(f"  {len(df):,} bougies  ({df['timestamp'].min().date()} -> {df['timestamp'].max().date()})")

    print(f"\n  Config : SL={SL_PCT*100:.1f}%  Trail={TRAIL_PCT*100:.1f}%  "
          f"Sizing={POSITION_FRACTION*100:.0f}%  Strategies=MOM+PB")

    # --- Période d'entraînement ---
    _run_period(df, "2020-01-01", "2023-01-01", "OPTIM 2020-2023 (in-sample)")

    # --- Période OOS ---
    _run_period(df, "2023-01-01", None, "OOS 2023-2025 (out-of-sample)")

    # --- Historique complet ---
    _run_period(df, "2017-01-01", None, "HISTORIQUE 2017-2025")

    # --- ETH si disponible ---
    eth_csv = "app/data/ETHUSD_15.csv"
    if os.path.exists(eth_csv):
        print(f"\n\n{'#'*60}")
        print(f"  ETH/USD")
        print(f"{'#'*60}")
        df_eth = pd.read_csv(eth_csv, parse_dates=["timestamp"])
        if df_eth["timestamp"].dt.tz is None:
            df_eth["timestamp"] = df_eth["timestamp"].dt.tz_localize("UTC")
        df_eth = FeatureEngine.add_features(df_eth, fast_ema=Settings.FAST_EMA,
                                             slow_ema=Settings.SLOW_EMA,
                                             rsi_period=Settings.RSI_PERIOD)
        print(f"  ETH: {len(df_eth):,} bougies  ({df_eth['timestamp'].min().date()} -> {df_eth['timestamp'].max().date()})")
        _run_period(df_eth, "2020-01-01", "2023-01-01", "ETH OPTIM 2020-2023")
        _run_period(df_eth, "2023-01-01", None,          "ETH OOS 2023-2025")
        _run_period(df_eth, "2017-01-01", None,           "ETH HISTORIQUE 2017-2025")
    else:
        print(f"\n  (ETH non disponible — lance: python -m app.data.binance_downloader)")

    print("\n=== Terminé ===")


if __name__ == "__main__":
    main()
