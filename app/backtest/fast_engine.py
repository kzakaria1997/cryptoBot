"""
Moteur de backtest vectorisé pour le grid search.
Utilise numpy au lieu de iterrows — 50-100x plus rapide.

Au lieu de boucler sur chaque bougie, on boucle seulement
sur les trades (beaucoup moins d'itérations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Génération de signaux vectorisée
# ------------------------------------------------------------------ #

def momentum_signals(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
    trend_gap = (df["ema_trend"] - df["ema_base"]) / df["ema_base"]

    entry = (
        (df["ema_trend"] > df["ema_base"]) &
        (df["ema_fast"]  > df["ema_slow"])  &
        (df["close"]     > df["ema_slow"])  &
        (trend_gap       >= params.get("trend_gap_min", 0.0)) &
        (df["rsi"]       >= params.get("rsi_min", 54)) &
        (df["rsi"]       <= params.get("rsi_max", 58)) &
        (df["rel_volume"]>= params.get("rel_volume_min", 1.35)) &
        (df["return_3"]  >= params.get("return_3_min", 0.003)) &
        (df["volatility_10"] >= params.get("volatility_10_min", 0.0015))
    ).fillna(False).values

    exit_ = (
        (df["close"] < df["ema_fast"]) & (df["rsi"] < 45)
    ).fillna(False).values

    return entry, exit_


def pullback_signals(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
    trend_gap      = (df["ema_trend"] - df["ema_base"]) / df["ema_base"]
    pb_fast        = (df["close"] - df["ema_fast"]) / df["ema_fast"]
    pb_slow        = (df["close"] - df["ema_slow"]) / df["ema_slow"]

    entry = (
        (df["ema_trend"] > df["ema_base"]) &
        (df["ema_fast"]  > df["ema_slow"])  &
        (df["close"]     > df["ema_base"])  &
        (trend_gap       >= params.get("trend_gap_min", 0.001)) &
        (pb_fast         >= params.get("pullback_fast_min", -0.0035)) &
        (pb_fast         <= params.get("pullback_fast_max",  0.0015)) &
        (pb_slow         >= params.get("pullback_slow_min", -0.0025)) &
        (df["return_1"]  > 0) &
        (df["rsi"]       >= params.get("rsi_min", 48)) &
        (df["rsi"]       <= params.get("rsi_max", 58)) &
        (df["rel_volume"]>= params.get("rel_volume_min", 1.05)) &
        (df["volatility_10"] >= params.get("volatility_10_min", 0.0012))
    ).fillna(False).values

    exit_ = (
        (df["close"] < df["ema_slow"]) |
        (df["rsi"]   < params.get("exit_rsi_threshold", 43))
    ).fillna(False).values

    return entry, exit_


# ------------------------------------------------------------------ #
# Backtest vectorisé (boucle sur trades, pas sur bougies)
# ------------------------------------------------------------------ #

def fast_backtest(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    entry_sig: np.ndarray,
    exit_sig: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    fee_rate: float,
    trade_size_usd: float,
    initial_cash: float,
    use_exit_signal: bool = False,
) -> dict:
    n = len(closes)
    entry_indices = np.flatnonzero(entry_sig)

    if len(entry_indices) == 0:
        return _empty_result(initial_cash)

    cash = initial_cash
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    last_exit = -1

    for ei in entry_indices:
        if ei <= last_exit:
            continue  # déjà en position, on skip

        entry_price = closes[ei]
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)
        qty = trade_size_usd / entry_price

        # Chercher la sortie avec numpy (très rapide)
        future_slice = slice(ei + 1, n)
        fut_highs = highs[future_slice]
        fut_lows  = lows[future_slice]

        tp_hits = np.flatnonzero(fut_highs >= tp_price)
        sl_hits = np.flatnonzero(fut_lows  <= sl_price)

        tp_offset = tp_hits[0] if len(tp_hits) else n
        sl_offset = sl_hits[0] if len(sl_hits) else n

        if use_exit_signal:
            ex_hits   = np.flatnonzero(exit_sig[future_slice])
            ex_offset = ex_hits[0] if len(ex_hits) else n
        else:
            ex_offset = n

        first_offset = min(tp_offset, sl_offset, ex_offset)

        if first_offset == n:
            exit_price = closes[-1]
            last_exit  = n - 1
        elif first_offset == tp_offset:
            exit_price = tp_price
            last_exit  = ei + 1 + tp_offset
        elif first_offset == sl_offset:
            exit_price = sl_price
            last_exit  = ei + 1 + sl_offset
        else:
            exit_price = closes[ei + 1 + ex_offset]
            last_exit  = ei + 1 + ex_offset

        proceeds   = qty * exit_price * (1.0 - fee_rate)
        cost_basis = trade_size_usd * (1.0 + fee_rate)
        pnl        = proceeds - cost_basis
        cash      += proceeds - cost_basis + trade_size_usd  # net change

        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)

    total_trades = wins + losses
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    win_rate = wins / total_trades * 100.0 if total_trades > 0 else 0.0

    return {
        "final_cash":    initial_cash + gross_profit - gross_loss,
        "total_trades":  total_trades,
        "wins":          wins,
        "losses":        losses,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "total_pnl":     gross_profit - gross_loss,
        "max_drawdown":  0.0,
    }


def _empty_result(initial_cash: float) -> dict:
    return {
        "final_cash": initial_cash, "total_trades": 0,
        "wins": 0, "losses": 0, "win_rate": 0.0,
        "profit_factor": 0.0, "total_pnl": 0.0, "max_drawdown": 0.0,
    }


# ------------------------------------------------------------------ #
# Point d'entrée unifié
# ------------------------------------------------------------------ #

def run_fast(
    df: pd.DataFrame,
    signal_fn,
    params: dict,
    sl_pct: float,
    tp_pct: float,
    fee_rate: float,
    trade_size_usd: float,
    initial_cash: float,
    use_exit_signal: bool = False,
) -> dict | None:
    required = ["close", "high", "low", "ema_fast", "ema_slow",
                "ema_trend", "ema_base", "rsi", "rel_volume"]
    df_clean = df.dropna(subset=required)
    if len(df_clean) < 50:
        return None

    entry_sig, exit_sig = signal_fn(df_clean, params)
    return fast_backtest(
        df_clean["close"].values, df_clean["high"].values, df_clean["low"].values,
        entry_sig, exit_sig, sl_pct, tp_pct, fee_rate, trade_size_usd, initial_cash,
        use_exit_signal,
    )


# ------------------------------------------------------------------ #
# Backtest combiné — 2 stratégies + dynamic sizing + trailing stop
# ------------------------------------------------------------------ #

def combined_backtest(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    entry_mom: np.ndarray,
    entry_pb: np.ndarray,
    sl_pct: float,
    trail_pct: float,        # trailing stop depuis le peak (ex: 0.02 = 2%)
    fee_rate: float,
    position_fraction: float, # fraction du cash par trade (ex: 0.30)
    initial_cash: float,
) -> dict:
    """
    Backtest combiné momentum+pullback avec trailing stop et dynamic sizing.
    - Priorité au signal pullback sur momentum si les deux se déclenchent au même bar.
    - Trailing stop vectorisé : suit le peak, sort quand price baisse de trail_pct.
    - Dynamic sizing : chaque trade utilise position_fraction du cash courant.
    """
    n = len(closes)
    all_entries = np.flatnonzero(entry_mom | entry_pb)

    if len(all_entries) == 0:
        return _empty_result(initial_cash)

    cash      = initial_cash
    peak_eq   = initial_cash
    max_dd    = 0.0
    wins = losses = 0
    gross_profit = gross_loss = 0.0
    last_exit = -1
    trades_log = []  # (entry_bar, exit_bar, pnl)

    for ei in all_entries:
        if ei <= last_exit:
            continue

        trade_size = cash * position_fraction
        if trade_size < 5.0:
            continue

        entry_price = closes[ei]
        sl_price    = entry_price * (1.0 - sl_pct)
        qty         = trade_size / entry_price

        future_slice = slice(ei + 1, n)
        fut_highs = highs[future_slice]
        fut_lows  = lows[future_slice]

        if len(fut_highs) == 0:
            break

        # --- SL fixe ---
        sl_hits   = np.flatnonzero(fut_lows <= sl_price)
        sl_offset = int(sl_hits[0]) if len(sl_hits) else n

        # --- Trailing stop vectorisé ---
        cum_peak     = np.maximum.accumulate(fut_highs)
        trail_prices = cum_peak * (1.0 - trail_pct)
        trail_hits   = np.flatnonzero(fut_lows <= trail_prices)
        trail_offset = int(trail_hits[0]) if len(trail_hits) else n

        first_offset = min(sl_offset, trail_offset)

        if first_offset == n:
            exit_price = closes[-1]
            last_exit  = n - 1
        elif first_offset == sl_offset:
            exit_price = sl_price
            last_exit  = ei + 1 + sl_offset
        else:
            # sortie au trailing stop = peak connu à ce bar - trail_pct
            exit_price = float(trail_prices[trail_offset])
            last_exit  = ei + 1 + trail_offset

        proceeds   = qty * exit_price * (1.0 - fee_rate)
        cost_basis = trade_size * (1.0 + fee_rate)
        pnl        = proceeds - cost_basis
        cash      += pnl

        # drawdown sur equity
        if cash > peak_eq:
            peak_eq = cash
        dd = (peak_eq - cash) / peak_eq * 100.0
        if dd > max_dd:
            max_dd = dd

        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)

        trades_log.append((ei, last_exit, pnl))

    total_trades  = wins + losses
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    win_rate      = wins / total_trades * 100.0 if total_trades > 0 else 0.0

    return {
        "final_cash":    cash,
        "total_trades":  total_trades,
        "wins":          wins,
        "losses":        losses,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "total_pnl":     cash - initial_cash,
        "max_drawdown":  max_dd,
        "trades_log":    trades_log,
    }


def run_combined(
    df: pd.DataFrame,
    mom_params: dict,
    pb_params:  dict,
    sl_pct: float,
    trail_pct: float,
    fee_rate: float,
    position_fraction: float,
    initial_cash: float,
    regime_return_30d_min: float = -0.03,   # filtre marché baissier
) -> dict | None:
    required = ["close", "high", "low", "ema_fast", "ema_slow",
                "ema_trend", "ema_base", "rsi", "rel_volume"]
    df_clean = df.dropna(subset=required)
    if len(df_clean) < 50:
        return None

    entry_mom, _ = momentum_signals(df_clean, mom_params)
    entry_pb,  _ = pullback_signals(df_clean, pb_params)

    # Regime filter : ne pas trader en tendance baissière
    if "regime_bull" in df_clean.columns and "return_30d" in df_clean.columns:
        bull  = df_clean["regime_bull"].fillna(0).values.astype(bool)
        r30   = df_clean["return_30d"].fillna(-1).values
        regime = bull & (r30 > regime_return_30d_min)
        entry_mom = entry_mom & regime
        entry_pb  = entry_pb  & regime

    return combined_backtest(
        df_clean["close"].values,
        df_clean["high"].values,
        df_clean["low"].values,
        entry_mom, entry_pb,
        sl_pct, trail_pct, fee_rate,
        position_fraction, initial_cash,
    )
