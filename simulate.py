"""
Simulation locale qui reproduit exactement la logique du live trader :
  - Trailing stop 25% depuis le peak
  - Stop loss fixe 1%
  - 40% du portefeuille par trade
  - Filtre regime (return_30d >= -3%)
  - Strategies : Momentum + Pullback (params de Settings)

Usage :
  python simulate.py                    # BTC, 2024-01-01 -> aujourd'hui
  python simulate.py --pair eth         # ETH
  python simulate.py --start 2022-01-01 # autre date de debut
  python simulate.py --trades           # affiche chaque trade
"""
from __future__ import annotations

import argparse

import pandas as pd

from app.config.settings import Settings
from app.data.csv_loader import CsvLoader
from app.features.feature_engine import FeatureEngine
from app.ml.predictor import MLPredictor
from app.strategy.short_strategy import ShortMomentumStrategy


def _load_csv(path: str) -> pd.DataFrame:
    """Charge BTC (7 col, unix ts) ou ETH (6 col, header, datetime ts)."""
    raw = pd.read_csv(path, header=None, nrows=1, low_memory=False)
    has_header = not str(raw.iloc[0, 0]).lstrip("-").replace(".", "").isdigit()

    if has_header:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"timestamp": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    else:
        loader = CsvLoader()
        df = loader.get_ohlc(path)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "high" not in df.columns:
        df["high"] = df["close"]
        df["low"]  = df["close"]
    return df
from app.strategy.momentum_strategy import MomentumStrategy
from app.strategy.pullback_strategy import PullbackStrategy


# ------------------------------------------------------------------ #
# Boucle de simulation
# ------------------------------------------------------------------ #

def _smart_fraction(row: pd.Series) -> float:
    """
    Taille de position dynamique selon la qualite du signal :
      - Base : 40%
      - +20% si tendance forte  (trend_gap_pct >= 0.5%)
      - +20% si bull market     (return_30d    >= 10%)
      - +20% si volume eleve    (rel_volume    >= 1.5x)
    Max : 100%
    """
    fraction = 0.40

    gap = row.get("trend_gap_pct")
    if gap is not None and pd.notna(gap) and float(gap) >= 0.005:
        fraction += 0.20

    r30 = row.get("return_30d")
    if r30 is not None and pd.notna(r30) and float(r30) >= 0.10:
        fraction += 0.20

    rvol = row.get("rel_volume")
    if rvol is not None and pd.notna(rvol) and float(rvol) >= 1.5:
        fraction += 0.20

    return min(fraction, 1.0)


def simulate(df: pd.DataFrame, pair_name: str, initial_cash: float = 1000.0,
             fixed_trade_size: float | None = None, smart: bool = False,
             trail_stop: float | None = None, take_profit: float | None = None,
             ml_threshold: float | None = None, ml_only: bool = False,
             enable_short: bool = False) -> dict:
    long_strategies  = [
        ("momentum", MomentumStrategy(Settings.MOMENTUM_PARAMS)),
        ("pullback",  PullbackStrategy(Settings.PULLBACK_PARAMS)),
    ]
    short_strategies = [("short_momentum", ShortMomentumStrategy())] if enable_short else []
    predictor = MLPredictor(Settings.ML_MODEL_PATH) if ml_threshold is not None else None

    cash            = initial_cash
    in_position     = False
    direction       = "long"   # "long" ou "short"
    entry_price     = 0.0
    peak_price      = 0.0      # pour long : peak haut / pour short : peak bas
    qty             = 0.0
    entry_time      = None
    strat_name      = ""
    entry_fraction  = 0.0
    realized_peak   = initial_cash   # pic de cash REALISE (post-sortie), pour circuit-breaker
    DD_LIMIT        = 0.20           # bloque les nouvelles entrees si cash DD > 20%
    trades: list[dict] = []
    equity: list[dict] = []

    for _, row in df.iterrows():
        price = float(row["close"])
        cur_equity = cash + (qty * price if in_position else 0.0)
        equity.append({"timestamp": row["timestamp"], "equity": cur_equity})

        # ---- Sortie ----
        if in_position:
            _trail_pct = trail_stop if trail_stop is not None else Settings.TRAIL_STOP_PCT

            if direction == "long":
                if price > peak_price:
                    peak_price = price
                pnl_pct = (price - entry_price) / entry_price
                _ts     = peak_price * (1.0 - _trail_pct)

                reason = None
                if price <= entry_price * (1.0 - Settings.STOP_LOSS_PCT):
                    reason = "STOP_LOSS"
                elif take_profit is not None and pnl_pct >= take_profit:
                    reason = "TAKE_PROFIT"
                elif price <= _ts and peak_price > entry_price:
                    reason = "TRAIL_STOP"

                if reason:
                    gross    = qty * price
                    received = gross * (1.0 - Settings.FEE_RATE)
                    pnl_usd  = received - qty * entry_price
                    cash    += received
                    if cash > realized_peak:
                        realized_peak = cash
                    trades.append(_trade(pair_name, strat_name, entry_time, row["timestamp"],
                                         entry_price, price, qty, pnl_usd, pnl_pct, reason,
                                         entry_fraction, "long"))
                    in_position = False

            else:  # SHORT
                if price < peak_price:
                    peak_price = price
                pnl_pct   = (entry_price - price) / entry_price
                _ts_short = peak_price * (1.0 + _trail_pct)  # trail au-dessus du creux

                reason = None
                if price >= entry_price * (1.0 + Settings.STOP_LOSS_PCT):
                    reason = "STOP_LOSS"
                elif take_profit is not None and pnl_pct >= take_profit:
                    reason = "TAKE_PROFIT"
                elif price >= _ts_short and peak_price < entry_price:
                    reason = "TRAIL_STOP"

                if reason:
                    pnl_usd = qty * (entry_price - price) * (1.0 - Settings.FEE_RATE)
                    cash   += qty * entry_price + pnl_usd   # retour capital + profit
                    if cash > realized_peak:
                        realized_peak = cash
                    trades.append(_trade(pair_name, strat_name, entry_time, row["timestamp"],
                                         entry_price, price, qty, pnl_usd, pnl_pct, reason,
                                         entry_fraction, "short"))
                    in_position = False

            continue   # pas d'entrée sur la même bougie

        # ---- Score ML (calculé une seule fois par bougie) ----
        ml_score = None
        if predictor is not None:
            try:
                ml_score = predictor.predict_proba(row)
            except Exception:
                ml_score = None

        # ---- Entrée LONG ----
        signal_name = None
        r30 = row.get("return_30d")
        long_regime_ok = not (pd.notna(r30) and float(r30) < Settings.REGIME_RETURN_30D_MIN)

        if long_regime_ok:
            if ml_only:
                if ml_score is not None and ml_score >= ml_threshold:
                    signal_name = f"ml({ml_score:.2f})"
            else:
                for name, strategy in long_strategies:
                    if not strategy.validate_row(row):
                        continue
                    if not strategy.generate_entry_signal(row):
                        continue
                    if ml_score is not None and ml_score < ml_threshold:
                        continue
                    signal_name = name
                    break

        # ---- Entrée SHORT (si pas de signal long) ----
        entry_dir = "long"
        if signal_name is None and enable_short:
            for name, strategy in short_strategies:
                if not strategy.validate_row(row):
                    continue
                if not strategy.generate_entry_signal(row):
                    continue
                signal_name = name
                entry_dir   = "short"
                break

        if signal_name is None:
            continue

        # Circuit-breaker : bloque si le cash realise est en DD > 20% depuis son pic
        if (realized_peak - cash) / realized_peak > DD_LIMIT:
            continue

        if fixed_trade_size is not None:
            trade_size = fixed_trade_size
        elif entry_dir == "short":
            # Shorts : taille fixe (pas de smart sizing) pour limiter le risque
            trade_size = cash * Settings.POSITION_FRACTION
        elif smart:
            base_frac = _smart_fraction(row)
            if ml_score is not None and ml_threshold is not None:
                ml_boost = max(0.0, (ml_score - ml_threshold) / (1.0 - ml_threshold)) * 0.20
                base_frac = min(base_frac + ml_boost, 1.0)
            trade_size = cash * base_frac
        else:
            trade_size = cash * Settings.POSITION_FRACTION

        if trade_size < 10.0 or trade_size > cash:
            continue

        entry_fraction = trade_size / cash
        cost           = trade_size * (1.0 + Settings.FEE_RATE)
        qty            = trade_size / price
        if entry_dir == "short":
            cash -= cost      # on immobilise le capital pour la marge
        else:
            cash -= min(cost, cash)
        in_position = True
        direction   = entry_dir
        entry_price = price
        peak_price  = price
        entry_time  = row["timestamp"]
        strat_name  = signal_name

    # Ferme position ouverte en fin de données
    if in_position:
        last  = df.iloc[-1]
        price = float(last["close"])
        if direction == "long":
            pnl_pct = (price - entry_price) / entry_price
            gross   = qty * price
            received = gross * (1.0 - Settings.FEE_RATE)
            pnl_usd = received - qty * entry_price
            cash   += received
        else:
            pnl_pct = (entry_price - price) / entry_price
            pnl_usd = qty * (entry_price - price) * (1.0 - Settings.FEE_RATE)
            cash   += qty * entry_price + pnl_usd
        trades.append(_trade(pair_name, strat_name, entry_time, last["timestamp"],
                             entry_price, price, qty, pnl_usd, pnl_pct, "END_OF_DATA",
                             entry_fraction, direction))

    return {"cash": cash, "trades": trades, "equity": equity, "initial_cash": initial_cash}


def _trade(pair, strategy, entry_time, exit_time,
           entry_price, exit_price, qty, pnl_usd, pnl_pct, reason,
           fraction: float = 0.0, direction: str = "long") -> dict:
    return {
        "pair":        pair,
        "strategy":    strategy,
        "entry_time":  entry_time,
        "exit_time":   exit_time,
        "entry_price": round(entry_price, 2),
        "exit_price":  round(exit_price, 2),
        "qty":         round(qty, 6),
        "pnl_usd":     round(pnl_usd, 2),
        "pnl_pct":     round(pnl_pct * 100, 3),
        "reason":      reason,
        "fraction":    round(fraction * 100, 0),
        "direction":   direction,
    }


# ------------------------------------------------------------------ #
# Rapport
# ------------------------------------------------------------------ #

def print_report(result: dict, show_trades: bool = False) -> None:
    trades = result["trades"]
    wins   = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]

    total_pnl   = sum(t["pnl_usd"] for t in trades)
    gross_win   = sum(t["pnl_usd"] for t in wins)
    gross_loss  = abs(sum(t["pnl_usd"] for t in losses))
    win_rate    = len(wins) / len(trades) * 100 if trades else 0
    pf          = gross_win / gross_loss if gross_loss else float("inf")
    avg_win     = gross_win / len(wins) if wins else 0
    avg_loss    = gross_loss / len(losses) if losses else 0
    total_ret   = (result["cash"] - result["initial_cash"]) / result["initial_cash"] * 100

    # Drawdown
    eq_vals = [e["equity"] for e in result["equity"]]
    max_dd  = 0.0
    peak_eq = eq_vals[0] if eq_vals else 0
    for v in eq_vals:
        if v > peak_eq:
            peak_eq = v
        dd = (peak_eq - v) / peak_eq * 100
        if dd > max_dd:
            max_dd = dd

    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1

    print("\n" + "=" * 55)
    print("  SIMULATION - logique identique au live trader")
    print("=" * 55)
    print(f"  Capital initial   : ${result['initial_cash']:,.2f}")
    print(f"  Capital final     : ${result['cash']:,.2f}")
    print(f"  Retour total      : {total_ret:+.2f}%")
    print(f"  PnL total         : ${total_pnl:+.2f}")
    print("-" * 55)
    print(f"  Trades            : {len(trades)}  (W:{len(wins)} / L:{len(losses)})")
    print(f"  Win rate          : {win_rate:.1f}%")
    print(f"  Profit factor     : {pf:.3f}")
    print(f"  Gain moyen        : ${avg_win:.2f}")
    print(f"  Perte moyenne     : ${avg_loss:.2f}")
    print(f"  Max drawdown      : {max_dd:.2f}%")
    print(f"  Sorties           : { '  '.join(f'{r}:{c}' for r,c in reasons.items()) }")
    print("=" * 55)

    if show_trades and trades:
        has_frac = any(t.get("fraction", 0) > 0 for t in trades)
        frac_hdr = "  Mise%" if has_frac else ""
        has_short = any(t.get("direction") == "short" for t in trades)
        dir_hdr   = "  Dir  " if has_short else ""
        print(f"\n{'#':>3}  {'Date entree':16}  {'Date sortie':16}  "
              f"{'Strat':16}  {'Entree':>10}  {'Sortie':>10}  "
              f"{'PnL $':>8}  {'PnL %':>7}  {'Raison'}{dir_hdr}{frac_hdr}")
        print("-" * (120 + (8 if has_frac else 0) + (7 if has_short else 0)))
        for i, t in enumerate(trades, 1):
            entry_s = t["entry_time"].strftime("%Y-%m-%d %H:%M") if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])[:16]
            exit_s  = t["exit_time"].strftime("%Y-%m-%d %H:%M")  if hasattr(t["exit_time"],  "strftime") else str(t["exit_time"])[:16]
            pnl_tag = "+" if t["pnl_usd"] >= 0 else ""
            frac_s  = f"  {int(t['fraction']):>3}%" if has_frac and t.get("fraction") else ""
            dir_s   = f"  {'SHORT' if t.get('direction') == 'short' else 'LONG ':5}" if has_short else ""
            print(f"{i:>3}  {entry_s:16}  {exit_s:16}  "
                  f"{t['strategy']:16}  ${t['entry_price']:>9,.2f}  ${t['exit_price']:>9,.2f}  "
                  f"{pnl_tag}${t['pnl_usd']:>7.2f}  {t['pnl_pct']:>+6.2f}%  {t['reason']}{dir_s}{frac_s}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def _run_pair(pair: str, start: str, end: str | None, cash: float, show_trades: bool,
              label: str = "", fixed_size: float | None = None, smart: bool = False,
              trail_stop: float | None = None, take_profit: float | None = None,
              ml_threshold: float | None = None, ml_only: bool = False,
              enable_short: bool = False) -> dict:
    csv_map   = {"btc": "app/data/XBTUSD_15.csv", "eth": "app/data/ETHUSD_15.csv"}
    pair_name = "XXBTZUSD" if pair == "btc" else "XETHZUSD"

    print(f"Chargement {csv_map[pair]}...")
    df = _load_csv(csv_map[pair])

    start_dt = pd.Timestamp(start)
    df = df[df["timestamp"] >= start_dt]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]

    print(f"Features sur {len(df):,} bougies ({df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()})...")
    df = FeatureEngine.add_features(df, fast_ema=Settings.FAST_EMA,
                                        slow_ema=Settings.SLOW_EMA,
                                        rsi_period=Settings.RSI_PERIOD)
    df = df.reset_index(drop=True)

    print(f"Simulation {pair.upper()} en cours...")
    result = simulate(df, pair_name, initial_cash=cash, fixed_trade_size=fixed_size,
                      smart=smart, trail_stop=trail_stop, take_profit=take_profit,
                      ml_threshold=ml_threshold, ml_only=ml_only,
                      enable_short=enable_short)
    if label:
        print(f"\n{'='*55}\n  {label}\n{'='*55}")
    print_report(result, show_trades=show_trades)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair",   default="btc", choices=["btc", "eth", "both"])
    parser.add_argument("--start",  default="2024-01-01")
    parser.add_argument("--end",    default=None)
    parser.add_argument("--trades",     action="store_true", help="Afficher chaque trade")
    parser.add_argument("--cash",       type=float, default=1000.0)
    parser.add_argument("--fixed-size", type=float, default=None,
                        help="Taille de trade fixe en $ (ex: 500). Par defaut: fraction du portefeuille")
    parser.add_argument("--fraction",   type=float, default=None,
                        help="Fraction du portefeuille par trade (ex: 1.0 = 100%%, 0.4 = 40%%)")
    parser.add_argument("--smart",       action="store_true",
                        help="Taille dynamique selon qualite du signal (40%% a 100%%)")
    parser.add_argument("--trail-stop",    type=float, default=None,
                        help="Trailing stop en fraction (ex: 0.08 = 8%%). Defaut: 0.25")
    parser.add_argument("--take-profit",   type=float, default=None,
                        help="Take profit fixe en fraction (ex: 0.045 = 4.5%%).")
    parser.add_argument("--ml-threshold",  type=float, default=None,
                        help="Activer filtre ML avec ce seuil (ex: 0.60). Desactive = pas de ML.")
    parser.add_argument("--ml-only",       action="store_true",
                        help="ML comme signal principal (ignore les strategies techniques).")
    parser.add_argument("--short",          action="store_true",
                        help="Activer les positions short (ShortMomentumStrategy).")
    args = parser.parse_args()

    fixed = args.fixed_size
    if args.fraction is not None:
        Settings.POSITION_FRACTION = args.fraction

    ml_thr = args.ml_threshold

    if args.pair == "both":
        _run_pair("btc", args.start, args.end, args.cash, args.trades, "BTC",
                  fixed, args.smart, args.trail_stop, args.take_profit, ml_thr, args.ml_only,
                  args.short)
        _run_pair("eth", args.start, args.end, args.cash, args.trades, "ETH",
                  fixed, args.smart, args.trail_stop, args.take_profit, ml_thr, args.ml_only,
                  args.short)
    else:
        _run_pair(args.pair, args.start, args.end, args.cash, args.trades,
                  fixed_size=fixed, smart=args.smart,
                  trail_stop=args.trail_stop, take_profit=args.take_profit,
                  ml_threshold=ml_thr, ml_only=args.ml_only,
                  enable_short=args.short)


if __name__ == "__main__":
    main()
