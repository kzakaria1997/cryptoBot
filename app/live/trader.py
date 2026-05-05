from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.config.settings import Settings
from app.data.collector import KrakenCollector
from app.features.feature_engine import FeatureEngine
from app.live.kraken_broker import LiveKrakenBroker
from app.live.reporter import build_report, load_trades, save_trade, send_email
from app.live.state import TradeState
from app.strategy.momentum_strategy import MomentumStrategy
from app.strategy.pullback_strategy import PullbackStrategy

REPORT_INTERVAL_DAYS = 7
START_TIME_FILE = "app/live/start_time.txt"


class LiveTrader:
    """
    Boucle live multi-stratégie avec trailing stop et regime filter.
    - Momentum + Pullback sur chaque paire configurée
    - Trailing stop : suit le peak, sort quand price baisse de TRAIL_STOP_PCT
    - Regime filter : n'entre pas si return_30d < REGIME_RETURN_30D_MIN
    - Dynamic sizing : POSITION_FRACTION × cash disponible par trade
    """

    def __init__(self):
        mode_label = "PAPER" if Settings.PAPER_MODE else "LIVE"
        self._log(f"=== LiveTrader ({mode_label}) ===")
        sizing_label = f"smart(40-100%)" if Settings.SMART_SIZING else f"{Settings.POSITION_FRACTION*100:.0f}%"
        self._log(f"SL={Settings.STOP_LOSS_PCT*100:.1f}%  Trail={Settings.TRAIL_STOP_PCT*100:.0f}%  "
                  f"Sizing={sizing_label}  Regime>=return_30d{Settings.REGIME_RETURN_30D_MIN*100:.0f}%  "
                  f"Pairs={Settings.TRADING_PAIRS}")

        self.state     = TradeState(Settings.LIVE_STATE_FILE)
        self.broker    = LiveKrakenBroker(
            api_key    = Settings.KRAKEN_API_KEY,
            api_secret = Settings.KRAKEN_API_SECRET,
            fee_rate   = Settings.FEE_RATE,
            paper_mode = Settings.PAPER_MODE,
        )
        self.collector = KrakenCollector()

        self.strategies = [
            ("pullback",  PullbackStrategy(Settings.PULLBACK_PARAMS)),
            ("momentum",  MomentumStrategy(Settings.MOMENTUM_PARAMS)),
        ]

        self.predictor = None
        if Settings.ML_CONFIDENCE_THRESHOLD > 0.0 and os.path.exists(Settings.ML_MODEL_PATH):
            try:
                from app.ml.predictor import MLPredictor
                self.predictor = MLPredictor(model_path=Settings.ML_MODEL_PATH)
                self._log(f"ML charge : {Settings.ML_MODEL_PATH}  seuil={Settings.ML_CONFIDENCE_THRESHOLD}")
            except Exception as e:
                self._log(f"ML non disponible ({e}). Trading sans filtre ML.")
        else:
            self._log("ML desactive (threshold=0.0). Regime filter actif.")

        # Horodatage de démarrage pour le rapport hebdomadaire
        self._start_time = self._load_start_time()
        self._report_sent = False
        self._log(f"Rapport hebdo prévu le : {(self._start_time + timedelta(days=REPORT_INTERVAL_DAYS)).strftime('%Y-%m-%d %H:%M')} UTC")

    # ------------------------------------------------------------------ #
    # Boucle principale
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self._log("Boucle démarrée. Ctrl+C pour arrêter.")
        while True:
            try:
                wait = self._seconds_until_next_candle()
                self._log(f"Prochaine bougie dans {wait:.0f}s...")
                time.sleep(wait)
                for pair in Settings.TRADING_PAIRS:
                    self._tick(pair)
                self._check_weekly_report()
            except KeyboardInterrupt:
                self._log("Arrêt manuel.")
                break
            except Exception as e:
                self._log(f"ERREUR: {e}\n{traceback.format_exc()}")
                self._log("Nouvelle tentative dans 60s...")
                time.sleep(60)

    # ------------------------------------------------------------------ #
    # Tick par paire
    # ------------------------------------------------------------------ #

    def _tick(self, pair: str) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        self._log(f"\n--- {now}  {pair} ---")

        df = self._fetch_and_featurize(pair)
        if df is None or len(df) < 10:
            self._log("Données insuffisantes, skip.")
            return

        row = df.iloc[-1]
        current_price = float(row["close"])
        self._log(f"Prix: ${current_price:,.2f}")

        # ---- Sortie si en position sur cette paire ----
        if self.state.in_position and self.state.position_pair == pair:
            self._check_exit(row, current_price, pair)
            return

        # ---- Entrée (si pas déjà en position sur une autre paire) ----
        if not self.state.in_position:
            self._check_entry(row, current_price, pair)

    # ------------------------------------------------------------------ #
    # Sortie — trailing stop + SL fixe
    # ------------------------------------------------------------------ #

    def _check_exit(self, row: pd.Series, price: float, pair: str) -> None:
        entry   = self.state.entry_price
        pnl_pct = (price - entry) / entry

        # Met à jour le peak (trailing stop)
        self.state.update_peak(price)
        peak    = self.state.peak_price
        trail_stop = peak * (1.0 - Settings.TRAIL_STOP_PCT)

        self._log(f"Position {pair} | entry=${entry:,.2f}  peak=${peak:,.2f}"
                  f"  trail_stop=${trail_stop:,.2f}  pnl={pnl_pct*100:+.2f}%")

        reason = None
        if price <= entry * (1.0 - Settings.STOP_LOSS_PCT):
            reason = "STOP_LOSS"
        elif price <= trail_stop and peak > entry:
            reason = "TRAIL_STOP"

        if reason:
            self._sell(price, reason, pnl_pct, pair)

    def _sell(self, price: float, reason: str, pnl_pct: float, pair: str) -> None:
        qty     = self.state.position_qty
        order   = self.broker.sell_market(pair, qty, price)
        pnl_usd = qty * price * (1 - Settings.FEE_RATE) - qty * self.state.entry_price
        self._log(f"SELL [{reason}] {pair}  qty={qty:.6f} @ ${price:,.2f}"
                  f"  pnl=${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%)")
        save_trade({
            "pair":         pair,
            "strategy":     self.state.position_pair or "?",
            "entry_price":  self.state.entry_price,
            "exit_price":   price,
            "qty":          qty,
            "pnl_usd":      round(pnl_usd, 4),
            "pnl_pct":      round(pnl_pct * 100, 3),
            "exit_reason":  reason,
            "entry_time":   self.state.entry_time or "",
            "exit_time":    datetime.now(timezone.utc).isoformat(),
            "paper_mode":   Settings.PAPER_MODE,
        })
        self.state.close_position()

    # ------------------------------------------------------------------ #
    # Entrée — regime filter + ML + signals
    # ------------------------------------------------------------------ #

    def _check_entry(self, row: pd.Series, price: float, pair: str) -> None:
        # Regime filter : évite les bear markets
        r30 = row.get("return_30d", None)
        if r30 is not None and not pd.isna(r30):
            if float(r30) < Settings.REGIME_RETURN_30D_MIN:
                self._log(f"  Regime baissier (return_30d={r30*100:.1f}%), skip.")
                return

        # Log diagnostic de tendance (EMA50 vs EMA200)
        ema_t = row.get("ema_trend")
        ema_b = row.get("ema_base")
        rsi   = row.get("rsi")
        rvol  = row.get("rel_volume")
        if ema_t is not None and ema_b is not None and not pd.isna(ema_t) and not pd.isna(ema_b):
            gap_pct  = (float(ema_t) - float(ema_b)) / float(ema_b) * 100
            trend_ok = float(ema_t) > float(ema_b)
            extra    = f"  RSI={float(rsi):.1f}  RelVol={float(rvol):.2f}" if (rsi is not None and rvol is not None and not pd.isna(rsi) and not pd.isna(rvol)) else ""
            self._log(f"  Trend EMA50/EMA200: {'OK' if trend_ok else 'NON'} (gap={gap_pct:+.3f}%){extra}")

        # Cherche un signal dans l'ordre de priorité
        for name, strategy in self.strategies:
            if not strategy.validate_row(row):
                continue
            if not strategy.generate_entry_signal(row):
                continue

            ml_score = 0.0
            if self.predictor is not None:
                ml_score = self.predictor.predict_proba(row)
                self._log(f"  Signal {name} | ML={ml_score:.3f} (seuil={Settings.ML_CONFIDENCE_THRESHOLD})")
                if ml_score < Settings.ML_CONFIDENCE_THRESHOLD:
                    self._log(f"  -> Rejete ML.")
                    continue

            self._buy(price, pair, name, row=row)
            return   # un seul trade par tick

    @staticmethod
    def _smart_fraction(row: pd.Series) -> float:
        fraction = Settings.POSITION_FRACTION  # base 40%
        gap = row.get("trend_gap_pct")
        if gap is not None and not pd.isna(gap) and float(gap) >= 0.005:
            fraction += 0.20
        r30 = row.get("return_30d")
        if r30 is not None and not pd.isna(r30) and float(r30) >= 0.10:
            fraction += 0.20
        rvol = row.get("rel_volume")
        if rvol is not None and not pd.isna(rvol) and float(rvol) >= 1.5:
            fraction += 0.20
        return min(fraction, 1.0)

    def _buy(self, price: float, pair: str, strategy_name: str, row: pd.Series | None = None) -> None:
        cash = self.state.available_cash
        if Settings.SMART_SIZING and row is not None:
            fraction = self._smart_fraction(row)
        else:
            fraction = Settings.POSITION_FRACTION
        trade_size = cash * fraction
        if trade_size < 10.0:
            self._log(f"  Cash insuffisant (${cash:.2f}), skip.")
            return
        qty = trade_size / price
        self.broker.buy_market(pair, trade_size, price)
        self.state.open_position(price, qty, pair=pair, cash_used=trade_size)
        self._log(f"BUY [{strategy_name}] {pair}  qty={qty:.6f} @ ${price:,.2f}"
                  f"  size=${trade_size:.2f}  ({fraction*100:.0f}% du cash)")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #

    def _fetch_and_featurize(self, pair: str) -> pd.DataFrame | None:
        try:
            df = self.collector.get_ohlc(pair, Settings.TIMEFRAME_MINUTES)
            return FeatureEngine.add_features(
                df,
                fast_ema=Settings.FAST_EMA,
                slow_ema=Settings.SLOW_EMA,
                rsi_period=Settings.RSI_PERIOD,
            )
        except Exception as e:
            self._log(f"Erreur fetch {pair}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Timing
    # ------------------------------------------------------------------ #

    def _seconds_until_next_candle(self) -> float:
        now      = datetime.now(timezone.utc)
        interval = Settings.TIMEFRAME_MINUTES * 60
        elapsed  = (now.minute * 60 + now.second) % interval
        return float((interval - elapsed) + 30)

    # ------------------------------------------------------------------ #
    # Rapport hebdomadaire
    # ------------------------------------------------------------------ #

    def _load_start_time(self) -> datetime:
        os.makedirs(os.path.dirname(START_TIME_FILE), exist_ok=True)
        if os.path.exists(START_TIME_FILE):
            with open(START_TIME_FILE) as f:
                return datetime.fromisoformat(f.read().strip())
        now = datetime.now(timezone.utc)
        with open(START_TIME_FILE, "w") as f:
            f.write(now.isoformat())
        return now

    def _check_weekly_report(self) -> None:
        if self._report_sent:
            return
        elapsed = datetime.now(timezone.utc) - self._start_time
        if elapsed < timedelta(days=REPORT_INTERVAL_DAYS):
            return

        to_email = Settings.REPORT_EMAIL
        api_key  = Settings.RESEND_API_KEY
        if not to_email or not api_key:
            self._log("Rapport hebdo : REPORT_EMAIL ou RESEND_API_KEY manquant, skip.")
            self._report_sent = True
            return

        trades = load_trades()
        report = build_report(trades, REPORT_INTERVAL_DAYS)
        self._log(f"\n{'='*50}\n{report}\n{'='*50}")

        ok = send_email(report, to_email, api_key)
        if ok:
            self._log(f"Rapport hebdo envoyé à {to_email}")
        else:
            self._log(f"Echec envoi email — rapport affiché dans les logs ci-dessus.")
        self._report_sent = True

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        os.makedirs(os.path.dirname(Settings.LIVE_LOG_FILE), exist_ok=True)
        with open(Settings.LIVE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
