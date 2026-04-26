from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

from app.config.settings import Settings
from app.data.collector import KrakenCollector
from app.features.feature_engine import FeatureEngine
from app.live.kraken_broker import LiveKrakenBroker
from app.live.state import TradeState
from app.ml.predictor import MLPredictor
from app.strategy.momentum_strategy import MomentumStrategy
from app.strategy.pullback_strategy import PullbackStrategy


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
        self._log(f"SL={Settings.STOP_LOSS_PCT*100:.1f}%  Trail={Settings.TRAIL_STOP_PCT*100:.0f}%  "
                  f"Sizing={Settings.POSITION_FRACTION*100:.0f}%  "
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

        self.predictor: MLPredictor | None = None
        if os.path.exists(Settings.ML_MODEL_PATH):
            self.predictor = MLPredictor(model_path=Settings.ML_MODEL_PATH)
            self._log(f"ML chargé : {Settings.ML_MODEL_PATH}  seuil={Settings.ML_CONFIDENCE_THRESHOLD}")
        else:
            self._log(f"AVERT: ML introuvable ({Settings.ML_MODEL_PATH}). Trading sans filtre ML.")

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
        qty   = self.state.position_qty
        order = self.broker.sell_market(pair, qty, price)
        pnl_usd = qty * price * (1 - Settings.FEE_RATE) - qty * self.state.entry_price
        self._log(f"SELL [{reason}] {pair}  qty={qty:.6f} @ ${price:,.2f}"
                  f"  pnl=${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%)")
        self.state.close_position()

    # ------------------------------------------------------------------ #
    # Entrée — regime filter + ML + signals
    # ------------------------------------------------------------------ #

    def _check_entry(self, row: pd.Series, price: float, pair: str) -> None:
        # Regime filter : évite les bear markets
        r30 = row.get("return_30d", None)
        if r30 is not None and not pd.isna(r30):
            if float(r30) < Settings.REGIME_RETURN_30D_MIN:
                self._log(f"  Régime baissier (return_30d={r30*100:.1f}%), skip.")
                return

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
                    self._log(f"  -> Rejeté ML.")
                    continue

            self._buy(price, pair, name)
            return   # un seul trade par tick

    def _buy(self, price: float, pair: str, strategy_name: str) -> None:
        cash       = self.state.available_cash
        trade_size = cash * Settings.POSITION_FRACTION
        if trade_size < 10.0:
            self._log(f"  Cash insuffisant (${cash:.2f}), skip.")
            return
        qty = trade_size / price
        self.broker.buy_market(pair, trade_size, price)
        self.state.open_position(price, qty, pair=pair, cash_used=trade_size)
        self._log(f"BUY [{strategy_name}] {pair}  qty={qty:.6f} @ ${price:,.2f}"
                  f"  size=${trade_size:.2f}  ({Settings.POSITION_FRACTION*100:.0f}% du cash)")

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
    # Logging
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        os.makedirs(os.path.dirname(Settings.LIVE_LOG_FILE), exist_ok=True)
        with open(Settings.LIVE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
