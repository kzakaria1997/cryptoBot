from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngine:
    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def add_features(
        df: pd.DataFrame,
        fast_ema: int = 9,
        slow_ema: int = 21,
        trend_ema: int = 50,
        base_ema: int = 200,
        rsi_period: int = 14,
    ) -> pd.DataFrame:
        result = df.copy()

        for col in ["close", "high", "low", "volume"]:
            result[col] = result[col].astype(float)

        # --- EMAs de base ---
        result["ema_fast"] = result["close"].ewm(span=fast_ema, adjust=False).mean()
        result["ema_slow"] = result["close"].ewm(span=slow_ema, adjust=False).mean()
        result["ema_trend"] = result["close"].ewm(span=trend_ema, adjust=False).mean()
        result["ema_base"] = result["close"].ewm(span=base_ema, adjust=False).mean()

        # --- RSI ---
        result["rsi"] = FeatureEngine.compute_rsi(result["close"], rsi_period)
        result["rsi_lag1"] = result["rsi"].shift(1)
        result["rsi_lag2"] = result["rsi"].shift(2)
        result["rsi_lag3"] = result["rsi"].shift(3)

        # --- Volume ---
        result["volume_ma"] = result["volume"].rolling(20).mean()
        result["rel_volume"] = result["volume"] / result["volume_ma"]
        result["vol_ratio_5"] = result["volume"] / result["volume"].rolling(5).mean()

        # --- Retours ---
        result["return_1"] = result["close"].pct_change(1)
        result["return_3"] = result["close"].pct_change(3)
        result["return_6"] = result["close"].pct_change(6)
        result["return_1_lag1"] = result["return_1"].shift(1)
        result["return_1_lag2"] = result["return_1"].shift(2)
        result["return_1_lag3"] = result["return_1"].shift(3)

        # --- Volatilité / Range ---
        result["volatility_10"] = result["close"].pct_change().rolling(10).std()
        result["range_pct"] = (result["high"] - result["low"]) / result["close"]

        # --- ATR normalisé ---
        tr1 = result["high"] - result["low"]
        tr2 = (result["high"] - result["close"].shift(1)).abs()
        tr3 = (result["low"] - result["close"].shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result["atr_pct"] = true_range.rolling(14).mean() / result["close"]

        # --- MACD normalisé par le prix ---
        ema12 = result["close"].ewm(span=12, adjust=False).mean()
        ema26 = result["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        result["macd_pct"] = macd / result["close"]
        result["macd_signal_pct"] = macd_signal / result["close"]
        result["macd_hist_pct"] = (macd - macd_signal) / result["close"]

        # --- Bollinger Bands ---
        bb_mid = result["close"].rolling(20).mean()
        bb_std = result["close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        result["bb_width"] = (bb_upper - bb_lower) / bb_mid
        result["bb_pct"] = (result["close"] - bb_lower) / (bb_upper - bb_lower)

        # --- Stochastique (14, 3) ---
        low14 = result["low"].rolling(14).min()
        high14 = result["high"].rolling(14).max()
        denom = (high14 - low14).replace(0, np.nan)
        result["stoch_k"] = (result["close"] - low14) / denom * 100
        result["stoch_d"] = result["stoch_k"].rolling(3).mean()

        # --- Distances EMA (%) ---
        result["dist_ema_fast_pct"] = (result["close"] - result["ema_fast"]) / result["ema_fast"]
        result["dist_ema_slow_pct"] = (result["close"] - result["ema_slow"]) / result["ema_slow"]
        result["ema_gap_pct"] = (result["ema_fast"] - result["ema_slow"]) / result["ema_slow"]
        result["trend_gap_pct"] = (result["ema_trend"] - result["ema_base"]) / result["ema_base"]

        # --- Features temporelles ---
        if pd.api.types.is_datetime64_any_dtype(result["timestamp"]):
            result["hour"] = result["timestamp"].dt.hour
            result["dayofweek"] = result["timestamp"].dt.dayofweek
        else:
            result["hour"] = 0
            result["dayofweek"] = 0

        # ------------------------------------------------------------------ #
        # FEATURES DE REGIME DE MARCHE
        # Pour bougies 15min : 1j=96, 7j=672, 30j=2880, 20j=1920
        # ------------------------------------------------------------------ #

        # Retours multi-timeframes : dit au modele si on est en bull/bear
        result["return_1d"] = result["close"].pct_change(96)
        result["return_7d"] = result["close"].pct_change(672)
        result["return_30d"] = result["close"].pct_change(2880)

        # RSI lent (50 periodes) : momentum moyen terme
        result["rsi_slow"] = FeatureEngine.compute_rsi(result["close"], 50)

        # Regime de volatilite : court terme vs long terme
        result["vol_long"] = result["close"].pct_change().rolling(100).std()
        result["vol_regime"] = result["volatility_10"] / result["vol_long"].replace(0, np.nan)

        # Position dans le range 20 jours
        high_20d = result["high"].rolling(1920).max()
        low_20d = result["low"].rolling(1920).min()
        range_20d = (high_20d - low_20d).replace(0, np.nan)
        result["dist_high_20d"] = (result["close"] - high_20d) / result["close"]   # <= 0 si sous le high
        result["dist_low_20d"] = (result["close"] - low_20d) / result["close"]     # >= 0 si au dessus du low
        result["pos_in_range_20d"] = (result["close"] - low_20d) / range_20d       # 0=bas du range, 1=haut

        # Flag regime bull/bear macro
        result["regime_bull"] = (
            (result["close"] > result["ema_base"]) &
            (result["ema_trend"] > result["ema_base"])
        ).astype(int)

        return result
