from __future__ import annotations

import pandas as pd

from app.strategy.base_strategy import BaseStrategy


class PullbackStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        default_params = {
            "rsi_min": 48,
            "rsi_max": 58,
            "rel_volume_min": 1.05,
            "volatility_10_min": 0.0012,
            "trend_gap_min": 0.001,
            "pullback_fast_min": -0.0035,
            "pullback_fast_max": 0.0015,
            "pullback_slow_min": -0.0025,
            "use_exit_signal": True,
            "exit_rsi_threshold": 43,
        }

        merged = default_params.copy()
        if params:
            merged.update(params)

        super().__init__(name="pullback_strategy", params=merged)

    def required_columns(self) -> list[str]:
        return [
            "close",
            "ema_fast",
            "ema_slow",
            "ema_trend",
            "ema_base",
            "rsi",
            "rel_volume",
            "return_1",
            "volatility_10",
        ]

    def generate_entry_signal(self, row: pd.Series) -> bool:
        trend_gap_pct = (row["ema_trend"] - row["ema_base"]) / row["ema_base"]
        pullback_to_fast = (row["close"] - row["ema_fast"]) / row["ema_fast"]
        pullback_to_slow = (row["close"] - row["ema_slow"]) / row["ema_slow"]

        trend_is_bullish = (
            row["ema_trend"] > row["ema_base"]
            and row["ema_fast"] > row["ema_slow"]
            and row["close"] > row["ema_base"]
            and trend_gap_pct >= self.params["trend_gap_min"]
        )

        pullback_is_clean = (
            self.params["pullback_fast_min"] <= pullback_to_fast <= self.params["pullback_fast_max"]
            and pullback_to_slow >= self.params["pullback_slow_min"]
            and row["return_1"] > 0
        )

        momentum_ok = (
            self.params["rsi_min"] <= row["rsi"] <= self.params["rsi_max"]
            and row["rel_volume"] >= self.params["rel_volume_min"]
            and row["volatility_10"] >= self.params["volatility_10_min"]
        )

        return trend_is_bullish and pullback_is_clean and momentum_ok

    def generate_exit_signal(self, row: pd.Series) -> bool:
        if not self.params["use_exit_signal"]:
            return False

        return (
            row["close"] < row["ema_slow"]
            or row["rsi"] < self.params["exit_rsi_threshold"]
        )