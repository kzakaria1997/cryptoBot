from __future__ import annotations

import pandas as pd

from app.strategy.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        default_params = {
            "rsi_min": 54,
            "rsi_max": 58,
            "rel_volume_min": 1.35,
            "return_3_min": 0.003,
            "volatility_10_min": 0.0015,
            "use_exit_signal": False,
            "trend_gap_min": 0.0,
        }

        merged = default_params.copy()
        if params:
            merged.update(params)

        super().__init__(name="momentum_strategy", params=merged)

    def required_columns(self) -> list[str]:
        return [
            "close",
            "ema_fast",
            "ema_slow",
            "ema_trend",
            "ema_base",
            "rsi",
            "rel_volume",
            "return_3",
            "volatility_10",
        ]

    def generate_entry_signal(self, row: pd.Series) -> bool:
        trend_gap_pct = (row["ema_trend"] - row["ema_base"]) / row["ema_base"]

        trend_is_bullish = (
            row["ema_trend"] > row["ema_base"]
            and row["ema_fast"] > row["ema_slow"]
            and row["close"] > row["ema_slow"]
            and trend_gap_pct >= self.params["trend_gap_min"]
        )

        momentum_ok = (
            self.params["rsi_min"] <= row["rsi"] <= self.params["rsi_max"]
            and row["rel_volume"] >= self.params["rel_volume_min"]
            and row["return_3"] >= self.params["return_3_min"]
            and row["volatility_10"] >= self.params["volatility_10_min"]
        )

        return trend_is_bullish and momentum_ok

    def generate_exit_signal(self, row: pd.Series) -> bool:
        if not self.params["use_exit_signal"]:
            return False

        return (
            row["close"] < row["ema_fast"]
            and row["rsi"] < 45
        )