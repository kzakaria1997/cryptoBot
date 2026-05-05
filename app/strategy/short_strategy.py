from __future__ import annotations

import pandas as pd

from app.strategy.base_strategy import BaseStrategy


class ShortMomentumStrategy(BaseStrategy):
    """
    Signal SHORT : entre en position courte quand le marche est en downtrend
    avec un rebond technique (RSI eleve dans un downtrend = signal de vente).

    Conditions d'entree short :
    - Tendance baissiere confirmee (EMA50 < EMA200, gap > 0.5%)
    - RSI en zone "elevee" dans un downtrend (55-70) = rebond a vendre
    - Volume eleve (confirmation)
    - return_30d negatif (marche baissier macro)
    """

    def __init__(self, params: dict | None = None):
        default_params = {
            "rsi_min":          58,      # RSI elevé = rebond a vendre (58-72)
            "rsi_max":          72,
            "rel_volume_min":   1.4,     # volume fort sur le rebond
            "trend_gap_max":   -0.008,   # EMA50 < EMA200 de plus de 0.8% (bear fort)
            "return_30d_max":  -0.12,    # return 30j < -12% (vrai bear market)
            "return_7d_max":   -0.05,    # return 7j encore negatif (pas en recovery V-shape)
        }
        merged = default_params.copy()
        if params:
            merged.update(params)
        super().__init__(name="short_momentum", params=merged)

    def required_columns(self) -> list[str]:
        return [
            "close", "ema_fast", "ema_slow", "ema_trend", "ema_base",
            "rsi", "rel_volume", "return_7d", "return_30d", "trend_gap_pct",
        ]

    def generate_entry_signal(self, row: pd.Series) -> bool:
        trend_gap = float(row["trend_gap_pct"])
        rsi       = float(row["rsi"])
        rvol      = float(row["rel_volume"])
        r30       = row.get("return_30d")
        r7        = row.get("return_7d")

        bearish_trend = (
            trend_gap <= self.params["trend_gap_max"]
            and float(row["ema_trend"]) < float(row["ema_base"])
            and float(row["ema_fast"]) < float(row["ema_slow"])
        )

        overbought_in_downtrend = (
            self.params["rsi_min"] <= rsi <= self.params["rsi_max"]
            and rvol >= self.params["rel_volume_min"]
        )

        bearish_regime_30d = (
            r30 is not None
            and not pd.isna(r30)
            and float(r30) <= self.params["return_30d_max"]
        )

        # Filtre anti V-shape : marche toujours en descente sur 7j (pas en recovery)
        bearish_regime_7d = (
            r7 is not None
            and not pd.isna(r7)
            and float(r7) <= self.params["return_7d_max"]
        )

        return bearish_trend and overbought_in_downtrend and bearish_regime_30d and bearish_regime_7d

    def generate_exit_signal(self, row: pd.Series) -> bool:
        return False
