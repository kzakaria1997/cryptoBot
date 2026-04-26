from __future__ import annotations

import numpy as np
import pandas as pd


class LabelGenerator:
    """
    Pour chaque bougie i, simule un long à close[i] et regarde les max_bars
    bougies suivantes : label=1 si le TP est touché avant le SL, 0 sinon.
    On utilise high/low pour détecter le toucher intra-bougie.
    """

    def __init__(self, stop_loss_pct: float, take_profit_pct: float, max_bars: int = 96):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_bars = max_bars

    def generate(self, df: pd.DataFrame) -> pd.Series:
        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        n = len(df)
        labels = np.zeros(n, dtype=np.int8)

        for i in range(n - 1):
            entry = closes[i]
            tp_price = entry * (1.0 + self.take_profit_pct)
            sl_price = entry * (1.0 - self.stop_loss_pct)

            end = min(i + 1 + self.max_bars, n)
            fut_highs = highs[i + 1 : end]
            fut_lows = lows[i + 1 : end]

            tp_hits = np.flatnonzero(fut_highs >= tp_price)
            sl_hits = np.flatnonzero(fut_lows <= sl_price)

            tp_idx = tp_hits[0] if len(tp_hits) else self.max_bars
            sl_idx = sl_hits[0] if len(sl_hits) else self.max_bars

            labels[i] = 1 if tp_idx < sl_idx else 0

        labels[n - 1] = -1  # dernière bougie : pas de données futures
        return pd.Series(labels, index=df.index, name="label")
