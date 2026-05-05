from __future__ import annotations

import time

import krakenex
import pandas as pd


class KrakenCollector:
    def __init__(self):
        self.api = krakenex.API()

    def get_ohlc(self, pair: str, interval: int = 15, num_candles: int = 3000) -> pd.DataFrame:
        """Récupère num_candles bougies en paginant l'API Kraken (max 720/appel)."""
        since = int(time.time()) - num_candles * interval * 60
        all_rows: list = []

        for _ in range(10):  # max 10 appels = 7200 bougies
            response = self.api.query_public("OHLC", {
                "pair":     pair,
                "interval": interval,
                "since":    since,
            })
            if response.get("error"):
                raise Exception(f"Kraken API error: {response['error']}")

            result   = response["result"]
            data_key = next(k for k in result if k != "last")
            rows     = result[data_key]

            if not rows:
                break

            all_rows.extend(rows)

            last = result.get("last")
            if last is None or len(rows) < 720:
                break
            since = last
            time.sleep(0.5)  # respecte le rate-limit Kraken

        df = pd.DataFrame(all_rows, columns=[
            "timestamp", "open", "high", "low", "close", "vwap", "volume", "count",
        ])
        df = df.drop_duplicates(subset=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        numeric_cols = ["open", "high", "low", "close", "vwap", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["count"] = df["count"].astype(int)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
