from __future__ import annotations

import krakenex
import pandas as pd


class KrakenCollector:
    def __init__(self):
        self.api = krakenex.API()

    def get_ohlc(self, pair: str, interval: int = 15) -> pd.DataFrame:
        response = self.api.query_public("OHLC", {"pair": pair, "interval": interval})

        if response.get("error"):
            raise Exception(f"Kraken API error: {response['error']}")

        result = response["result"]
        data_key = [key for key in result.keys() if key != "last"][0]
        rows = result[data_key]

        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "vwap",
                "volume",
                "count",
            ],
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        numeric_cols = ["open", "high", "low", "close", "vwap", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["count"] = df["count"].astype(int)

        return df