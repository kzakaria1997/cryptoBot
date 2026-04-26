from __future__ import annotations

import pandas as pd


class CsvLoader:
    def get_ohlc(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, header=None)

        if df.shape[1] == 7:
            df.columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "count",
            ]
        elif df.shape[1] == 8:
            df.columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "vwap",
                "volume",
                "count",
            ]
        else:
            raise ValueError(f"Unexpected number of columns: {df.shape[1]}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "count" in df.columns:
            df["count"] = pd.to_numeric(df["count"], errors="coerce")

        df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df