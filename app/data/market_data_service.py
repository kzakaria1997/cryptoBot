from __future__ import annotations

import pandas as pd

from app.data.csv_loader import CsvLoader
from app.data.collector import KrakenCollector


class MarketDataService:
    def __init__(self, use_csv: bool, csv_filepath: str, trading_pair: str, timeframe_minutes: int):
        self.use_csv = use_csv
        self.csv_filepath = csv_filepath
        self.trading_pair = trading_pair
        self.timeframe_minutes = timeframe_minutes

    def load_ohlc(self) -> pd.DataFrame:
        if self.use_csv:
            loader = CsvLoader()
            return loader.get_ohlc(self.csv_filepath)

        collector = KrakenCollector()
        return collector.get_ohlc(self.trading_pair, self.timeframe_minutes)