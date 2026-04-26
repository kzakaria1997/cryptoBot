"""
Télécharge les données OHLCV 15min depuis l'API publique Binance.
Pas de clé API nécessaire. Supporte ETH/USDT, BTC/USDT, SOL/USDT...

Usage:
    python -m app.data.binance_downloader --symbol ETHUSDT --out app/data/ETHUSD_15.csv
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import pandas as pd
import requests

BINANCE_URL = "https://api.binance.com/api/v3/klines"
LIMIT       = 1000   # max par requête
INTERVAL    = "15m"


def _fetch_chunk(symbol: str, start_ms: int, end_ms: int) -> list:
    params = {
        "symbol":    symbol,
        "interval":  INTERVAL,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     LIMIT,
    }
    r = requests.get(BINANCE_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def download(symbol: str, start: str = "2017-01-01", out: str | None = None) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows: list[list] = []
    cursor = start_ms
    print(f"Téléchargement {symbol} 15min depuis {start}...")

    while cursor < end_ms:
        chunk = _fetch_chunk(symbol, cursor, end_ms)
        if not chunk:
            break
        all_rows.extend(chunk)
        cursor = int(chunk[-1][0]) + 1   # open_time du dernier + 1ms
        pct = (cursor - start_ms) / (end_ms - start_ms) * 100
        print(f"  {pct:.1f}%  ({len(all_rows):,} bougies)...", end="\r")
        time.sleep(0.1)   # politesse envers l'API

    print(f"\n  Total: {len(all_rows):,} bougies")

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if out:
        df.to_csv(out, index=False)
        print(f"  Sauvegardé : {out}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--start",  default="2017-01-01")
    parser.add_argument("--out",    default="app/data/ETHUSD_15.csv")
    args = parser.parse_args()
    download(args.symbol, args.start, args.out)
