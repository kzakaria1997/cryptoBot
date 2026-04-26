import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_features(df: pd.DataFrame, fast_ema: int, slow_ema: int, rsi_period: int) -> pd.DataFrame:
    df = df.copy()

    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df["ema_fast"] = df["close"].ewm(span=fast_ema, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_ema, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_base"] = df["close"].ewm(span=200, adjust=False).mean()

    df["rsi"] = compute_rsi(df["close"], rsi_period)

    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["rel_volume"] = df["volume"] / df["volume_ma"]

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)

    df["volatility_10"] = df["close"].pct_change().rolling(10).std()

    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    return df