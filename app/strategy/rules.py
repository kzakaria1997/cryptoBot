def generate_signal(row, rsi_buy_min: float, rsi_buy_max: float) -> str:
    trend_is_bullish = (
        row["ema_trend"] > row["ema_base"]
        and row["ema_fast"] > row["ema_slow"]
        and row["close"] > row["ema_slow"]
    )

    buy_condition = (
        trend_is_bullish
        and 54 <= row["rsi"] <= 58
        and row["rel_volume"] > 1.35
        and row["return_3"] > 0.003
        and row["volatility_10"] > 0.0015
    )

    if buy_condition:
        return "BUY"

    return "HOLD"