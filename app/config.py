from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    INITIAL_CASH = 1000.0
    TRADING_PAIR = "XXBTZUSD"
    TIMEFRAME_MINUTES = 15

    # Simulation
    FEE_RATE = 0.0025
    SLIPPAGE_RATE = 0.0005
    TRADE_SIZE_USD = 100.0
    STOP_LOSS_PCT = 0.006
    TAKE_PROFIT_PCT = 0.018

    # Strategy
    FAST_EMA = 9
    SLOW_EMA = 21
    RSI_PERIOD = 14
    RSI_BUY_MIN = 48
    RSI_BUY_MAX = 58

    USE_CSV = True
    CSV_FILEPATH = "app/data/XBTUSD_15.csv"