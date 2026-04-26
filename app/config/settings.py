import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    INITIAL_CASH = 1000.0
    TIMEFRAME_MINUTES = 15

    FEE_RATE      = 0.0025
    SLIPPAGE_RATE = 0.0005

    FAST_EMA   = 9
    SLOW_EMA   = 21
    RSI_PERIOD = 14

    USE_CSV       = True
    CSV_FILEPATH  = "app/data/XBTUSD_15.csv"

    # ------------------------------------------------------------------ #
    # Paramètres stratégie (backtest-optimisés, trail + regime filter)
    # ------------------------------------------------------------------ #
    STOP_LOSS_PCT      = 0.010   # SL fixe 1% depuis l'entrée
    TRAIL_STOP_PCT     = 0.250   # trailing stop 25% depuis le peak
    POSITION_FRACTION  = 0.40    # 40% du portefeuille par trade

    MOMENTUM_PARAMS = {
        "rsi_min":        58,
        "rsi_max":        62,
        "rel_volume_min": 1.3,
        "return_3_min":   0.001,
        "trend_gap_min":  0.001,
    }
    PULLBACK_PARAMS = {
        "rsi_min":        50,
        "rsi_max":        55,
        "rel_volume_min": 1.1,
        "trend_gap_min":  0.001,
    }

    # Regime filter : évite les marchés baissiers
    REGIME_RETURN_30D_MIN = -0.03   # return 30j >= -3% pour trader

    # Paires à trader
    TRADING_PAIRS = ["XXBTZUSD", "XETHZUSD"]   # BTC + ETH
    TRADING_PAIR  = "XXBTZUSD"                  # paire principale (legacy)

    # ML
    ML_MODEL_PATH            = "app/ml/model_oos.pkl"
    ML_CONFIDENCE_THRESHOLD  = 0.52
    ML_MAX_BARS              = 96

    # Live trading — PAPER_MODE=false dans Railway pour activer les vrais ordres
    PAPER_MODE       = os.getenv("PAPER_MODE", "true").lower() != "false"
    LIVE_STATE_FILE  = "app/live/state.json"
    LIVE_LOG_FILE    = "app/live/trade_log.txt"
    KRAKEN_WARMUP_CANDLES = 500

    KRAKEN_API_KEY    = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

    # Rapport hebdomadaire par email (Resend.com)
    REPORT_EMAIL   = os.getenv("REPORT_EMAIL",   "")  # adresse de réception
    RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")  # clé API resend.com
