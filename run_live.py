"""
Service de trading live.

AVANT DE LANCER :
  1. Copier .env.example -> .env et remplir KRAKEN_API_KEY + KRAKEN_API_SECRET
  2. Entrainer le modele ML :  python train.py
  3. Tester en paper mode d'abord (Settings.PAPER_MODE = True)
  4. Pour vrais trades : mettre PAPER_MODE = False dans settings.py

Usage:
  python run_live.py
"""
from app.live.trader import LiveTrader

if __name__ == "__main__":
    LiveTrader().run()
