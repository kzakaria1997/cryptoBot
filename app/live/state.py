from __future__ import annotations

import json
import os
from datetime import datetime


class TradeState:
    """
    Persiste l'état de la position sur disque — survit aux redémarrages.
    Gère : entry_price, peak_price (trailing stop), paire, cash disponible.
    """

    def __init__(self, state_file: str):
        self.state_file    = state_file
        self.in_position   = False
        self.entry_price   = 0.0
        self.peak_price    = 0.0
        self.position_qty  = 0.0
        self.position_pair = ""
        self.entry_time    = None
        self.available_cash = 1000.0   # mis à jour après chaque trade
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.state_file):
            self.save()
            return
        with open(self.state_file, "r") as f:
            data = json.load(f)
        self.in_position    = data.get("in_position",   False)
        self.entry_price    = data.get("entry_price",   0.0)
        self.peak_price     = data.get("peak_price",    0.0)
        self.position_qty   = data.get("position_qty",  0.0)
        self.position_pair  = data.get("position_pair", "")
        self.entry_time     = data.get("entry_time",    None)
        self.available_cash = data.get("available_cash", 1000.0)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({
                "in_position":    self.in_position,
                "entry_price":    self.entry_price,
                "peak_price":     self.peak_price,
                "position_qty":   self.position_qty,
                "position_pair":  self.position_pair,
                "entry_time":     self.entry_time,
                "available_cash": self.available_cash,
                "last_update":    datetime.utcnow().isoformat(),
            }, f, indent=2)

    def open_position(self, price: float, qty: float,
                      pair: str = "", cash_used: float = 0.0) -> None:
        self.in_position    = True
        self.entry_price    = price
        self.peak_price     = price
        self.position_qty   = qty
        self.position_pair  = pair
        self.entry_time     = datetime.utcnow().isoformat()
        self.available_cash = max(0.0, self.available_cash - cash_used)
        self.save()

    def update_peak(self, current_price: float) -> None:
        if current_price > self.peak_price:
            self.peak_price = current_price
            self.save()

    def close_position(self) -> None:
        if self.in_position:
            # récupère le cash (approximation — le broker gère le vrai solde)
            proceeds = self.position_qty * self.peak_price * 0.9975  # net fees approx
            self.available_cash += proceeds
        self.in_position   = False
        self.entry_price   = 0.0
        self.peak_price    = 0.0
        self.position_qty  = 0.0
        self.position_pair = ""
        self.entry_time    = None
        self.save()
