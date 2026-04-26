from __future__ import annotations

import time
import krakenex


class LiveKrakenBroker:
    """
    Exécute de vrais ordres sur Kraken (ou simule en paper mode).
    Paper mode = aucun appel API privé, juste des logs.
    """

    def __init__(self, api_key: str, api_secret: str,
                 fee_rate: float = 0.0026, paper_mode: bool = True):
        self.fee_rate   = fee_rate
        self.paper_mode = paper_mode
        self._api = krakenex.API()
        if not paper_mode:
            self._api.key    = api_key
            self._api.secret = api_secret

    # ------------------------------------------------------------------ #
    # Prix
    # ------------------------------------------------------------------ #

    def get_ticker_price(self, pair: str) -> float:
        resp = self._api.query_public("Ticker", {"pair": pair})
        if resp.get("error"):
            raise RuntimeError(f"Ticker error: {resp['error']}")
        key = list(resp["result"].keys())[0]
        return float(resp["result"][key]["c"][0])

    # ------------------------------------------------------------------ #
    # Balance
    # ------------------------------------------------------------------ #

    def get_balance_usd(self) -> float:
        if self.paper_mode:
            return 9999.0
        resp = self._api.query_private("Balance")
        if resp.get("error"):
            raise RuntimeError(f"Balance error: {resp['error']}")
        return float(resp["result"].get("ZUSD", 0.0))

    # ------------------------------------------------------------------ #
    # Ordres
    # ------------------------------------------------------------------ #

    def buy_market(self, pair: str, usd_amount: float, current_price: float) -> dict:
        qty = round(usd_amount / current_price, 8)
        cost = usd_amount * (1 + self.fee_rate)

        if self.paper_mode:
            return {"paper": True, "qty": qty, "price": current_price, "cost": cost}

        resp = self._api.query_private("AddOrder", {
            "pair":      pair,
            "type":      "buy",
            "ordertype": "market",
            "volume":    str(qty),
        })
        if resp.get("error"):
            raise RuntimeError(f"Buy error: {resp['error']}")
        return resp["result"]

    def sell_market(self, pair: str, qty: float, current_price: float) -> dict:
        if self.paper_mode:
            proceeds = qty * current_price * (1 - self.fee_rate)
            return {"paper": True, "qty": qty, "price": current_price, "proceeds": proceeds}

        resp = self._api.query_private("AddOrder", {
            "pair":      pair,
            "type":      "sell",
            "ordertype": "market",
            "volume":    str(round(qty, 8)),
        })
        if resp.get("error"):
            raise RuntimeError(f"Sell error: {resp['error']}")
        return resp["result"]
