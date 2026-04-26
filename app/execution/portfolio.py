class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []

    @property
    def in_position(self) -> bool:
        return self.position_qty > 0

    def current_equity(self, market_price: float) -> float:
        if self.in_position:
            return self.cash + (self.position_qty * market_price)
        return self.cash