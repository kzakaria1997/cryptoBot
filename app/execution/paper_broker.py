class PaperBroker:
    def __init__(self, fee_rate: float, slippage_rate: float):
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def buy(self, portfolio, price: float, usd_amount: float, timestamp):
        execution_price = price * (1 + self.slippage_rate)
        gross_qty = usd_amount / execution_price
        fee = usd_amount * self.fee_rate
        total_cost = usd_amount + fee

        if portfolio.cash < total_cost:
            return

        portfolio.cash -= total_cost
        portfolio.position_qty = gross_qty
        portfolio.entry_price = execution_price

        portfolio.trades.append({
            "side": "BUY",
            "timestamp": timestamp,
            "price": execution_price,
            "qty": gross_qty,
            "fee": fee,
        })

    def sell(self, portfolio, price: float, timestamp, reason: str = "UNKNOWN"):
        if not portfolio.in_position:
            return

        execution_price = price * (1 - self.slippage_rate)
        gross_value = portfolio.position_qty * execution_price
        fee = gross_value * self.fee_rate
        net_value = gross_value - fee

        pnl = net_value - (portfolio.position_qty * portfolio.entry_price)

        portfolio.cash += net_value

        portfolio.trades.append({
            "side": "SELL",
            "timestamp": timestamp,
            "price": execution_price,
            "qty": portfolio.position_qty,
            "fee": fee,
            "pnl": pnl,
            "reason": reason,
        })

        portfolio.position_qty = 0.0
        portfolio.entry_price = 0.0