from __future__ import annotations

import pandas as pd

from app.backtest.metrics import BacktestMetrics
from app.backtest.report import BacktestReport
from app.execution.paper_broker import PaperBroker
from app.execution.portfolio import Portfolio


class BacktestEngine:
    def __init__(
        self,
        strategy,
        fee_rate: float,
        slippage_rate: float,
        initial_cash: float,
        trade_size_usd: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        use_exit_signal: bool = False,
        predictor=None,
        ml_threshold: float = 0.60,
    ):
        self.strategy = strategy
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.initial_cash = initial_cash
        self.trade_size_usd = trade_size_usd
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_exit_signal = use_exit_signal
        self.predictor = predictor
        self.ml_threshold = ml_threshold

    def run(self, df: pd.DataFrame) -> dict:
        df = self.strategy.prepare_dataframe(df)

        if self.predictor is not None:
            df = df.copy()
            df["ml_score"] = self.predictor.predict_dataframe(df)

        portfolio = Portfolio(self.initial_cash)
        broker = PaperBroker(self.fee_rate, self.slippage_rate)

        for _, row in df.iterrows():
            if not self.strategy.validate_row(row):
                continue

            current_price = float(row["close"])
            timestamp = row["timestamp"]

            portfolio.equity_curve.append({
                "timestamp": timestamp,
                "equity": portfolio.current_equity(current_price),
            })

            if portfolio.in_position:
                pnl_pct = (current_price - portfolio.entry_price) / portfolio.entry_price

                if pnl_pct <= -self.stop_loss_pct:
                    broker.sell(portfolio, current_price, timestamp, reason="STOP_LOSS")
                    continue

                if pnl_pct >= self.take_profit_pct:
                    broker.sell(portfolio, current_price, timestamp, reason="TAKE_PROFIT")
                    continue

                if self.use_exit_signal and self.strategy.generate_exit_signal(row):
                    broker.sell(portfolio, current_price, timestamp, reason="SIGNAL_EXIT")
                    continue

            ml_ok = (
                self.predictor is None
                or (
                    "ml_score" in row.index
                    and not pd.isna(row["ml_score"])
                    and float(row["ml_score"]) >= self.ml_threshold
                )
            )

            if not portfolio.in_position and self.strategy.generate_entry_signal(row) and ml_ok:
                broker.buy(portfolio, current_price, self.trade_size_usd, timestamp)

        if portfolio.in_position:
            last_row = df.iloc[-1]
            broker.sell(
                portfolio,
                float(last_row["close"]),
                last_row["timestamp"],
                reason="FINAL_CLOSE",
            )

        summary = BacktestMetrics.summarize_trades(portfolio.trades)
        max_drawdown = BacktestMetrics.calculate_max_drawdown(portfolio.equity_curve)
        exit_reasons = BacktestReport.get_exit_reason_counts(portfolio.trades)

        return {
            "strategy_name": self.strategy.get_name(),
            "strategy_params": self.strategy.get_params(),
            "final_cash": portfolio.cash,
            "summary": summary,
            "max_drawdown": max_drawdown,
            "exit_reasons": exit_reasons,
            "trades": portfolio.trades,
            "equity_curve": portfolio.equity_curve,
        }