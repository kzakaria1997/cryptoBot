from __future__ import annotations

from collections import Counter


class BacktestReport:
    @staticmethod
    def get_exit_reason_counts(trades: list[dict]) -> dict:
        sell_trades = [t for t in trades if t["side"] == "SELL"]
        reasons = [t.get("reason", "UNKNOWN") for t in sell_trades]
        return dict(Counter(reasons))

    @staticmethod
    def print_report(result: dict, show_trades: bool = False) -> None:
        summary = result["summary"]
        exit_reasons = result["exit_reasons"]

        print("=== BACKTEST RESULT ===")
        print(f"Strategy      : {result['strategy_name']}")
        print(f"Cash final    : {result['final_cash']:.2f}")
        print(f"Trades        : {summary['total_trades']}  (W:{summary['wins']} / L:{summary['losses']})")
        print(f"Win rate      : {summary['win_rate']:.2f}%")
        print(f"Profit factor : {summary['profit_factor']:.3f}")
        print(f"PnL total     : {summary['total_pnl']:.2f}")
        print(f"Avg win/loss  : {summary['avg_win']:.2f} / {summary['avg_loss']:.2f}")
        print(f"Max drawdown  : {result['max_drawdown']:.2f}%")
        exits = "  |  ".join(f"{r}:{c}" for r, c in exit_reasons.items())
        print(f"Exit reasons  : {exits}")

        if show_trades:
            print("\n=== TRADES ===")
            for trade in result["trades"]:
                print(trade)