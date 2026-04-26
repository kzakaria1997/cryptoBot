from __future__ import annotations


class BacktestMetrics:
    @staticmethod
    def summarize_trades(trades: list[dict]) -> dict:
        sells = [t for t in trades if t["side"] == "SELL"]

        total_pnl = sum(t.get("pnl", 0.0) for t in sells)
        wins = [t for t in sells if t.get("pnl", 0.0) > 0]
        losses = [t for t in sells if t.get("pnl", 0.0) <= 0]

        gross_profit = sum(t["pnl"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0

        avg_win = gross_profit / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            "total_trades": len(sells),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(sells) * 100.0) if sells else 0.0,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
        }

    @staticmethod
    def calculate_max_drawdown(equity_curve: list[dict]) -> float:
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]["equity"]
        max_drawdown = 0.0

        for point in equity_curve:
            equity = point["equity"]
            if equity > peak:
                peak = equity

            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown * 100.0