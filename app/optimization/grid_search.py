from __future__ import annotations

from itertools import product


class GridSearch:
    def __init__(self, strategy_class, engine_class):
        self.strategy_class = strategy_class
        self.engine_class = engine_class

    def run(self, df, param_grid: dict, engine_params: dict, top_n: int = 10) -> list[dict]:
        keys = list(param_grid.keys())
        values_product = product(*(param_grid[key] for key in keys))

        results = []

        for values in values_product:
            strategy_params = dict(zip(keys, values))
            strategy = self.strategy_class(strategy_params)

            engine = self.engine_class(strategy=strategy, **engine_params)
            result = engine.run(df)

            results.append({
                "strategy_params": strategy_params,
                "final_cash": result["final_cash"],
                "profit_factor": result["summary"]["profit_factor"],
                "win_rate": result["summary"]["win_rate"],
                "total_trades": result["summary"]["total_trades"],
                "max_drawdown": result["max_drawdown"],
                "total_pnl": result["summary"]["total_pnl"],
            })

        results.sort(
            key=lambda x: (x["profit_factor"], x["total_pnl"], -x["max_drawdown"]),
            reverse=True,
        )

        return results[:top_n]