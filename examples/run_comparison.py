"""
Run a few strategies on synthetic data and rank them by Algo Level Score.

This is the script you actually want to look at to see how the pieces fit together.
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from qaad.utils.data import make_synthetic_ohlcv
from qaad.strategies import (
    SMACrossover,
    RSIMeanReversion,
    BollingerMeanReversion,
    VolatilityBreakout,
)
from qaad.core.scoring import compare_algos

console = Console()


def main():
    console.print(Panel.fit("[bold cyan]QAAD — Strategy Comparison[/bold cyan]"))

    # synthetic data so the demo works offline
    data = make_synthetic_ohlcv(n=1500, volatility=0.012, drift=0.00015, seed=7)
    console.print(f"Data: {len(data)} bars  |  {data.index[0].date()} → {data.index[-1].date()}")
    console.print()

    strategies = [
        SMACrossover(fast=10, slow=40),
        RSIMeanReversion(period=14, lower=28, upper=72),
        BollingerMeanReversion(period=20, num_std=2.0),
        VolatilityBreakout(lookback=20, atr_mult=1.2),
    ]

    scores = {}
    results = {}

    for strat in strategies:
        console.print(f"Running [bold]{strat.name}[/bold] ...", end=" ")
        bt = strat.run(data, fee_bps=4.0, slippage_bps=2.0)
        score = strat.score(
            data,
            implementation_signals={
                "has_transaction_costs": 1.0,
                "no_lookahead": 1.0,
                "realistic_slippage": 1.0,
                "position_sizing_present": 0.5,
                "clean_code": 1.0,
            },
        )
        scores[strat.name] = score
        results[strat.name] = bt
        console.print(f"Sharpe {bt.metrics['sharpe']:.2f}  |  Level {score.total:.1f}")

    ranking = compare_algos(scores)

    console.print()
    table = Table(title="Ranked by Algo Level Score", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Strategy")
    table.add_column("Total", justify="right")
    table.add_column("Level")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("CAGR", justify="right")

    for idx, row in ranking.iterrows():
        name = row["algo"]
        m = results[name].metrics
        style = "green" if row["total"] >= 75 else "yellow" if row["total"] >= 55 else "red"
        table.add_row(
            str(idx),
            name,
            f"{row['total']:.1f}",
            f"[{style}]{row['level']}[/{style}]",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown']:.1%}",
            f"{m['cagr']:.1%}",
        )

    console.print(table)
    console.print()
    console.print("[dim]Higher score = stronger / more advanced quant algo on this data.[/dim]")
    console.print("[dim]Synthetic data is just for demo — replace with real OHLCV for real ranking.[/dim]")


if __name__ == "__main__":
    main()
