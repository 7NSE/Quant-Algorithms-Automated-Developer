"""
CLI entrypoint for scoring an existing quant strategy / bot.

Usage:
    python -m qaad.score --metrics metrics.json
    python -m qaad.score --example
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from qaad.core.scoring import AlgoLevelScorer, compare_algos

console = Console()


def run_example():
    """Demonstrate scoring on a few fictional algos."""
    scorer = AlgoLevelScorer()

    # Example 1: Strong institutional-style strategy
    elite = scorer.score(
        metrics={
            "sharpe": 1.85,
            "sortino": 2.6,
            "profit_factor": 1.9,
            "cagr": 0.22,
            "max_drawdown": 0.11,
            "win_rate": 0.58,
        },
        robustness_metrics={
            "walk_forward_ratio": 0.88,
            "param_sensitivity": 0.18,
        },
        sophistication_signals={
            "num_features": 8,
            "has_regime_filter": 1.0,
            "adaptive_sizing": 1.0,
            "multi_timeframe": 1.0,
            "risk_parity_or_vol_target": 1.0,
        },
        implementation_signals={
            "has_transaction_costs": 1.0,
            "no_lookahead": 1.0,
            "realistic_slippage": 1.0,
            "clean_code": 1.0,
            "position_sizing_present": 1.0,
        },
    )

    # Example 2: Typical retail mean-reversion bot
    retail = scorer.score(
        metrics={
            "sharpe": 0.95,
            "sortino": 1.3,
            "profit_factor": 1.35,
            "cagr": 0.14,
            "max_drawdown": 0.22,
            "win_rate": 0.61,
        },
        robustness_metrics={
            "walk_forward_ratio": 0.62,
            "param_sensitivity": 0.55,
        },
        sophistication_signals={
            "num_features": 3,
            "has_regime_filter": 0.0,
            "adaptive_sizing": 0.0,
        },
        implementation_signals={
            "has_transaction_costs": 0.5,
            "no_lookahead": 1.0,
            "position_sizing_present": 0.5,
        },
    )

    # Example 3: Overfitted / weak strategy
    weak = scorer.score(
        metrics={
            "sharpe": 2.8,  # looks great in-sample
            "sortino": 3.5,
            "profit_factor": 2.4,
            "cagr": 0.45,
            "max_drawdown": 0.07,
            "win_rate": 0.72,
        },
        robustness_metrics={
            "walk_forward_ratio": 0.25,  # collapses out-of-sample
            "param_sensitivity": 0.85,
        },
        sophistication_signals={
            "num_features": 15,  # many features but no real edge
            "has_regime_filter": 0.0,
        },
        implementation_signals={
            "has_transaction_costs": 0.0,
            "no_lookahead": 0.0,  # look-ahead bias present
        },
    )

    scores = {
        "Elite Momentum + Regime": elite,
        "Retail RSI Mean-Reversion": retail,
        "Overfitted High-Sharpe": weak,
    }

    ranking = compare_algos(scores)

    console.print(Panel.fit("[bold cyan]QAAD — Algo Level Comparison Demo[/bold cyan]"))
    console.print()

    table = Table(title="Ranked Strategies", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim")
    table.add_column("Algo")
    table.add_column("Total", justify="right")
    table.add_column("Level")
    table.add_column("Perf", justify="right")
    table.add_column("Rob", justify="right")
    table.add_column("Risk", justify="right")
    table.add_column("Soph", justify="right")
    table.add_column("Impl", justify="right")

    for idx, row in ranking.iterrows():
        level_style = "green" if row["total"] >= 75 else "yellow" if row["total"] >= 60 else "red"
        table.add_row(
            str(idx),
            row["algo"],
            f"{row['total']:.1f}",
            f"[{level_style}]{row['level']}[/{level_style}]",
            f"{row['performance']:.0f}",
            f"{row['robustness']:.0f}",
            f"{row['risk_control']:.0f}",
            f"{row['sophistication']:.0f}",
            f"{row['implementation']:.0f}",
        )

    console.print(table)
    console.print()
    console.print("[dim]Higher total score = stronger / more advanced quant algo or bot.[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Score quant algorithms / bots")
    parser.add_argument("--example", action="store_true", help="Run the built-in comparison demo")
    parser.add_argument("--metrics", type=str, help="Path to JSON file with metrics")
    args = parser.parse_args()

    if args.example or not args.metrics:
        run_example()
        return

    path = Path(args.metrics)
    data = json.loads(path.read_text())
    scorer = AlgoLevelScorer()
    result = scorer.score(
        metrics=data.get("metrics", data),
        robustness_metrics=data.get("robustness_metrics"),
        sophistication_signals=data.get("sophistication_signals"),
        implementation_signals=data.get("implementation_signals"),
    )

    console.print(Panel.fit(f"[bold]Algo Level Score: {result.total:.1f}/100[/bold]\n{result.level}"))
    console.print(result.to_dict())


if __name__ == "__main__":
    main()
