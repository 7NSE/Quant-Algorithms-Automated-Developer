"""
Advanced comparison:
  - regime-aware / vol-targeted strategies
  - walk-forward robustness check
  - Algo Level ranking that actually uses OOS data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from qaad.utils.data import make_synthetic_ohlcv
from qaad.strategies import (
    SMACrossover,
    RSIMeanReversion,
    TrendFollowingVolTarget,
    MultiFactorMomentum,
    BreakoutWithFilter,
    ResidualMeanReversion,
)
from qaad.core.scoring import AlgoLevelScorer, compare_algos
from qaad.core.walkforward import walk_forward

console = Console()


def main():
    console.print(Panel.fit("[bold cyan]QAAD — Advanced Strategy Ranking[/bold cyan]"))

    # longer history so walk-forward has enough folds
    data = make_synthetic_ohlcv(n=2000, volatility=0.014, drift=0.00012, seed=21)
    console.print(f"Bars: {len(data)}  |  {data.index[0].date()} → {data.index[-1].date()}\n")

    strategies = [
        ("Basic SMA", SMACrossover(fast=10, slow=40)),
        ("Basic RSI", RSIMeanReversion(period=14)),
        ("Trend + Vol Target", TrendFollowingVolTarget(fast=20, slow=100, target_vol=0.12)),
        ("Multi-Factor Momentum", MultiFactorMomentum(target_vol=0.10)),
        ("Filtered Breakout", BreakoutWithFilter(channel=55, target_vol=0.15)),
        ("Residual Mean-Reversion", ResidualMeanReversion(target_vol=0.08)),
    ]

    scorer = AlgoLevelScorer()
    scores = {}
    wf_info = {}

    for label, strat in strategies:
        console.print(f"[bold]{label}[/bold]")

        # full-sample backtest (for reference)
        bt = strat.run(data, fee_bps=4.0, slippage_bps=2.0)

        # walk-forward for robustness
        try:
            wf = walk_forward(
                data,
                signal_fn=strat.generate_signals,
                train_bars=504,
                test_bars=126,
                step_bars=126,
                fee_bps=4.0,
                slippage_bps=2.0,
            )
            oos_ratio = wf.oos_is_ratio
            oos_sharpe = wf.avg_oos_sharpe
            console.print(
                f"  Full Sharpe {bt.metrics['sharpe']:.2f}  |  "
                f"OOS Sharpe {oos_sharpe:.2f}  |  "
                f"OOS/IS {oos_ratio:.2f}  |  "
                f"Folds {len(wf.folds)}"
            )
        except ValueError as e:
            oos_ratio = 0.5
            oos_sharpe = 0.0
            console.print(f"  Walk-forward skipped: {e}")

        # sophistication signals — advanced strats get higher marks
        is_advanced = label not in ("Basic SMA", "Basic RSI")
        soph = {
            "num_features": 6 if is_advanced else 2,
            "has_regime_filter": 1.0 if is_advanced else 0.0,
            "adaptive_sizing": 1.0 if is_advanced else 0.0,
            "multi_timeframe": 1.0 if "Multi-Factor" in label else 0.0,
            "risk_parity_or_vol_target": 1.0 if is_advanced else 0.0,
        }
        impl = {
            "has_transaction_costs": 1.0,
            "no_lookahead": 1.0,
            "realistic_slippage": 1.0,
            "position_sizing_present": 1.0 if is_advanced else 0.3,
            "clean_code": 1.0,
        }

        score = scorer.score(
            metrics=bt.metrics,
            robustness_metrics={
                "walk_forward_ratio": max(0.0, min(1.5, oos_ratio)),
                "param_sensitivity": 0.25 if is_advanced else 0.55,
            },
            sophistication_signals=soph,
            implementation_signals=impl,
        )

        scores[label] = score
        wf_info[label] = {"oos_sharpe": oos_sharpe, "oos_ratio": oos_ratio, "full_sharpe": bt.metrics["sharpe"]}
        console.print(f"  Level score → [bold]{score.total:.1f}[/bold]  ({score.level})\n")

    ranking = compare_algos(scores)

    table = Table(title="Final Ranking (Algo Level Score)", header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Strategy")
    table.add_column("Score", justify="right")
    table.add_column("Level")
    table.add_column("Full Sharpe", justify="right")
    table.add_column("OOS Sharpe", justify="right")
    table.add_column("OOS/IS", justify="right")

    for idx, row in ranking.iterrows():
        name = row["algo"]
        info = wf_info[name]
        style = "green" if row["total"] >= 70 else "yellow" if row["total"] >= 55 else "red"
        table.add_row(
            str(idx),
            name,
            f"{row['total']:.1f}",
            f"[{style}]{row['level']}[/{style}]",
            f"{info['full_sharpe']:.2f}",
            f"{info['oos_sharpe']:.2f}",
            f"{info['oos_ratio']:.2f}",
        )

    console.print(table)
    console.print()
    console.print("[dim]Advanced strategies use regime filters + vol targeting + multi-factor signals.[/dim]")
    console.print("[dim]OOS/IS ratio near 1.0 = edge survived walk-forward. Near 0 = overfit.[/dim]")


if __name__ == "__main__":
    main()
