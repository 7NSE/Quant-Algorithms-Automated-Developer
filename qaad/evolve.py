"""
Evolutionary strategy discovery entrypoint (scaffold).

Future versions will run a full multi-agent evolutionary loop:
  hypothesis → code generation → backtest → score → mutate / combine → repeat
"""

from __future__ import annotations

import argparse
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evolve quant strategies from an idea")
    parser.add_argument("--idea", type=str, required=True, help="Natural language strategy idea")
    parser.add_argument("--generations", type=int, default=10, help="Number of evolutionary generations")
    parser.add_argument("--population", type=int, default=8, help="Strategies per generation")
    args = parser.parse_args()

    console.print(Panel.fit(
        f"[bold cyan]QAAD Evolutionary Loop (Scaffold)[/bold cyan]\n\n"
        f"Idea: [yellow]{args.idea}[/yellow]\n"
        f"Generations: {args.generations} | Population: {args.population}\n\n"
        "[dim]Full multi-agent evolutionary engine is under development.\n"
        "Current version provides the scoring & ranking core.\n"
        "Next releases will add: hypothesis agents, strategy coder, backtester integration,[/dim]"
    ))

    console.print("\n[green]✓[/green] Scaffold ready. Use [bold]python -m qaad.score --example[/bold] to see the ranking engine in action.")


if __name__ == "__main__":
    main()
