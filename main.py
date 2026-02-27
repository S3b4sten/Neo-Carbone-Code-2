"""main.py – CLI entry point for the Self-Evolving Agent.

Usage:
    python main.py "Your goal here"
    python main.py --interactive
"""

import argparse
import os
import sys

from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt

import config
import tools as tool_registry


def _check_api_key() -> None:
    key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        rprint(
            "[bold red]ERROR:[/] No OpenAI API key found.\n"
            "Set [cyan]OPENAI_API_KEY[/] as an environment variable, e.g.:\n"
            '  [dim]$env:OPENAI_API_KEY = "sk-..."[/]  (PowerShell)\n'
            '  [dim]export OPENAI_API_KEY="sk-..."[/]  (Bash)'
        )
        sys.exit(1)
    config.OPENAI_API_KEY = key  # ensure it's set in config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-Evolving AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "Calculate the 20th Fibonacci number"\n'
            '  python main.py "Search the web for today\'s top AI news and summarise it"\n'
            "  python main.py --interactive"
        ),
    )
    parser.add_argument("goal", nargs="?", default=None, help="Goal for the agent to accomplish.")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive REPL mode (enter goals one at a time).",
    )
    parser.add_argument(
        "--model", "-m",
        default=config.MODEL,
        help=f"LLM model to use (default: {config.MODEL}).",
    )
    args = parser.parse_args()

    _check_api_key()
    config.MODEL = args.model

    # Initialise tool registry
    tool_registry.initialise()

    tool_list = tool_registry.list_tools()
    rprint(Panel(
        f"[bold cyan]Self-Evolving Agent[/]  |  Model: [yellow]{config.MODEL}[/]\n"
        f"Tools loaded: [green]{len(tool_list)}[/]  |  "
        f"Workspace: [dim]{config.WORKSPACE_DIR}[/]",
        title="🤖 Ready",
        border_style="cyan",
    ))

    from agent.core import SelfEvolvingAgent
    agent = SelfEvolvingAgent()

    if args.interactive:
        rprint("[dim]Interactive mode – type 'exit' or 'quit' to stop.[/]")
        while True:
            goal = Prompt.ask("\n[bold green]Goal[/]").strip()
            if goal.lower() in ("exit", "quit", "q"):
                rprint("[dim]Goodbye.[/]")
                break
            if not goal:
                continue
            outcome = agent.run(goal)
            rprint(Panel(outcome, title="[bold green]Outcome[/]", border_style="green"))
    elif args.goal:
        outcome = agent.run(args.goal)
        rprint(Panel(outcome, title="[bold green]Outcome[/]", border_style="green"))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
