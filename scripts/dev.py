"""Simple helper commands for everyday project development."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence

COMMANDS: dict[str, list[list[str]]] = {
    "format": [["ruff", "format", "."]],
    "check": [["ruff", "check", "."]],
    "test": [["pytest", "-q"]],
    "all": [["ruff", "format", "."], ["ruff", "check", "."], ["pytest", "-q"]],
}


def run_commands(command_groups: Sequence[Sequence[str]]) -> int:
    """Run shell commands in order and stop on the first failure."""
    for command in command_groups:
        print(f"$ {' '.join(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the helper script."""
    parser = argparse.ArgumentParser(
        description="Run simple development helpers for formatting, checking, and testing."
    )
    parser.add_argument(
        "command",
        choices=sorted(COMMANDS),
        help="Which helper to run: format, check, test, or all.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the selected helper command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_commands(COMMANDS[args.command])


if __name__ == "__main__":
    raise SystemExit(main())
