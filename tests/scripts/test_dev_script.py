"""Tests for the simple development helper script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_dev_module():
    module_path = Path("scripts/dev.py")
    spec = importlib.util.spec_from_file_location("dev_script", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load scripts/dev.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_commands_cover_beginner_workflow() -> None:
    """The helper script should expose the small command set documented in the README."""
    module = _load_dev_module()

    assert module.COMMANDS["format"] == [["ruff", "format", "."]]
    assert module.COMMANDS["check"] == [["ruff", "check", "."]]
    assert module.COMMANDS["test"] == [["pytest", "-q"]]
    assert module.COMMANDS["all"] == [
        ["ruff", "format", "."],
        ["ruff", "check", "."],
        ["pytest", "-q"],
    ]


def test_parser_accepts_documented_command_names() -> None:
    """The parser should accept the simple command names shown to the user."""
    module = _load_dev_module()
    parser = module.build_parser()

    parsed = parser.parse_args(["all"])

    assert parsed.command == "all"
