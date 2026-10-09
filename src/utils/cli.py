"""Command-line helpers: every on/off option takes an explicit value, yes or no."""

from __future__ import annotations

import argparse

TRUE = {"yes", "y", "true", "1"}
FALSE = {"no", "n", "false", "0"}


def yes_no(value: str) -> bool:
    v = str(value).strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise argparse.ArgumentTypeError(f"expected yes or no, got '{value}'")


def add_yes_no(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(name, type=yes_no, default=default, metavar="yes|no",
                        help=f"{help_text} (default: {'yes' if default else 'no'})")
