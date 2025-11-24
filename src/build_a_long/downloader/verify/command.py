"""Verify command for LEGO instruction data integrity.

This module wraps the existing verify functionality.
"""

import argparse
from pathlib import Path

from .verify import verify_data_integrity as _verify_data_integrity


def add_verify_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add verify subcommand parser.

    Args:
        subparsers: The subparsers action from argparse.
    """
    verify_parser = subparsers.add_parser(
        "verify", help="Verify integrity of downloaded files."
    )
    verify_parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the downloaded LEGO set data.",
    )


def run_verify(args: argparse.Namespace) -> int:
    """Execute the verify command.

    Args:
        args: Parsed command-line arguments for the verify command.

    Returns:
        Exit code from verify_data_integrity.
    """
    return _verify_data_integrity(Path(args.data_dir))
