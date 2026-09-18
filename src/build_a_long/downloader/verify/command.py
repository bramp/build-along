"""Verify command for LEGO instruction data integrity.

This module wraps the existing verify functionality.
"""

import argparse
import os
import sys
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
        default=os.environ.get("LEGO_DATA_DIR"),
        help="Directory containing the downloaded LEGO set data (defaults to LEGO_DATA_DIR env var).",
    )


def run_verify(args: argparse.Namespace) -> int:
    """Execute the verify command.

    Args:
        args: Parsed command-line arguments for the verify command.

    Returns:
        Exit code from verify_data_integrity.
    """
    data_dir = args.data_dir or os.environ.get("LEGO_DATA_DIR")
    if not data_dir:
        print(
            "Error: Data directory must be specified via --data-dir or the LEGO_DATA_DIR environment variable.",
            file=sys.stderr,
        )
        return 1
    return _verify_data_integrity(Path(data_dir))
