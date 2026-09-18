"""Summarize command for LEGO instruction metadata.

This module wraps the existing summarize_metadata functionality.
"""

import argparse
import os
import sys
from pathlib import Path

from .summarize_metadata import summarize_metadata as _summarize_metadata


def add_summarize_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add summarize subcommand parser.

    Args:
        subparsers: The subparsers action from argparse.
    """
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize downloaded metadata."
    )
    summarize_parser.add_argument(
        "--data-dir",
        default=os.environ.get("LEGO_DATA_DIR"),
        help="Directory containing the downloaded LEGO set data (defaults to LEGO_DATA_DIR env var).",
    )
    summarize_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store the generated index files (defaults to <data-dir>/indices).",
    )


def run_summarize(args: argparse.Namespace) -> int:
    """Execute the summarize command.

    Args:
        args: Parsed command-line arguments for the summarize command.

    Returns:
        Exit code from summarize_metadata.
    """
    data_dir = args.data_dir or os.environ.get("LEGO_DATA_DIR")
    if not data_dir:
        print(
            "Error: Data directory must be specified via --data-dir or the LEGO_DATA_DIR environment variable.",
            file=sys.stderr,
        )
        return 1
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(data_dir) / "indices"
    )
    return _summarize_metadata(Path(data_dir), output_dir)
