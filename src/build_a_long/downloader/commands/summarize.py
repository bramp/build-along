"""Summarize command for LEGO instruction metadata.

This module wraps the existing summarize_metadata functionality.
"""

import argparse
from pathlib import Path

from build_a_long.downloader.summarize_metadata import (
    summarize_metadata as _summarize_metadata,
)


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
        default="data",
        help="Directory containing the downloaded LEGO set data.",
    )
    summarize_parser.add_argument(
        "--output-dir",
        default="data/indices",
        help="Directory to store the generated index files.",
    )


def run_summarize(args: argparse.Namespace) -> int:
    """Execute the summarize command.

    Args:
        args: Parsed command-line arguments for the summarize command.

    Returns:
        Exit code from summarize_metadata.
    """
    return _summarize_metadata(Path(args.data_dir), Path(args.output_dir))
