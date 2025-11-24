"""Main entry point for the LEGO instruction downloader CLI."""

import argparse
import sys

from build_a_long.downloader.commands.download import (
    add_download_parser,
    run_download,
)
from build_a_long.downloader.commands.summarize import (
    add_summarize_parser,
    run_summarize,
)
from build_a_long.downloader.commands.verify import add_verify_parser, run_verify


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the downloader script.

    Returns:
        Parsed arguments with command and command-specific options.
    """
    parser = argparse.ArgumentParser(description="Manage LEGO instruction manuals.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add subcommand parsers
    add_download_parser(subparsers)
    add_summarize_parser(subparsers)
    add_verify_parser(subparsers)

    return parser.parse_args()


def main() -> int:
    """Main entry point for the LEGO instruction downloader.

    Routes to the appropriate subcommand handler.

    Returns:
        Exit code: 0 for success, non-zero on error.
    """
    args = _parse_args()

    if args.command == "download":
        return run_download(args)
    elif args.command == "summarize":
        return run_summarize(args)
    elif args.command == "verify":
        return run_verify(args)
    else:
        # This case should not be reached if subparsers are configured correctly
        print(
            "Error: No command specified. Use 'download', 'summarize', or 'verify'.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
