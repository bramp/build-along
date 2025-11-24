"""Download command for LEGO instruction manuals."""

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pytimeparse2

from build_a_long.downloader.downloader import LegoInstructionDownloader
from build_a_long.downloader.legocom import LEGO_BASE, build_metadata
from build_a_long.downloader.util import is_valid_set_id


def get_set_numbers_from_args(args: argparse.Namespace) -> list[str]:
    """Extract and validate LEGO set numbers from command-line arguments.

    Args:
        args: Parsed command-line arguments containing either set_number or stdin flag.

    Returns:
        List of validated LEGO set IDs (all numeric strings).
        Invalid IDs are skipped with a warning to stderr.
    """
    set_numbers = []
    if args.set_number:
        candidate = str(args.set_number).strip()
        if is_valid_set_id(candidate):
            set_numbers.append(candidate)
        else:
            print(
                f"Warning: Invalid set ID '{candidate}' from argument. Skipping.",
                file=sys.stderr,
            )
    elif args.stdin:
        for line in sys.stdin:
            candidate = line.strip()
            if is_valid_set_id(candidate):
                set_numbers.append(candidate)
            else:
                print(
                    f"Warning: Invalid set ID '{candidate}' from stdin. Skipping.",
                    file=sys.stderr,
                )
    return [s for s in set_numbers if s]


def add_download_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add download subcommand parser.

    Args:
        subparsers: The subparsers action from argparse.
    """
    download_parser = subparsers.add_parser(
        "download", help="Download LEGO instruction manuals."
    )
    input_group = download_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "set_number",
        nargs="?",  # Make it optional
        help="The LEGO set number, e.g. 75419 (for single set download)",
    )
    input_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read LEGO set numbers from standard input (one per line)",
    )
    download_parser.add_argument(
        "--locale",
        default="en-us",
        help="LEGO locale to use, e.g. en-us, en-gb, de-de",
    )
    download_parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to store PDFs. Defaults to data/<set_number>",
    )
    download_parser.add_argument(
        "--print-metadata",
        action="store_true",
        help="Only fetch and print metadata as JSON (no downloads or saving)",
    )
    download_parser.add_argument(
        "--skip-pdfs",
        action="store_true",
        help="Download and save metadata files only, skip PDF downloads",
    )
    download_parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Force metadata update, even if it exists.",
    )
    download_parser.add_argument(
        "--overwrite-metadata-if-older-than",
        default="1d",
        help=(
            "Overwrite metadata if older than a specified duration "
            "(e.g., 1d, 12h, 1m, 30s). Defaults to 1 day."
        ),
    )
    download_parser.add_argument(
        "--overwrite-pdfs",
        action="store_true",
        help="Force re-downloading of PDFs, even if they exist.",
    )
    download_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )


def run_download(args: argparse.Namespace) -> int:
    """Execute the download command.

    Args:
        args: Parsed command-line arguments for the download command.

    Returns:
        Exit code: 0 for success, 1 on error.
    """
    all_set_numbers = get_set_numbers_from_args(args)

    if not all_set_numbers:
        print("Error: No LEGO set numbers provided.", file=sys.stderr)
        return 1

    # Validate conflicting flags
    if args.skip_pdfs and args.overwrite_pdfs:
        print(
            "Error: --skip-pdfs and --overwrite-pdfs cannot be used together.",
            file=sys.stderr,
        )
        return 1

    overwrite_metadata_if_older_than: timedelta | None = None
    if args.overwrite_metadata_if_older_than:
        duration_str = args.overwrite_metadata_if_older_than
        try:
            duration = pytimeparse2.parse(duration_str, as_timedelta=True)
            if not isinstance(duration, timedelta):
                raise ValueError("Invalid duration string format")
            overwrite_metadata_if_older_than = duration
        except (ValueError, TypeError):
            print(
                f"Error: Invalid duration string: {duration_str}",
                file=sys.stderr,
            )
            return 1

    if args.overwrite_metadata:
        overwrite_metadata_if_older_than = timedelta(seconds=0)

    # Print metadata mode: fetch and print JSON without downloading or saving
    if args.print_metadata:
        with LegoInstructionDownloader(locale=args.locale) as downloader:
            for set_number in all_set_numbers:
                try:
                    html = downloader.fetch_instructions_page(set_number)
                    meta = build_metadata(
                        html,
                        set_number,
                        args.locale,
                        base=LEGO_BASE,
                        debug=args.debug,
                    )
                    print(meta.model_dump_json(indent=2, exclude_unset=True))
                except Exception as e:
                    print(
                        f"Error fetching metadata for set {set_number}: {e}",
                        file=sys.stderr,
                    )
                    return 1
        return 0

    # Download mode: use the class-based downloader with shared state
    with LegoInstructionDownloader(
        locale=args.locale,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        overwrite_metadata_if_older_than=overwrite_metadata_if_older_than,
        overwrite_download=args.overwrite_pdfs,
        show_progress=True,
        debug=args.debug,
        skip_pdfs=args.skip_pdfs,
    ) as downloader:
        exit_code = downloader.process_sets(all_set_numbers)

    print("All done.")
    return exit_code
