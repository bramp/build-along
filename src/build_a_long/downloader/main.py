import argparse
import sys
from pathlib import Path

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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the downloader script.

    Returns:
        Parsed arguments with set_number, stdin flag, locale, out_dir,
        dry_run, and force options.
    """
    parser = argparse.ArgumentParser(description="Download LEGO instruction manuals.")

    input_group = parser.add_mutually_exclusive_group(required=True)
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
    parser.add_argument(
        "--locale",
        default="en-us",
        help="LEGO locale to use, e.g. en-us, en-gb, de-de",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to store PDFs. Defaults to data/<set_number>",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Only fetch and print metadata as JSON (no downloads)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download PDFs even if the file already exists",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the LEGO instruction downloader.

    Parses arguments, validates set numbers, and downloads instruction PDFs
    for one or more LEGO sets using the LegoInstructionDownloader class.

    Returns:
        Exit code: 0 for success, 1 if no valid set numbers provided or on error.
    """
    args = _parse_args()
    all_set_numbers = get_set_numbers_from_args(args)

    if not all_set_numbers:
        print("Error: No LEGO set numbers provided.", file=sys.stderr)
        return 1

    # Metadata mode: fetch and print JSON without downloading
    if args.metadata:
        with LegoInstructionDownloader(locale=args.locale) as downloader:
            for set_number in all_set_numbers:
                try:
                    html = downloader.fetch_instructions_page(set_number)
                    meta = build_metadata(html, set_number, args.locale, base=LEGO_BASE)
                    print(meta.model_dump_json(indent=2))
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
        overwrite=args.force,
        show_progress=True,
    ) as downloader:
        exit_code = downloader.process_sets(all_set_numbers)

    print("All done.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
