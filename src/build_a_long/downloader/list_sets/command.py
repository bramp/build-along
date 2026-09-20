"""Command to list LEGO set IDs from LEGO.com sitemap or Rebrickable."""

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytimeparse2

from build_a_long.downloader.downloader import LegoInstructionDownloader


def add_list_sets_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add list-sets subcommand parser.

    Args:
        subparsers: The subparsers action from argparse.
    """
    list_parser = subparsers.add_parser(
        "list-sets",
        aliases=["list"],
        help="List LEGO set numbers from LEGO.com product sitemap or Rebrickable.",
    )
    list_parser.add_argument(
        "--source",
        choices=["lego", "rebrickable"],
        default="lego",
        help="Source to retrieve set list from (lego or rebrickable). Defaults to lego.",
    )
    list_parser.add_argument(
        "--locale",
        default="en-us",
        help="LEGO locale for sitemap (e.g. en-us, en-gb). Defaults to en-us.",
    )
    list_parser.add_argument(
        "--data-dir",
        default=os.environ.get("LEGO_DATA_DIR"),
        help="Base data directory (used for cache storage). Defaults to LEGO_DATA_DIR env var.",
    )
    list_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-fetching from source without using cache.",
    )
    list_parser.add_argument(
        "--cache-ttl",
        default="1d",
        help="Cache validity duration (e.g. 1d, 12h, 1w). Defaults to 1 day.",
    )
    list_parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Minimum release year filter (supported for rebrickable source).",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit output to first N sets.",
    )
    list_parser.add_argument(
        "--count",
        action="store_true",
        help="Only print total count of sets instead of listing them.",
    )
    list_parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="Maximum HTTP requests per minute. Defaults to 60.",
    )
    list_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )


def run_list_sets(args: argparse.Namespace) -> int:
    """Execute the list-sets command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 for success, 1 on error.
    """
    cache_ttl: timedelta | None = None
    if not args.no_cache and args.cache_ttl:
        try:
            duration = pytimeparse2.parse(args.cache_ttl, as_timedelta=True)
            if not isinstance(duration, timedelta):
                raise ValueError("Invalid duration string")
            cache_ttl = duration
        except (ValueError, TypeError):
            print(f"Error: Invalid duration string for --cache-ttl: {args.cache_ttl}", file=sys.stderr)
            return 1

    data_dir_path = Path(args.data_dir) if args.data_dir else None

    with LegoInstructionDownloader(
        locale=args.locale,
        data_dir=data_dir_path,
        max_calls=args.rate_limit,
        period=60,
        debug=args.debug,
    ) as downloader:
        try:
            sets = downloader.get_set_list(
                source=args.source,
                use_cache=not args.no_cache,
                cache_ttl=cache_ttl,
                min_year=args.min_year,
            )
        except Exception as e:
            print(f"Error retrieving set list from {args.source}: {e}", file=sys.stderr)
            return 1

    if args.limit is not None and args.limit > 0:
        sets = sets[: args.limit]

    if args.count:
        print(len(sets))
    else:
        for set_id in sets:
            print(set_id)

    return 0
