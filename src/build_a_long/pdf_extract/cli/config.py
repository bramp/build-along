"""CLI configuration and argument parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessingConfig:
    """Configuration for PDF processing."""

    pdf_paths: list[Path]
    output_dir: Path | None
    include_types: set[str]
    page_ranges: str | None = None

    # Output flags
    save_summary: bool = True
    summary_detailed: bool = False
    save_debug_json: bool = False
    compress_json: bool = False
    draw_blocks: bool = False
    draw_elements: bool = False
    draw_deleted: bool = False

    # Debug flags
    debug_classification: bool = False
    debug_candidates: bool = False
    debug_candidates_label: str | None = None
    print_histogram: bool = False
    print_font_hints: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ProcessingConfig:
        """Create config from parsed arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            ProcessingConfig instance
        """
        pdf_paths = [Path(p) for p in args.pdf_paths]
        include_types = set(t.strip() for t in args.include_types.split(","))

        return cls(
            pdf_paths=pdf_paths,
            output_dir=args.output_dir,
            include_types=include_types,
            page_ranges=args.pages,
            save_summary=args.summary,
            summary_detailed=args.summary_detailed,
            save_debug_json=args.debug_json,
            compress_json=args.compress_json,
            draw_blocks=args.draw_blocks,
            draw_elements=args.draw_elements,
            draw_deleted=args.draw_deleted,
            debug_classification=args.debug_classification,
            debug_candidates=args.debug_candidates,
            debug_candidates_label=args.debug_candidates_label,
            print_histogram=args.print_histogram,
            print_font_hints=args.print_font_hints,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract bounding boxes from PDF files and export images/JSON for "
            "debugging."
        ),
        allow_abbrev=False,
    )

    # Basic arguments
    parser.add_argument(
        "pdf_paths",
        nargs="+",
        help="Path(s) to one or more PDF files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save images and JSON files. Defaults to same directory as "
        "PDF.",
    )
    parser.add_argument(
        "--pages",
        type=str,
        help=(
            "Pages to process (1-indexed). Accepts single pages and ranges, "
            "optionally comma-separated. Examples: '5', '5-10', '10-' (from 10 to "
            "end), '-5' (from 1 to 5), or '10-20,180' for multiple segments. "
            "(default: all)"
        ),
        default="all",
    )
    parser.add_argument(
        "--include-types",
        type=str,
        help=(
            "Comma-separated list of block types to decode from PDF. "
            "Valid types: text, image, drawing. "
            "Examples: 'text', 'text,image', or 'text,image,drawing' "
            "(default: text,image,drawing)."
        ),
        default="text,image,drawing",
    )

    # Output options group
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a classification summary to stdout.",
    )
    output_group.add_argument(
        "--summary-detailed",
        action="store_true",
        help=(
            "Print a slightly more detailed summary, "
            "including pages missing page numbers."
        ),
    )
    output_group.add_argument(
        "--draw-blocks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw bounding boxes for classified PDF blocks in annotated images.",
    )
    output_group.add_argument(
        "--draw-elements",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Draw bounding boxes for classified LEGO page elements in annotated images."
        ),
    )
    output_group.add_argument(
        "--draw-deleted",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Draw bounding boxes for elements marked as deleted "
            "(requires --draw-blocks or --draw-elements)."
        ),
    )

    # Debug options group
    debug_group = parser.add_argument_group("debug options")
    debug_group.add_argument(
        "--debug-json",
        action="store_true",
        help="Export debug JSON files: raw page data and classification details (candidates, scores, removal reasons).",
    )
    debug_group.add_argument(
        "--compress-json",
        action="store_true",
        help="Enable bz2 compression for JSON output (default: uncompressed).",
    )
    debug_group.add_argument(
        "--debug-classification",
        action="store_true",
        help="Print detailed classification debugging information for each page.",
    )
    debug_group.add_argument(
        "--debug-candidates",
        action="store_true",
        help=(
            "Print detailed analysis of classification candidates, including "
            "winners, scores, and constructed elements. Useful for debugging "
            "classification issues."
        ),
    )
    debug_group.add_argument(
        "--debug-candidates-label",
        type=str,
        help=(
            "Limit --debug-candidates output to a specific label "
            "(e.g., 'step_number', 'page_number'). If not specified, shows all labels."
        ),
    )
    debug_group.add_argument(
        "--print-histogram",
        action="store_true",
        help="Print text histogram showing font size and name distributions.",
    )
    debug_group.add_argument(
        "--print-font-hints",
        action="store_true",
        help="Print font size hints derived from text analysis.",
    )
    debug_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO).",
    )

    return parser.parse_args()
