import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from build_a_long.bounding_box_extractor.extractor.extractor import (
    extract_bounding_boxes,
)
from build_a_long.bounding_box_extractor.parser.parser import parse_page_range


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract bounding boxes from a PDF file and export images/JSON for debugging."
    )
    parser.add_argument("pdf_path", help="The path to the PDF file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save images and JSON files. Defaults to same directory as PDF.",
    )
    parser.add_argument(
        "--pages",
        type=str,
        help='Page range to process (1-indexed), e.g., "5", "5-10", "10-" (from 10 to end), or "-5" (from 1 to 5). Defaults to all pages.',
    )
    parser.add_argument(
        "--include-types",
        type=str,
        help="Comma-separated list of element types to include (e.g., text,image,drawing,path). Defaults to all types.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return 2

    # Default output directory to same directory as PDF
    output_dir = args.output_dir if args.output_dir else pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse page range
    start_page = None
    end_page = None
    if args.pages:
        try:
            start_page, end_page = parse_page_range(args.pages)
        except ValueError as e:
            print(f"Error: {e}")
            return 2

    include_types = args.include_types.split(",") if args.include_types else None

    extract_bounding_boxes(
        str(pdf_path), output_dir, start_page, end_page, include_types=include_types
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
