import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import pymupdf

from build_a_long.pdf_extract.extractor import (
    extract_bounding_boxes,
    PageData,
    ExtractionResult,
)
from build_a_long.pdf_extract.classifier import classify_pages
from build_a_long.pdf_extract.classifier.types import ClassificationResult
from build_a_long.pdf_extract.drawing import draw_and_save_bboxes
from build_a_long.pdf_extract.parser import parse_page_ranges
from build_a_long.pdf_extract.parser.page_ranges import PageRanges

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def save_classified_json(
    pages: List[PageData],
    results: List[ClassificationResult],
    output_dir: Path,
    pdf_path: Path,
) -> None:
    """Save extracted data as JSON file.

    Args:
        pages: List of PageData to serialize
        results: List of ClassificationResult (one per page) - not currently serialized
        output_dir: Directory where JSON should be saved
        pdf_path: Original PDF path (used for naming the JSON file)
    """
    json_data = ExtractionResult(pages=pages).to_dict()
    output_json_path = output_dir / (pdf_path.stem + ".json")
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    logger.info("Saved JSON to %s", output_json_path)


def save_raw_json(pages: List[PageData], output_dir: Path, pdf_path: Path) -> None:
    """Save extracted raw data as JSON files, one per page.

    Args:
        pages: List of PageData to serialize
        output_dir: Directory where JSON should be saved
        pdf_path: Original PDF path (used for naming the JSON file)
    """

    def _prune_element_metadata(page: Dict[str, Any]) -> Dict[str, Any]:
        """Prune noisy/empty fields from PageData dict in-place (non-recursive).

        Applies to each element in page["elements"] only:
        - Drop "deleted" when falsy (e.g., False)
        - Drop "label" when None
        """
        elements = page.get("elements", [])
        if isinstance(elements, list):
            for ele in elements:
                if not isinstance(ele, dict):
                    continue
                if ("deleted" in ele) and (not bool(ele.get("deleted"))):
                    del ele["deleted"]
                if ("label" in ele) and (ele.get("label") is None):
                    del ele["label"]
        return page

    for page_data in pages:
        json_page: Dict[str, Any] = page_data.to_dict()
        json_page = _prune_element_metadata(json_page)

        output_json_path = output_dir / (
            f"{pdf_path.stem}_page_{page_data.page_number:03d}_raw.json"
        )
        with open(output_json_path, "w") as f:
            json.dump(json_page, f, indent=4)
        logger.info(
            "Saved raw JSON for page %d to %s",
            page_data.page_number,
            output_json_path,
        )


def render_annotated_images(
    doc: pymupdf.Document,
    pages: List[PageData],
    results: List[ClassificationResult],
    output_dir: Path,
    *,
    draw_deleted: bool = False,
) -> None:
    """Render PDF pages with annotated bounding boxes as PNG images.

    Args:
        doc: The open PyMuPDF Document
        pages: List of PageData containing extracted elements
        results: List of ClassificationResult with labels for elements
        output_dir: Directory where PNG images should be saved
        draw_deleted: If True, also render elements marked as deleted.
    """
    for page_data, result in zip(pages, results):
        page_num = page_data.page_number  # 1-indexed
        page = doc[page_num - 1]  # 0-indexed
        output_path = output_dir / f"page_{page_num:03d}.png"
        draw_and_save_bboxes(
            page, page_data.elements, result, output_path, draw_deleted=draw_deleted
        )


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        Parsed arguments namespace
    """
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
        help=(
            "Pages to process (1-indexed). Accepts single pages and ranges, "
            "optionally comma-separated. Examples: '5', '5-10', '10-' (from 10 to end), "
            "'-5' (from 1 to 5), or '10-20,180' for multiple segments. Defaults to all pages."
        ),
    )
    parser.add_argument(
        "--include-types",
        type=str,
        help="Comma-separated list of element types to include. Defaults to all types.",
        default="text,image,drawing",
    )
    # Summary controls
    parser.add_argument(
        "--no-summary",
        dest="summary",
        action="store_false",
        help="Do not print a classification summary to stdout.",
    )
    parser.add_argument(
        "--summary-detailed",
        action="store_true",
        help="Print a slightly more detailed summary, including pages missing page numbers.",
    )
    parser.add_argument(
        "--debug-json",
        action="store_true",
        help="Export raw page elements as a JSON document for debugging.",
    )
    parser.add_argument(
        "--draw-deleted",
        action="store_true",
        help="Draw bounding boxes for elements marked as deleted.",
    )
    parser.set_defaults(summary=True, summary_detailed=False, draw_deleted=False)
    return parser.parse_args()


def _print_summary(
    pages: List[PageData],
    results: List["ClassificationResult"],
    *,
    detailed: bool = False,
) -> None:
    """Print a human-readable summary of classification results to stdout."""
    total_pages = len(pages)
    total_elements = 0
    elements_by_type: Dict[str, int] = {}
    labeled_counts: Dict[str, int] = {}

    pages_with_page_number = 0
    missing_page_numbers: List[int] = []

    for page, result in zip(pages, results):
        total_elements += len(page.elements)
        # Tally element types and labels
        has_page_number = False
        for ele in page.elements:
            t = ele.__class__.__name__.lower()
            elements_by_type[t] = elements_by_type.get(t, 0) + 1

            label = result.get_label(ele)
            if label:
                labeled_counts[label] = labeled_counts.get(label, 0) + 1
                if label == "page_number":
                    has_page_number = True

        if has_page_number:
            pages_with_page_number += 1
        else:
            missing_page_numbers.append(page.page_number)

    coverage = (pages_with_page_number / total_pages * 100.0) if total_pages else 0.0

    # Human-friendly, single-shot summary
    print("=== Classification summary ===")
    print(f"Pages processed: {total_pages}")
    print(f"Total elements: {total_elements}")
    if elements_by_type:
        parts = [f"{k}={v}" for k, v in sorted(elements_by_type.items())]
        print("Elements by type: " + ", ".join(parts))
    if labeled_counts:
        parts = [f"{k}={v}" for k, v in sorted(labeled_counts.items())]
        print("Labeled elements: " + ", ".join(parts))
    print(
        f"Page-number coverage: {pages_with_page_number}/{total_pages} ({coverage:.1f}%)"
    )
    if detailed and missing_page_numbers:
        sample = ", ".join(str(n) for n in missing_page_numbers[:20])
        more = " ..." if len(missing_page_numbers) > 20 else ""
        print(f"Pages missing page number: {sample}{more}")


def _print_label_counts(page: PageData, result: ClassificationResult) -> None:
    label_counts = defaultdict(int)
    for e in page.elements:
        label = result.get_label(e) or "<unknown>"
        label_counts[label] += 1

    # TODO The following logging shows "defaultdict(<class 'int'>,..." figure
    # out how to avoid that.
    logger.info(f"Page {page.page_number} Label counts: {label_counts}")


def main() -> int:
    """Main entry point for the bounding box extractor CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_arguments()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        return 2

    # Default output directory to same directory as PDF
    output_dir = args.output_dir if args.output_dir else pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    include_types = args.include_types.split(",")

    logging.info("Processing PDF: %s", pdf_path)

    # Compute page selection before opening the document. If unbounded (no
    # --pages), pass None to extractor so it processes all pages.
    page_ranges = PageRanges.all()
    page_numbers_for_extractor: List[int] | None = None
    if args.pages:
        try:
            page_ranges = parse_page_ranges(args.pages)
        except ValueError as e:
            logger.error("Invalid --pages: %s", e)
            return 2

    # Extract bounding box data from PDF (open document once and reuse)
    with pymupdf.open(str(pdf_path)) as doc:
        logger.info("Selected pages: %s", page_ranges)
        page_numbers_for_extractor = list(page_ranges.page_numbers(len(doc)))

        pages: List[PageData] = extract_bounding_boxes(
            doc, page_numbers_for_extractor, include_types=include_types
        )

        # Save raw extracted data as JSON if debug flag is set
        # The deleted field will be included in the JSON output
        if args.debug_json:
            save_raw_json(pages, output_dir, pdf_path)

        # Classify elements to add labels (e.g., page numbers)
        # This also marks elements as deleted if they're duplicates/shadows
        # Returns classification results with labels for each page
        results = classify_pages(pages)

        for page, result in zip(pages, results):
            _print_label_counts(page, result)

        # Optionally print a concise summary to stdout
        if args.summary:
            _print_summary(pages, results, detailed=args.summary_detailed)

        # Save results as JSON and render annotated images
        save_classified_json(pages, results, output_dir, pdf_path)
        render_annotated_images(
            doc, pages, results, output_dir, draw_deleted=args.draw_deleted
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
