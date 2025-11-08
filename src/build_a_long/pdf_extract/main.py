import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pymupdf

from build_a_long.pdf_extract.classifier import classify_pages
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.lego_page_builder import build_page
from build_a_long.pdf_extract.classifier.text_histogram import TextHistogram
from build_a_long.pdf_extract.drawing import draw_and_save_bboxes
from build_a_long.pdf_extract.extractor import (
    ExtractionResult,
    PageData,
    extract_bounding_boxes,
)
from build_a_long.pdf_extract.extractor.hierarchy import build_hierarchy_from_blocks
from build_a_long.pdf_extract.extractor.lego_page_elements import Page
from build_a_long.pdf_extract.parser import parse_page_ranges
from build_a_long.pdf_extract.parser.page_ranges import PageRanges

# Logging will be configured in main() based on command-line arguments
logger = logging.getLogger(__name__)


def save_classified_json(
    pages: list[PageData],
    results: list[ClassificationResult],
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


def save_raw_json(pages: list[PageData], output_dir: Path, pdf_path: Path) -> None:
    """Save extracted raw data as JSON files, one per page.

    Args:
        pages: List of PageData to serialize
        output_dir: Directory where JSON should be saved
        pdf_path: Original PDF path (used for naming the JSON file)
    """

    def _prune_element_metadata(page: dict[str, Any]) -> dict[str, Any]:
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
        json_page: dict[str, Any] = page_data.to_dict()
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
    pages: list[PageData],
    results: list[ClassificationResult],
    output_dir: Path,
    *,
    draw_deleted: bool = False,
) -> None:
    """Render PDF pages with annotated bounding boxes as PNG images.

    Args:
        doc: The open PyMuPDF Document
        pages: List of PageData containing extracted elements
        results: List of ClassificationResult with labels and elements
        output_dir: Directory where PNG images should be saved
        draw_deleted: If True, also render elements marked as deleted.
    """
    for page_data, result in zip(pages, results, strict=True):
        page_num = page_data.page_number  # 1-indexed
        page = doc[page_num - 1]  # 0-indexed
        output_path = output_dir / f"page_{page_num:03d}.png"
        draw_and_save_bboxes(page, result, output_path, draw_deleted=draw_deleted)


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
        help=(
            "Comma-separated list of block types to decode from PDF. "
            "Valid types: text, image, drawing. "
            "Examples: 'text', 'text,image', or 'text,image,drawing' (default: all types)."
        ),
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
        "--debug-classification",
        action="store_true",
        help="Print detailed classification debugging information for each page.",
    )
    parser.add_argument(
        "--print-histogram",
        action="store_true",
        help="Print text histogram showing font size and name distributions.",
    )
    parser.add_argument(
        "--draw-deleted",
        action="store_true",
        help="Draw bounding boxes for elements marked as deleted.",
    )
    parser.add_argument(
        "--no-draw",
        dest="draw",
        action="store_false",
        help="Do not draw annotated images.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO).",
    )
    parser.set_defaults(
        summary=True, summary_detailed=False, draw_deleted=False, draw=True
    )
    return parser.parse_args()


def _print_summary(
    pages: list[PageData],
    results: list[ClassificationResult],
    *,
    detailed: bool = False,
) -> None:
    """Print a human-readable summary of classification results to stdout."""
    total_pages = len(pages)
    total_blocks = 0
    blocks_by_type: dict[str, int] = {}
    labeled_counts: dict[str, int] = {}

    pages_with_page_number = 0
    missing_page_numbers: list[int] = []

    for page, result in zip(pages, results, strict=True):
        total_blocks += len(page.blocks)
        # Tally block types and labels
        has_page_number = False
        for block in page.blocks:
            t = block.__class__.__name__.lower()
            blocks_by_type[t] = blocks_by_type.get(t, 0) + 1

            label = result.get_label(block)
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
    print(f"Total blocks: {total_blocks}")
    if blocks_by_type:
        parts = [f"{k}={v}" for k, v in sorted(blocks_by_type.items())]
        print("Elements by type: " + ", ".join(parts))
    if labeled_counts:
        parts = [f"{k}={v}" for k, v in sorted(labeled_counts.items())]
        print("Labeled elements: " + ", ".join(parts))
    print(
        f"Page-number coverage: {pages_with_page_number}/{total_pages} "
        f"({coverage:.1f}%)"
    )
    if detailed and missing_page_numbers:
        sample = ", ".join(str(n) for n in missing_page_numbers[:20])
        more = " ..." if len(missing_page_numbers) > 20 else ""
        print(f"Pages missing page number: {sample}{more}")


def _print_font_size_distribution(
    title: str,
    counter: Any,
    *,
    max_items: int = 10,
    empty_message: str = "(no data)",
    total_label: str = "Total text elements",
    unique_label: str = "Total unique sizes",
) -> None:
    """Print a font size distribution with bar chart.

    Args:
        title: Section title to display
        counter: Counter/dict mapping font sizes to counts
        max_items: Maximum number of items to display
        empty_message: Message to show when counter is empty
        total_label: Label for total count summary
        unique_label: Label for unique size count
    """
    print(title)
    print("-" * 60)

    total = sum(counter.values())

    if total > 0:
        print(f"{'Size':>8} | {'Count':>6} | Distribution")
        print("-" * 60)

        # Get most common items
        if hasattr(counter, "most_common"):
            items = counter.most_common(max_items)
        else:
            items = sorted(counter.items(), key=lambda x: x[1], reverse=True)[
                :max_items
            ]

        max_count = items[0][1] if items else 1
        for size, count in items:
            bar_length = int((count / max_count) * 30)
            bar = "█" * bar_length
            print(f"{size:8.1f} | {count:6d} | {bar}")

        print("-" * 60)
        print(f"{unique_label}: {len(counter)}")
        print(f"{total_label}: {total}")
    else:
        print(empty_message)
    print()


def _print_histogram(histogram: TextHistogram) -> None:
    """Print the text histogram showing font size and name distributions.

    Args:
        histogram: TextHistogram containing font statistics across all pages
    """
    print("=== Text Histogram ===")
    print()

    # 1. Part counts (\dx pattern) - calculated first
    _print_font_size_distribution(
        "1. Part Count Font Sizes (\\dx pattern, e.g., '2x', '3x'):",
        histogram.part_count_font_sizes,
        empty_message="(no part count data)",
        total_label="Total part counts",
    )

    # 2. Page numbers (±1) - calculated second
    _print_font_size_distribution(
        "2. Page Number Font Sizes (digits ±1 from current page):",
        histogram.page_number_font_sizes,
        empty_message="(no page number data)",
        total_label="Total page numbers",
    )

    # 3. Element IDs (6-7 digit numbers) - calculated third
    _print_font_size_distribution(
        "3. Element ID Font Sizes (6-7 digit numbers):",
        histogram.element_id_font_sizes,
        empty_message="(no Element ID data)",
        total_label="Total Element IDs",
    )

    # 4. Other integer font sizes - calculated fourth
    _print_font_size_distribution(
        "4. Other Integer Font Sizes (integers not matching above patterns):",
        histogram.remaining_font_sizes,
        max_items=20,
        empty_message="(no other integer font size data)",
    )

    # 5. Font name distribution - calculated fifth
    print("5. Font Name Distribution:")
    print("-" * 60)

    font_name_total = sum(histogram.font_name_counts.values())

    if font_name_total > 0:
        print(f"{'Font Name':<30} | {'Count':>6} | Distribution")
        print("-" * 60)

        font_names = histogram.font_name_counts.most_common(20)
        max_count = font_names[0][1] if font_names else 1
        for font_name, count in font_names:
            bar_length = int((count / max_count) * 30)
            bar = "█" * bar_length
            name_display = font_name[:27] + "..." if len(font_name) > 30 else font_name
            print(f"{name_display:<30} | {count:6d} | {bar}")

        print("-" * 60)
        print(f"Total unique fonts:  {len(histogram.font_name_counts)}")
        print(f"Total text elements: {font_name_total}")
    else:
        print("(no font name data)")

    print()


def _print_classification_debug(page: PageData, result: ClassificationResult) -> None:
    """Print line-by-line classification status for all elements.

    For each element (ordered hierarchically), shows:
    - Element ID, type, and string representation
    - Winning candidate labels (if any)
    - Removal status and reason (if removed)
    - Indentation based on bbox nesting hierarchy

    Args:
        page: PageData containing all elements
        result: ClassificationResult with classification information
    """
    # ANSI color codes
    GREY = "\033[90m"
    RESET = "\033[0m"

    print(f"\n{'=' * 80}")
    print(f"CLASSIFICATION DEBUG - Page {page.page_number}")
    print(f"{'=' * 80}\n")

    # Build block hierarchy tree
    block_tree = build_hierarchy_from_blocks(page.blocks)

    # Get all winning candidates organized by block
    all_candidates = result.get_all_candidates()
    # block.id -> list of winning labels
    block_to_labels: dict[int, list[str]] = {}

    for label, candidates in all_candidates.items():
        for candidate in candidates:
            if candidate.is_winner and candidate.source_block is not None:
                block_id = candidate.source_block.id
                if block_id is not None:
                    if block_id not in block_to_labels:
                        block_to_labels[block_id] = []
                    block_to_labels[block_id].append(label)

    def print_element(elem, depth: int, is_last: bool = True) -> None:
        """Recursively print a block and its children."""
        # Build tree characters
        if depth == 0:
            tree_prefix = ""
            indent = ""
        else:
            # Use └─ for last child, ├─ for others
            tree_char = "└─" if is_last else "├─"
            # Build the prefix based on depth (spaces for alignment)
            indent = "  " * (depth - 1)
            tree_prefix = f"{indent}{tree_char} "

        # Base info: ID and type
        type_name = elem.__class__.__name__
        elem_id = elem.id if elem.id is not None else -1

        # Check if removed
        is_removed = result.is_removed(elem)
        color = GREY if is_removed else ""
        reset = RESET if is_removed else ""

        # Build the complete line with block details
        # Prefer constructed element string if available
        constructed = result.get_constructed_element(elem)
        elem_str = str(constructed) if constructed else str(elem)
        line = f"{color}{tree_prefix}{elem_id:3d} {type_name:8s} "

        if is_removed:
            reason = result.get_removal_reason(elem)
            reason_text = reason.reason_type if reason else "unknown"
            line += f"[REMOVED: {reason_text}"
            if reason and hasattr(reason, "target_element"):
                target = reason.target_block
                target_id = target.id if hasattr(target, "id") else "?"
                line += f" -> {target_id}"
                # Show what the target block won as
                if (
                    hasattr(target, "id")
                    and target.id is not None
                    and target.id in block_to_labels
                ):
                    target_labels = block_to_labels[target.id]
                    line += f" ({', '.join(target_labels)})"
            line += f"] {elem_str}"
        # Check if it has winning candidates
        elif elem.id is not None and elem.id in block_to_labels:
            labels = block_to_labels[elem.id]
            line += f"[{', '.join(labels)}] {elem_str}"
        else:
            # No candidates
            line += f"[no candidates] {elem_str}"

        line += reset
        print(line)

        # Recursively print children, sorted by ID
        children = block_tree.get_children(elem)
        sorted_children = sorted(
            children, key=lambda e: e.id if e.id is not None else 999999
        )
        for i, child in enumerate(sorted_children):
            child_is_last = i == len(sorted_children) - 1
            print_element(child, depth + 1, child_is_last)

    # Print root blocks sorted by ID, then their children recursively
    sorted_roots = sorted(
        block_tree.roots, key=lambda e: e.id if e.id is not None else 999999
    )
    for root in sorted_roots:
        print_element(root, 0)

    # Print summary statistics
    total = len(page.blocks)
    with_labels = len(block_to_labels)
    removed = sum(1 for e in page.blocks if result.is_removed(e))
    no_candidates = total - with_labels - removed

    print(f"\n{'─' * 80}")
    print(
        f"Total: {total} | Winners: {with_labels} | Removed: {removed} | No candidates: {no_candidates}"
    )

    warnings = result.get_warnings()
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    print(f"{'=' * 80}\n")


def _print_label_counts(page: PageData, result: ClassificationResult) -> None:
    label_counts = defaultdict(int)
    for e in page.blocks:
        label = result.get_label(e) or "<unknown>"
        label_counts[label] += 1

    # TODO The following logging shows "defaultdict(<class 'int'>,..." figure
    # out how to avoid that.
    logger.info(f"Page {page.page_number} Label counts: {label_counts}")


def _print_page_hierarchy(page_data: PageData, page: Page) -> None:
    print(f"Page {page_data.page_number}:")

    if page.page_number:
        print(f"  ✓ Page Number: {page.page_number.value}")

    if page.steps:
        print(f"  ✓ Steps: {len(page.steps)}")
        for step in page.steps:
            parts_count = len(step.parts_list.parts)
            print(f"    - Step {step.step_number.value} ({parts_count} parts)")
            # Print parts list details
            if step.parts_list.parts:
                print("      Parts List:")
                for part in step.parts_list.parts:
                    print(
                        f"        • {part.count.count}x {part.name or 'unnamed'} ({part.number or 'no number'})"
                    )
            else:
                print("      Parts List: (empty)")

            print(f"      Diagram: {step.diagram.bbox}")

    if page.parts_lists:
        print(f"  ✓ Unassigned Parts Lists: {len(page.parts_lists)}")
        for pl in page.parts_lists:
            total_items = sum(p.count.count for p in pl.parts)
            print(f"    - {len(pl.parts)} part types, {total_items} total items")

    if page.warnings:
        print(f"  ⚠ Warnings: {len(page.warnings)}")
        for warning in page.warnings:
            print(f"    - {warning}")

    if page.unprocessed_elements:
        print(f"  ℹ Unprocessed elements: {len(page.unprocessed_elements)}")


def _build_page_hierarchy(
    pages: list[PageData], results: list[ClassificationResult]
) -> None:
    """Build LEGO page hierarchy from classification results and log the structure.

    Args:
        pages: List of PageData containing extracted elements
        results: List of ClassificationResult with labels and relationships
    """
    print("Building LEGO page hierarchy...")

    for page_data, result in zip(pages, results, strict=True):
        page = build_page(page_data, result)
        _print_page_hierarchy(page_data, page)


def main() -> int:
    """Main entry point for the bounding box extractor CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_arguments()

    # Configure logging based on command-line argument
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        return 2

    # Default output directory to same directory as PDF
    output_dir = args.output_dir if args.output_dir else pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse block types (validation handled by extract_bounding_boxes)
    include_types = set(t.strip() for t in args.include_types.split(","))

    logging.info("Processing PDF: %s", pdf_path)

    # Compute page selection before opening the document. If unbounded (no
    # --pages), pass None to extractor so it processes all pages.
    page_ranges = PageRanges.all()
    page_numbers_for_extractor: list[int] | None = None
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

        pages: list[PageData] = extract_bounding_boxes(
            doc, page_numbers_for_extractor, include_types=include_types
        )

        # Save raw extracted data as JSON if debug flag is set
        # The deleted field will be included in the JSON output
        if args.debug_json:
            save_raw_json(pages, output_dir, pdf_path)

        # Classify elements to add labels (e.g., page numbers)
        # This also marks elements as deleted if they're duplicates/shadows
        # Returns batch classification result with per-page results and histogram
        batch_result = classify_pages(pages)

        # Optionally print the text histogram
        if args.print_histogram:
            _print_histogram(batch_result.histogram)

        if args.debug_classification:
            for page, result in zip(pages, batch_result.results, strict=True):
                _print_label_counts(page, result)
                _print_classification_debug(page, result)

            # Build structured LEGO page hierarchy from classification results
            _build_page_hierarchy(pages, batch_result.results)

        # Optionally print a concise summary to stdout
        if args.summary:
            _print_summary(pages, batch_result.results, detailed=args.summary_detailed)

        # Save results as JSON and render annotated images
        save_classified_json(pages, batch_result.results, output_dir, pdf_path)
        if args.draw:
            render_annotated_images(
                doc,
                pages,
                batch_result.results,
                output_dir,
                draw_deleted=args.draw_deleted,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
