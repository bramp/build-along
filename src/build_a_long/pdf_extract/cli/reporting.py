"""Reporting and output formatting for PDF extraction."""

import logging
from collections import defaultdict
from typing import Any

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.lego_page_builder import build_page
from build_a_long.pdf_extract.classifier.text_histogram import TextHistogram
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.hierarchy import build_hierarchy_from_blocks
from build_a_long.pdf_extract.extractor.lego_page_elements import Page

logger = logging.getLogger(__name__)

# ANSI color codes
GREY = "\033[90m"
RESET = "\033[0m"


def print_summary(
    pages: list[PageData],
    results: list[ClassificationResult],
    *,
    detailed: bool = False,
) -> None:
    """Print a human-readable summary of classification results to stdout.

    Args:
        pages: List of PageData containing extracted elements
        results: List of ClassificationResult with labels
        detailed: If True, include additional details like missing page numbers
    """
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


def print_histogram(histogram: TextHistogram) -> None:
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


def print_classification_debug(page: PageData, result: ClassificationResult) -> None:
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
        f"Total: {total} | Winners: {with_labels} | "
        f"Removed: {removed} | No candidates: {no_candidates}"
    )

    warnings = result.get_warnings()
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    print(f"{'=' * 80}\n")


def print_label_counts(page: PageData, result: ClassificationResult) -> None:
    """Print label count statistics for a page.

    Args:
        page: PageData containing all elements
        result: ClassificationResult with labels
    """
    label_counts = defaultdict(int)
    for e in page.blocks:
        label = result.get_label(e) or "<unknown>"
        label_counts[label] += 1

    # TODO The following logging shows "defaultdict(<class 'int'>,..." figure
    # out how to avoid that.
    logger.info(f"Page {page.page_number} Label counts: {label_counts}")


def print_page_hierarchy(page_data: PageData, page: Page) -> None:
    """Print the structured LEGO page hierarchy.

    Args:
        page_data: PageData containing the raw page number
        page: Structured Page object with steps, parts lists, etc.
    """
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
                        f"        • {part.count.count}x {part.name or 'unnamed'} "
                        f"({part.number or 'no number'})"
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


def build_and_print_page_hierarchy(
    pages: list[PageData], results: list[ClassificationResult]
) -> None:
    """Build LEGO page hierarchy from classification results and print structure.

    Args:
        pages: List of PageData containing extracted elements
        results: List of ClassificationResult with labels and relationships
    """
    print("Building LEGO page hierarchy...")

    for page_data, result in zip(pages, results, strict=True):
        page = build_page(page_data, result)
        print_page_hierarchy(page_data, page)
