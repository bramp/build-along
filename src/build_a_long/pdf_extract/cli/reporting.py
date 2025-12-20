"""Reporting and output formatting for PDF extraction."""

import logging
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from build_a_long.pdf_extract.classifier import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.text import FontSizeHints, TextHistogram
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.hierarchy import build_hierarchy_from_blocks
from build_a_long.pdf_extract.extractor.lego_page_elements import Page
from build_a_long.pdf_extract.extractor.page_blocks import Blocks

logger = logging.getLogger(__name__)

# ANSI color codes
GREY = "\033[90m"
RESET = "\033[0m"


class TreeNode(BaseModel):
    """Unified node for the classification debug tree.

    Represents either a Block with optional candidates, or a synthetic Candidate.
    """

    model_config = ConfigDict(frozen=True)

    bbox: BBox
    """Bounding box for this node"""

    block: Blocks | None = None
    """The source block, if this node represents a block"""

    candidates: list[Candidate] = Field(default_factory=list)
    """Candidates for this block (empty if synthetic or no candidates)"""

    synthetic_candidate: Candidate | None = None
    """If this is a synthetic candidate (no source blocks)"""


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

    # Human-friendly, single-shot summary
    print("=== Classification summary ===")
    print(f"    Pages processed: {total_pages}")
    print(f"    Total blocks: {total_blocks}")
    if blocks_by_type:
        parts = [f"{k}={v}" for k, v in sorted(blocks_by_type.items())]
        print("Elements by type: " + ", ".join(parts))
    if labeled_counts:
        parts = [f"{k}={v}" for k, v in sorted(labeled_counts.items())]
        print("    Labeled elements: " + ", ".join(parts))

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


def print_font_hints(hints: FontSizeHints) -> None:
    """Print font size hints extracted from the document.

    Args:
        hints: FontSizeHints containing identified font sizes for different elements
    """
    print("=== Font Size Hints ===")
    print()

    def format_size(size: float | None) -> str:
        """Format a font size for display."""
        return f"{size:.1f}pt" if size is not None else "N/A"

    print("Identified font sizes:")
    print(f"  Part count size:         {format_size(hints.part_count_size)}")
    print(f"  Catalog part count size: {format_size(hints.catalog_part_count_size)}")
    print(f"  Step number size:        {format_size(hints.step_number_size)}")
    print(f"  Step repeat size:        {format_size(hints.step_repeat_size)}")
    print(f"  Catalog element ID size: {format_size(hints.catalog_element_id_size)}")
    print(f"  Page number size:        {format_size(hints.page_number_size)}")

    print()
    print("Remaining font sizes after removing known patterns:")
    if hints.remaining_font_sizes:
        print(f"{'Size':>8} | {'Count':>6}")
        print("-" * 20)
        for size, count in hints.remaining_font_sizes:
            print(f"{size:8.1f} | {count:6d}")
        print(f"\nTotal unique sizes: {len(hints.remaining_font_sizes)}")
    else:
        print("  (no remaining font sizes)")
    print()


def print_classification_debug(
    page: PageData,
    result: ClassificationResult,
    *,
    show_candidates: bool = True,
    show_hierarchy: bool = True,
    label: str | None = None,
) -> None:
    """Print comprehensive classification debug information.

    Shows all classification details in one consolidated view with blocks
    and candidates intermixed in a unified tree.

    Args:
        page: PageData containing all elements
        result: ClassificationResult with classification information
        show_candidates: Include detailed candidate breakdown
        show_hierarchy: Include page hierarchy summary
        label: If provided, filter candidate analysis to this label only
    """
    print(f"\n{'=' * 80}")
    print(f"CLASSIFICATION DEBUG - Page {page.page_number}")
    print(f"{'=' * 80}\n")

    # Build mapping from blocks to their candidates
    block_to_candidates: dict[int, list[Candidate]] = {}
    all_candidates = result.get_all_candidates()

    for _label_name, candidates in all_candidates.items():
        for candidate in candidates:
            for source_block in candidate.source_blocks:
                if source_block.id not in block_to_candidates:
                    block_to_candidates[source_block.id] = []
                block_to_candidates[source_block.id].append(candidate)

    # Create TreeNode for each block
    block_nodes: list[TreeNode] = []
    block_node_map: dict[int, TreeNode] = {}  # block.id -> TreeNode

    for block in page.blocks:
        candidates_list = block_to_candidates.get(block.id, [])
        node = TreeNode(
            bbox=block.bbox,
            block=block,
            candidates=candidates_list,
        )
        block_nodes.append(node)
        block_node_map[block.id] = node

    # Create TreeNode for synthetic candidates
    synthetic_nodes: list[TreeNode] = []
    for _label_name, candidates in all_candidates.items():
        for candidate in candidates:
            if not candidate.source_blocks:
                node = TreeNode(
                    bbox=candidate.bbox,
                    synthetic_candidate=candidate,
                )
                synthetic_nodes.append(node)

    # Combine all nodes for hierarchy building
    all_nodes = block_nodes + synthetic_nodes

    # Build unified hierarchy
    tree = build_hierarchy_from_blocks(all_nodes)

    def print_node(node: TreeNode, depth: int, is_last: bool = True) -> None:
        """Recursively print a node and its children."""
        # Build tree characters
        if depth == 0:
            tree_prefix = ""
            indent = ""
        else:
            tree_char = "└─" if is_last else "├─"
            indent = "  " * (depth - 1)
            tree_prefix = f"{indent}{tree_char} "

        # Check if this is a removed block
        is_removed = node.block and result.is_removed(node.block)
        color = GREY if is_removed else ""
        reset = RESET if is_removed else ""

        line = f"{color}{tree_prefix}"

        if node.synthetic_candidate:
            # Synthetic candidate (Page, Step, OpenBag, etc.)
            candidate = node.synthetic_candidate
            elem_str = (
                str(candidate.constructed)
                if candidate.constructed
                else "NOT CONSTRUCTED"
            )
            line += f"[{candidate.label}] {elem_str} (score={candidate.score:.3f})"
        elif node.block:
            # Regular block
            block = node.block
            block_type = type(block).__name__
            line += f"{block.id:3d} ({block_type}) "

            if is_removed:
                reason = result.get_removal_reason(block)
                reason_text = reason.reason_type if reason else "unknown"
                line += f"* REMOVED: {reason_text}"
                if reason:
                    target = reason.target_block
                    if target:
                        line += f" by {target.id}"

                        target_best = result.get_best_candidate(target)
                        if target_best:
                            line += f" ({target_best.label})"
                line += f"* {str(block)}"
            elif node.candidates:
                # Show all candidates for this block
                sorted_candidates = sorted(
                    node.candidates, key=lambda c: c.score, reverse=True
                )
                best = sorted_candidates[0]

                # Show best candidate prominently
                elem_str = str(best.constructed) if best.constructed else str(block)
                line += f"[{best.label}] {elem_str} (score={best.score:.3f})"

                # If multiple candidates, show others on additional lines
                if len(sorted_candidates) > 1:
                    print(line + reset)
                    for other in sorted_candidates[1:]:
                        other_indent = indent + ("  " if is_last else "│ ")
                        elem_str = (
                            str(other.constructed) if other.constructed else str(block)
                        )
                        alt_line = (
                            f"{color}{other_indent}   alt [{other.label}] "
                            f"{elem_str} (score={other.score:.3f}){reset}"
                        )
                        print(alt_line)
                    line = None  # Already printed
            else:
                line += f"[no candidates] {str(block)}"

        if line:
            print(line + reset)

        # Print children
        children = tree.get_children(node)
        sorted_children = sorted(
            children, key=lambda n: (n.block.id if n.block else -1)
        )
        for i, child in enumerate(sorted_children):
            child_is_last = i == len(sorted_children) - 1
            print_node(child, depth + 1, child_is_last)

    # Print tree
    for root in tree.roots:
        print_node(root, 0)

    # Summary stats
    total = len(page.blocks)
    with_labels = sum(1 for b in page.blocks if result.get_label(b) is not None)
    removed = sum(1 for b in page.blocks if result.is_removed(b))
    no_candidates = total - with_labels - removed
    total_candidates = sum(len(candidates) for candidates in all_candidates.values())
    num_synthetic = len(synthetic_nodes)

    # Great a histogram of labels per block
    block_histogram = defaultdict(int)
    for block in page.blocks:
        labels = result.get_all_candidates_for_block(block)
        block_histogram[len(labels)] += 1

    print(f"\n{'─' * 80}")
    print(
        f"Blocks: {total} total | {with_labels} labeled | "
        f"{removed} removed | {no_candidates} no candidates"
        f" | Histogram: {dict(block_histogram)}"
    )
    print(f"Candidates: {total_candidates} total | {num_synthetic} synthetic")

    # Detailed candidate analysis
    if show_candidates:
        print(f"\n{'=' * 80}")
        print("CANDIDATES BY LABEL")
        print(f"{'=' * 80}")

        # Get all candidates
        all_candidates = result.get_all_candidates()

        # Filter to specific label if requested
        if label:
            labels_to_show = {label: all_candidates.get(label, [])}
        else:
            labels_to_show = all_candidates

        # Build set of elements that made it into the final Page hierarchy
        # These are the "winners" - candidates whose constructed elements
        # were actually used in the final output
        elements_in_page: set[int] = set()
        if result.page:
            for element in result.page.iter_elements():
                elements_in_page.add(id(element))

        # Summary table
        print(f"\n{'Label':<20} {'Total':<8} {'In Page':<8} {'Constructed':<12}")
        print(f"{'-' * 52}")
        for lbl in sorted(labels_to_show.keys()):
            candidates = labels_to_show[lbl]
            in_page = [
                c
                for c in candidates
                if c.constructed and id(c.constructed) in elements_in_page
            ]
            constructed = [c for c in candidates if c.constructed is not None]
            print(
                f"{lbl:<20} {len(candidates):<8} "
                f"{len(in_page):<8} {len(constructed):<12}"
            )

        # Detailed per-label breakdown
        for lbl in sorted(labels_to_show.keys()):
            candidates = labels_to_show[lbl]
            if not candidates:
                continue

            # Sort by score (highest first) for better readability
            sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

            print(f"\n{lbl} ({len(candidates)} candidates):")
            for candidate in sorted_candidates:
                # TODO Use all source blocks, not just the first one.
                block = candidate.source_blocks[0] if candidate.source_blocks else None
                block_id_str = f"{block.id:3d}" if block else "  ?"

                # Determine if this candidate made it into the final Page
                in_page = (
                    candidate.constructed
                    and id(candidate.constructed) in elements_in_page
                )
                winner_mark = "✓ " if in_page else "  "

                if candidate.constructed:
                    constructed_str = str(candidate.constructed)
                else:
                    constructed_str = "<never constructed>"

                source_str = str(block) if block else "no source"
                print(
                    f"  {winner_mark}{block_id_str} [{lbl}] {constructed_str} | "
                    f"score={candidate.score:.3f} | {source_str}"
                )

    # Page hierarchy
    if show_hierarchy:
        page_obj = result.page
        if page_obj:
            print(f"\n{'=' * 80}")
            print("PAGE HIERARCHY")
            print(f"{'=' * 80}")
            page_num_str = (
                page_obj.page_number.value if page_obj.page_number else "None"
            )
            categories_str = (
                ", ".join(c.name for c in page_obj.categories)
                if page_obj.categories
                else "None"
            )
            print(f"Page number: {page_num_str}")
            print(f"Categories: {categories_str}")
            print(f"Progress bar: {'Yes' if page_obj.progress_bar else 'No'}")

            if page_obj.catalog:
                parts_count = len(page_obj.catalog.parts)
                total_items = sum(p.count.count for p in page_obj.catalog.parts)
                print(f"Catalog: {parts_count} parts ({total_items} total items)")

            steps = page_obj.instruction.steps if page_obj.instruction else []
            print(f"Steps: {len(steps)}")

            for i, step in enumerate(steps, 1):
                parts_count = len(step.parts_list.parts) if step.parts_list else 0
                print(f"  Step {i}: #{step.step_number.value} ({parts_count} parts)")

    print(f"\n{'=' * 80}\n")


def print_page_hierarchy(page_data: PageData, page: Page) -> None:
    """Print the structured LEGO page hierarchy.

    Args:
        page_data: PageData containing the raw page number
        page: Structured Page object with steps, parts lists, etc.
    """
    categories_str = (
        f" ([{', '.join(c.name for c in page.categories)}])" if page.categories else ""
    )
    print(f"Page {page_data.page_number}{categories_str}:")

    if page.page_number:
        print(f"  ✓ Page Number: {page.page_number.value}")

    if page.scale:
        print(f"  ✓ Scale: 1:1 reference for length {page.scale.length.value}")

    if page.instruction and page.instruction.open_bags:
        print(f"  ✓ Open Bags: {len(page.instruction.open_bags)}")
        for open_bag in page.instruction.open_bags:
            bag_label = (
                f"Bag {open_bag.number.value}" if open_bag.number else "Bag (all)"
            )
            print(f"    - {bag_label} at {open_bag.bbox}")

    if page.catalog:
        parts_count = len(page.catalog.parts)
        total_items = sum(p.count.count for p in page.catalog.parts)
        print(f"  ✓ Catalog: {parts_count} parts ({total_items} total items)")
        if page.catalog.parts:
            print("      Parts:")
            for part in page.catalog.parts:
                number_str = part.number.element_id if part.number else "no number"
                print(f"        • {part.count.count}x ({number_str})")

    if page.instruction and page.instruction.steps:
        print(f"  ✓ Steps: {len(page.instruction.steps)}")
        for step in page.instruction.steps:
            parts_count = len(step.parts_list.parts) if step.parts_list else 0
            print(f"    - Step {step.step_number.value} ({parts_count} parts)")
            # Print parts list details
            if step.parts_list and step.parts_list.parts:
                print("      Parts List:")
                for part in step.parts_list.parts:
                    number_str = part.number.element_id if part.number else "no number"
                    print(f"        • {part.count.count}x ({number_str})")
            else:
                print("      Parts List: (none)")

            if step.diagram:
                print(f"      Diagram: {step.diagram.bbox}")


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
        page = result.page
        if page:
            print_page_hierarchy(page_data, page)
