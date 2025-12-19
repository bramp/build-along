"""Diagnostic utilities for analyzing unassigned blocks.

This module provides tools to categorize and explain why blocks weren't assigned
to any LEGO page element. It helps identify patterns in unassigned blocks and
provides actionable recommendations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing, Image, Text

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classification_result import (
        ClassificationResult,
    )
    from build_a_long.pdf_extract.extractor.extractor import PageData


class UnassignedCategory(Enum):
    """Categories of unassigned blocks."""

    ZERO_WIDTH = auto()
    """Drawing with zero width (x0 == x1)"""

    ZERO_HEIGHT = auto()
    """Drawing with zero height (y0 == y1)"""

    PAGE_EDGE_LINE = auto()
    """Line at page boundary (x=0 or x=page_width)"""

    WHITESPACE_TEXT = auto()
    """Text containing only whitespace"""

    COPYRIGHT_TEXT = auto()
    """Copyright or trademark text on info pages"""

    SMALL_DOT = auto()
    """Very small drawing (likely a dot or artifact)"""

    LARGE_UNCLASSIFIED = auto()
    """Large drawing that didn't match any classifier"""

    IMAGE_IN_COMPLEX_PAGE = auto()
    """Image on catalog/info page that wasn't assigned"""

    UNKNOWN = auto()
    """Block that doesn't fit known categories"""


@dataclass
class UnassignedBlockInfo:
    """Information about an unassigned block."""

    block: Blocks
    category: UnassignedCategory
    reason: str
    recommendation: str


def categorize_unassigned_block(
    block: Blocks,
    page_data: PageData,
) -> UnassignedBlockInfo:
    """Categorize an unassigned block and provide actionable information.

    Args:
        block: The unassigned block
        page_data: Page data for context (page dimensions, etc.)

    Returns:
        UnassignedBlockInfo with category, reason, and recommendation
    """
    page_width = page_data.bbox.width
    page_height = page_data.bbox.height

    # Check for zero-dimension drawings
    if isinstance(block, Drawing):
        bbox = block.bbox
        width = bbox.width
        height = bbox.height

        # Zero width
        if width == 0:
            is_at_edge = bbox.x0 == 0 or bbox.x0 == page_width
            if is_at_edge:
                return UnassignedBlockInfo(
                    block=block,
                    category=UnassignedCategory.PAGE_EDGE_LINE,
                    reason=f"Zero-width line at page edge (x={bbox.x0})",
                    recommendation="Filter page-edge lines in block_filter.py",
                )
            return UnassignedBlockInfo(
                block=block,
                category=UnassignedCategory.ZERO_WIDTH,
                reason=f"Zero-width drawing at x={bbox.x0}",
                recommendation="Filter zero-width drawings in block_filter.py",
            )

        # Zero height
        if height == 0:
            is_at_edge = bbox.y0 == 0 or bbox.y0 == page_height
            if is_at_edge:
                return UnassignedBlockInfo(
                    block=block,
                    category=UnassignedCategory.PAGE_EDGE_LINE,
                    reason=f"Zero-height line at page edge (y={bbox.y0})",
                    recommendation="Filter page-edge lines in block_filter.py",
                )
            return UnassignedBlockInfo(
                block=block,
                category=UnassignedCategory.ZERO_HEIGHT,
                reason=f"Zero-height drawing at y={bbox.y0}",
                recommendation="Filter zero-height drawings in block_filter.py",
            )

        # Small dot (area < 25 sq pts, roughly 5x5 or less)
        if bbox.area < 25:
            return UnassignedBlockInfo(
                block=block,
                category=UnassignedCategory.SMALL_DOT,
                reason=f"Very small drawing (area={bbox.area:.1f} sq pts)",
                recommendation="Consider if this is a significant element or artifact",
            )

        # Large unclassified drawing (> 5% of page area)
        page_area = page_width * page_height
        if bbox.area > page_area * 0.05:
            return UnassignedBlockInfo(
                block=block,
                category=UnassignedCategory.LARGE_UNCLASSIFIED,
                reason=f"Large drawing ({bbox.area / page_area * 100:.1f}% of page)",
                recommendation="Review if this should be a background, diagram, or other element",
            )

    # Check for whitespace-only text
    if isinstance(block, Text):
        if block.text.strip() == "":
            return UnassignedBlockInfo(
                block=block,
                category=UnassignedCategory.WHITESPACE_TEXT,
                reason="Text contains only whitespace",
                recommendation="Filter whitespace-only text in block_filter.py",
            )

        # Check for copyright/trademark text
        copyright_keywords = {
            "©",
            "™",
            "®",
            "copyright",
            "trademark",
            "lego.com",
            "lucasfilm",
            "disney",
            "marcas registradas",
        }
        text_lower = block.text.lower()
        if any(kw in text_lower for kw in copyright_keywords):
            return UnassignedBlockInfo(
                block=block,
                category=UnassignedCategory.COPYRIGHT_TEXT,
                reason=f"Copyright/trademark text: '{block.text[:40]}...'",
                recommendation="Add classifier for legal/copyright text",
            )

    # Check for unclassified images
    if isinstance(block, Image):
        return UnassignedBlockInfo(
            block=block,
            category=UnassignedCategory.IMAGE_IN_COMPLEX_PAGE,
            reason="Image not assigned to any element",
            recommendation="Review if this should be a diagram, part_image, or other element",
        )

    # Default: unknown category
    return UnassignedBlockInfo(
        block=block,
        category=UnassignedCategory.UNKNOWN,
        reason="Does not match known unassigned patterns",
        recommendation="Manual review required",
    )


def get_unassigned_blocks(result: ClassificationResult) -> list[Blocks]:
    """Get all unassigned blocks from a classification result.

    Args:
        result: Classification result to check

    Returns:
        List of blocks that are unassigned (no candidate and not removed)
    """
    unassigned = []
    for block in result.page_data.blocks:
        # Check if block is assigned to a constructed candidate
        best_candidate = result.get_best_candidate(block)
        if best_candidate:
            continue

        # Check if block was explicitly removed
        if result.is_removed(block):
            continue

        # Block is unassigned
        unassigned.append(block)

    return unassigned


def analyze_unassigned_blocks(
    result: ClassificationResult,
) -> dict[UnassignedCategory, list[UnassignedBlockInfo]]:
    """Analyze all unassigned blocks and group by category.

    Args:
        result: Classification result to analyze

    Returns:
        Dictionary mapping categories to lists of UnassignedBlockInfo
    """
    categorized: dict[UnassignedCategory, list[UnassignedBlockInfo]] = defaultdict(list)

    unassigned = get_unassigned_blocks(result)
    for block in unassigned:
        info = categorize_unassigned_block(block, result.page_data)
        categorized[info.category].append(info)

    return dict(categorized)


def print_unassigned_diagnostics(
    results: list[ClassificationResult],
    *,
    show_details: bool = True,
) -> None:
    """Print diagnostic report for unassigned blocks across all pages.

    Args:
        results: List of classification results
        show_details: If True, show individual block details
    """
    # Aggregate statistics
    total_unassigned = 0
    category_counts: dict[UnassignedCategory, int] = defaultdict(int)
    pages_with_unassigned: list[tuple[int, dict]] = []

    for result in results:
        if result.skipped_reason:
            continue

        analysis = analyze_unassigned_blocks(result)
        if not analysis:
            continue

        page_num = result.page_data.page_number
        page_total = sum(len(blocks) for blocks in analysis.values())
        total_unassigned += page_total

        for category, blocks in analysis.items():
            category_counts[category] += len(blocks)

        pages_with_unassigned.append((page_num, analysis))

    if total_unassigned == 0:
        print("\n✓ All blocks are assigned!")
        return

    print(f"\n{'=' * 80}")
    print("UNASSIGNED BLOCK DIAGNOSTICS")
    print(f"{'=' * 80}")
    print(f"\nTotal unassigned blocks: {total_unassigned}")
    print(f"Pages with unassigned blocks: {len(pages_with_unassigned)}")

    # Print category summary
    print("\nBy Category:")
    print("-" * 60)
    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    )
    for category, count in sorted_categories:
        pct = count / total_unassigned * 100
        print(f"  {category.name:25} {count:5d} ({pct:5.1f}%)")

    # Print recommendations
    print("\nRecommendations:")
    print("-" * 60)
    recommendations: dict[str, int] = defaultdict(int)

    for _page_num, analysis in pages_with_unassigned:
        for _category, blocks in analysis.items():
            for info in blocks:
                recommendations[info.recommendation] += 1

    for rec, count in sorted(recommendations.items(), key=lambda x: -x[1]):
        print(f"  • {rec} ({count} blocks)")

    # Print per-page details if requested
    if show_details:
        print(f"\n{'=' * 80}")
        print("PER-PAGE DETAILS")
        print(f"{'=' * 80}")

        for page_num, analysis in pages_with_unassigned:
            page_total = sum(len(blocks) for blocks in analysis.values())
            print(f"\nPage {page_num}: {page_total} unassigned blocks")

            for category, blocks in sorted(analysis.items(), key=lambda x: x[0].name):
                if not blocks:
                    continue
                print(f"  {category.name}:")
                for info in blocks[:5]:  # Limit to 5 per category per page
                    block = info.block
                    bbox_str = f"({block.bbox.x0:.1f},{block.bbox.y0:.1f},{block.bbox.x1:.1f},{block.bbox.y1:.1f})"
                    print(f"    #{block.id} {type(block).__name__} {bbox_str}")
                    print(f"       Reason: {info.reason}")

                if len(blocks) > 5:
                    print(f"    ... and {len(blocks) - 5} more")
