"""Filter duplicate/similar blocks from PDF pages.

This module provides utilities to remove duplicate blocks that are often
created for visual effects like drop shadows in PDF rendering.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING

from build_a_long.pdf_extract.classifier.removal_reason import RemovalReason
from build_a_long.pdf_extract.extractor.bbox import BBox, filter_contained
from build_a_long.pdf_extract.extractor.page_blocks import (
    Blocks,
    Drawing,
    Image,
    Text,
)

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classification_result import (
        ClassificationResult,
    )


logger = logging.getLogger(__name__)


def filter_duplicate_blocks(
    blocks: Sequence[Blocks],
) -> tuple[list[Blocks], dict[Blocks, RemovalReason]]:
    """Filter out duplicate blocks, keeping one from each group of true duplicates.

    Pages often contain multiple overlapping blocks at similar positions to create
    visual effects like drop shadows. This function identifies groups of blocks
    that are both spatially similar (high IOU) AND have the same visual properties
    (same text, same fill/stroke colors).

    Two blocks are considered duplicates if they:
    1. Have high IOU (≥0.9), indicating significant overlap
    2. Have the same visual content:
       - For Text blocks: same text content
       - For Drawing blocks: same fill and stroke colors

    Blocks with different visual properties (e.g., a white-filled box vs a
    black-bordered box at similar positions) are NOT considered duplicates
    and both will be kept.

    Args:
        blocks: List of blocks to filter.

    Returns:
        A tuple of:
        - Filtered list of blocks with duplicates removed, preserving original order
        - Dict mapping removed blocks to the RemovalReason
    """
    if not blocks:
        return [], {}

    IOU_THRESHOLD = 0.9

    def are_visually_same(block_i: Blocks, block_j: Blocks) -> bool:
        """Check if two blocks have the same visual properties."""
        # Must be the same type
        if type(block_i) is not type(block_j):
            return False

        if isinstance(block_i, Text) and isinstance(block_j, Text):
            # Text blocks: same text content
            return block_i.text == block_j.text

        if isinstance(block_i, Drawing) and isinstance(block_j, Drawing):
            # Drawing blocks: same fill and stroke colors
            return (
                block_i.fill_color == block_j.fill_color
                and block_i.stroke_color == block_j.stroke_color
            )

        # For other block types (e.g., Image), consider them the same
        # if they're the same type (already checked above)
        return True

    def are_duplicates(block_i: Blocks, block_j: Blocks) -> bool:
        """Check if two blocks are duplicates (similar bbox AND same visual content)."""
        if block_i.bbox.iou(block_j.bbox) < IOU_THRESHOLD:
            return False
        return are_visually_same(block_i, block_j)

    # Union-find data structure for grouping duplicate blocks
    n = len(blocks)
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    # TODO This is one of the slowest parts of processing.
    # Compare all pairs of blocks (O(n²))
    for i in range(n):
        for j in range(i + 1, n):
            if are_duplicates(blocks[i], blocks[j]):
                union(i, j)

    # Group blocks by their root parent
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # For each group, keep the block with the largest area
    result_indices = []
    removed_mapping: dict[Blocks, RemovalReason] = {}

    for group_indices in groups.values():
        # Find the block with the largest area in this group
        best_idx = max(group_indices, key=lambda idx: blocks[idx].bbox.area)
        result_indices.append(best_idx)

        # Map all other blocks in the group to the kept block
        kept_block = blocks[best_idx]
        for idx in group_indices:
            if idx != best_idx:
                removed_mapping[blocks[idx]] = RemovalReason(
                    reason_type="duplicate_bbox", target_block=kept_block
                )

    # Return blocks in their original order
    result_indices.sort()
    return [blocks[i] for i in result_indices], removed_mapping


def filter_background_blocks(
    blocks: Sequence[Blocks],
    page_width: float,
    page_height: float,
) -> tuple[list[Blocks], dict[Blocks, RemovalReason]]:
    """Filter out background blocks that cover most of the page.

    .. deprecated::
        Use :class:`BackgroundClassifier` instead, which properly scores and
        constructs Background elements rather than simply filtering them out.

    Removes blocks that have width >= 99% of page width AND
    height >= 99% of page height.
    These are typically background images or full-page rectangles.

    Args:
        blocks: List of blocks to filter.
        page_width: Width of the page.
        page_height: Height of the page.

    Returns:
        A tuple of:
        - Filtered list of blocks with background blocks removed, preserving order
        - Dict mapping removed blocks to the RemovalReason
    """
    if not blocks:
        return [], {}

    warnings.warn(
        "filter_background_blocks is deprecated. Use BackgroundClassifier instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    SIZE_THRESHOLD = 0.99
    min_width = page_width * SIZE_THRESHOLD
    min_height = page_height * SIZE_THRESHOLD

    filtered_blocks = []
    removed_mapping: dict[Blocks, RemovalReason] = {}

    for block in blocks:
        if block.bbox.width >= min_width and block.bbox.height >= min_height:
            removed_mapping[block] = RemovalReason(
                reason_type="background_block", target_block=None
            )
            logger.debug(
                "Filtered background block %s (size %.1fx%.1f on %.1fx%.1f page)",
                block.id,
                block.bbox.width,
                block.bbox.height,
                page_width,
                page_height,
            )
        else:
            filtered_blocks.append(block)

    return filtered_blocks, removed_mapping


def remove_child_bboxes(
    target: Blocks,
    result: ClassificationResult,
    keep_ids: set[int] | None = None,
) -> None:
    """Mark blocks fully inside target bbox as removed.

    Removes all blocks that are completely contained within the target block's
    bounding box, typically used to clean up text/drawings that overlap with
    a classified element.

    Args:
        target: The target block whose bbox defines the region.
        result: Classification result to mark removed blocks.
        keep_ids: Optional set of block IDs to keep (not remove).
    """
    if keep_ids is None:
        keep_ids = set()

    target_bbox = target.bbox

    candidates = [
        ele
        for ele in result.page_data.blocks
        if ele is not target and id(ele) not in keep_ids
    ]

    for ele in filter_contained(candidates, target_bbox):
        result.mark_removed(
            ele, RemovalReason(reason_type="child_bbox", target_block=target)
        )


def remove_similar_bboxes(
    target: Blocks,
    result: ClassificationResult,
    keep_ids: set[int] | None = None,
) -> None:
    """Mark blocks with similar position/size to target as removed.

    Removes blocks that are very similar to the target block based on:
    - High IOU (Intersection over Union) ≥ 0.8
    - Similar center position (within 1.5 pixels)
    - Similar area (within 12% tolerance)

    This is used to remove duplicates/shadows after a block has been classified.

    Args:
        target: The target block to compare against.
        result: Classification result to mark removed blocks.
        keep_ids: Optional set of block IDs to keep (not remove).
    """
    if keep_ids is None:
        keep_ids = set()

    target_area = target.bbox.area
    tx, ty = target.bbox.center

    IOU_THRESHOLD = 0.9
    CENTER_EPS = 1.5
    AREA_TOL = 0.12

    for ele in result.page_data.blocks:
        if ele is target or id(ele) in keep_ids:
            continue

        b = ele.bbox
        iou = target.bbox.iou(b)
        if iou >= IOU_THRESHOLD:
            result.mark_removed(
                ele,
                RemovalReason(reason_type="similar_bbox", target_block=target),
            )
            continue

        cx, cy = b.center
        if abs(cx - tx) <= CENTER_EPS and abs(cy - ty) <= CENTER_EPS:
            area = b.area
            if target_area > 0 and abs(area - target_area) / target_area <= AREA_TOL:
                result.mark_removed(
                    ele,
                    RemovalReason(reason_type="similar_bbox", target_block=target),
                )


def filter_overlapping_text_blocks(
    blocks: Sequence[Blocks],
) -> tuple[list[Blocks], dict[Blocks, RemovalReason]]:
    """Filter out overlapping text blocks with the same origin, keeping the longest.

    Some PDFs contain multiple text spans at the same origin where one is a
    substring of another (e.g., "4" and "43" both starting at the same position).
    This keeps only the text block with the longest text for each unique origin.

    Two text blocks are considered to have the same origin if their x0, y0, and y1
    coordinates match within a small tolerance (0.5 points).

    Args:
        blocks: List of blocks to filter (only Text blocks are affected).

    Returns:
        A tuple of:
        - Filtered list of blocks with duplicate text removed, preserving order
        - Dict mapping removed text blocks to the RemovalReason

    Example:
        >>> blocks = [
        ...     Text(bbox=BBox(34.0, 16.3, 48.6, 48.6), text="4"),
        ...     Text(bbox=BBox(34.0, 16.3, 63.2, 48.6), text="43"),
        ... ]
        >>> kept, removed = filter_overlapping_text_blocks(blocks)
        >>> len(kept)  # Returns 1, keeping "43"
        1
        >>> kept[0].text
        '43'
    """
    if not blocks:
        return [], {}

    # Collect text blocks and their indices
    text_blocks: list[Text] = []
    text_indices: list[int] = []

    for i, block in enumerate(blocks):
        if isinstance(block, Text):
            text_blocks.append(block)
            text_indices.append(i)

    if len(text_blocks) <= 1:
        return list(blocks), {}

    # Group text blocks by origin (x0, y0, y1) with tolerance
    tolerance = 0.5

    def origin_key(block: Text) -> tuple[float, float, float]:
        return (
            round(block.bbox.x0 / tolerance) * tolerance,
            round(block.bbox.y0 / tolerance) * tolerance,
            round(block.bbox.y1 / tolerance) * tolerance,
        )

    groups: dict[tuple[float, float, float], list[tuple[int, Text]]] = defaultdict(list)
    for idx, block in zip(text_indices, text_blocks, strict=True):
        key = origin_key(block)
        groups[key].append((idx, block))

    # For each group, keep the block with the longest text
    removed_mapping: dict[Blocks, RemovalReason] = {}
    removed_text_indices: set[int] = set()

    for blocks_in_group in groups.values():
        if len(blocks_in_group) > 1:
            # Find the block with the longest text (or widest bbox as tiebreaker)
            best_idx, best_block = max(
                blocks_in_group,
                key=lambda ib: (len(ib[1].text), ib[1].bbox.width),
            )

            # Map removed blocks to the kept block
            for idx, block in blocks_in_group:
                if idx != best_idx:
                    removed_text_indices.add(idx)
                    removed_mapping[block] = RemovalReason(
                        reason_type="overlapping_text", target_block=best_block
                    )
                    logger.debug(
                        "Filtered overlapping text at origin (%.1f, %.1f): "
                        "kept %r, removed %r",
                        best_block.bbox.x0,
                        best_block.bbox.y0,
                        best_block.text,
                        block.text,
                    )

    # Rebuild the block list preserving original order
    result = [b for i, b in enumerate(blocks) if i not in removed_text_indices]

    return result, removed_mapping


def find_contained_effects(
    primary_block: Blocks,
    all_blocks: Sequence[Blocks],
    *,
    margin: float = 2.0,
    max_area_ratio: float | None = None,
) -> list[Drawing | Image]:
    """Find Drawing/Image blocks that are visual effects for a primary block.

    LEGO PDFs often render elements (Text, Drawing, or Image) with multiple
    layers for visual effects like outlines, shadows, glows, or bevels. These
    appear as Drawing/Image elements fully contained within a slightly expanded
    version of the primary element's bbox.

    Args:
        primary_block: The block to find effects for.
        all_blocks: All blocks on the page to search through.
        margin: Margin to expand the primary bbox by when checking containment.
            Default 2.0 points.
        max_area_ratio: Optional maximum ratio of effect block area to primary
            block area. If provided, blocks larger than this ratio are excluded.

    Returns:
        List of Drawing/Image blocks that appear to be effects.
    """
    effects: list[Drawing | Image] = []
    primary_bbox = primary_block.bbox
    primary_area = primary_bbox.area

    # Expand primary bbox by margin to find contained blocks
    search_bbox = primary_bbox.expand(margin)

    for block in all_blocks:
        if block.id == primary_block.id:
            continue

        if not isinstance(block, (Drawing, Image)):
            continue

        # Effects should be fully contained within the expanded bbox
        if not search_bbox.contains(block.bbox):
            continue

        # Optional area ratio check
        if (
            max_area_ratio is not None
            and primary_area > 0
            and block.bbox.area / primary_area > max_area_ratio
        ):
            continue

        effects.append(block)

    return effects


def find_text_outline_effects(
    text_block: Text,
    all_blocks: Sequence[Blocks],
    *,
    margin: float = 2.0,
) -> list[Drawing]:
    """Find Drawing blocks that are text effects (outlines, shadows, etc.).

    .. deprecated::
        Use :func:`find_contained_effects` instead.

    LEGO PDFs often render text with visual effects like outlines or drop shadows,
    which appear as Drawing elements near/within the text bbox. Any Drawing whose
    bbox is fully contained within the text bbox (plus a small margin) is likely
    such an effect.

    This function identifies such effects so they can be included as source blocks
    when the Text is classified, preventing them from appearing as unconsumed.

    Args:
        text_block: The Text block to find effects for.
        all_blocks: All blocks on the page to search through.
        margin: Margin to expand the text bbox by when checking containment.
            Default 2.0 points.

    Returns:
        List of Drawing blocks that appear to be effects for the text.
    """
    warnings.warn(
        "find_text_outline_effects is deprecated. Use find_contained_effects instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return [
        b
        for b in find_contained_effects(text_block, all_blocks, margin=margin)
        if isinstance(b, Drawing)
    ]


def find_image_shadow_effects(
    primary_block: Drawing | Image,
    all_blocks: Sequence[Blocks],
    *,
    margin: float = 2.0,
) -> tuple[list[Drawing | Image], BBox]:
    """Find Drawing/Image blocks that are shadow effects for an image element.

    .. deprecated::
        Use :func:`find_contained_effects` instead.

    LEGO PDFs often render graphical elements with multiple layers for visual
    effects like shadows, glows, or bevels. These appear as Drawing/Image
    elements fully contained within a slightly expanded version of the primary
    element's bbox.

    This function identifies such effects so they can be included when
    calculating the element's bounding box, preventing them from appearing
    as unconsumed blocks.

    Args:
        primary_block: The primary Drawing/Image block to find effects for.
        all_blocks: All blocks on the page to search through.
        margin: Margin to expand the primary bbox by when checking containment.
            Default 2.0 points.

    Returns:
        A tuple of:
        - List of Drawing/Image blocks that appear to be effects
        - The combined bbox of primary_block and all effects
    """
    warnings.warn(
        "find_image_shadow_effects is deprecated. Use find_contained_effects instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    effects = find_contained_effects(primary_block, all_blocks, margin=margin)
    combined_bbox = primary_block.bbox
    for b in effects:
        combined_bbox = combined_bbox.union(b.bbox)

    return effects, combined_bbox


def find_horizontally_overlapping_blocks(
    primary_block: Drawing | Image,
    all_blocks: Sequence[Blocks],
    *,
    vertical_margin: float = 5.0,
) -> tuple[list[Drawing | Image], BBox]:
    """Find Drawing/Image blocks that overlap horizontally with the primary block.

    This is specifically designed for progress bars where multiple layers
    (borders, fills, inner bars) are stacked horizontally but may not be
    fully contained within each other.

    A block is included if:
    - It is vertically contained within the primary block (with margin)
    - It overlaps horizontally with the primary block

    This filters out full-page backgrounds (which extend beyond the bar
    vertically) while including inner elements.

    Args:
        primary_block: The primary Drawing/Image block.
        all_blocks: All blocks on the page to search through.
        vertical_margin: Margin to expand vertically when checking containment.
            Default 5.0 points.

    Returns:
        A tuple of:
        - List of Drawing/Image blocks that are horizontally overlapping
        - The combined bbox of primary_block and all overlapping blocks
    """
    effects: list[Drawing | Image] = []
    combined_bbox = primary_block.bbox

    # Expand vertically by margin to catch slight variations
    expanded_y0 = primary_block.bbox.y0 - vertical_margin
    expanded_y1 = primary_block.bbox.y1 + vertical_margin

    for block in all_blocks:
        if block is primary_block:
            continue

        if not isinstance(block, (Drawing, Image)):
            continue

        # Block must be vertically contained within the expanded extent
        # This filters out full-page backgrounds and vertical dividers
        if block.bbox.y0 < expanded_y0 or block.bbox.y1 > expanded_y1:
            continue

        # Block must overlap horizontally with the primary block
        if (
            block.bbox.x1 < primary_block.bbox.x0
            or block.bbox.x0 > primary_block.bbox.x1
        ):
            continue

        effects.append(block)
        combined_bbox = combined_bbox.union(block.bbox)

    return effects, combined_bbox
