"""Filter duplicate/similar blocks from PDF pages.

This module provides utilities to remove duplicate blocks that are often
created for visual effects like drop shadows in PDF rendering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from build_a_long.pdf_extract.classifier.removal_reason import RemovalReason
from build_a_long.pdf_extract.extractor.page_blocks import Blocks

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classification_result import (
        ClassificationResult,
    )


def filter_duplicate_blocks(
    blocks: Sequence[Blocks],
) -> tuple[list[Blocks], dict[Blocks, Blocks]]:
    """Filter out duplicate/similar blocks, keeping the largest one from each group.

    Pages often contain multiple overlapping blocks at similar positions to create
    visual effects like drop shadows. This function identifies groups of similar
    blocks based on their IOU (Intersection over Union) and keeps only the largest
    one from each group.

    Two blocks are considered similar if they have high IOU (≥0.8), indicating
    significant overlap.

    Args:
        blocks: List of blocks to filter.

    Returns:
        A tuple of:
        - Filtered list of blocks with duplicates removed, preserving original order
        - Dict mapping removed blocks to the block that was kept instead

    Example:
        >>> # Three blocks forming a drop shadow effect
        >>> blocks = [
        ...     Drawing(bbox=BBox(10, 10, 30, 30)),  # Main element
        ...     Drawing(bbox=BBox(11, 11, 31, 31)),  # Shadow offset by 1px
        ...     Drawing(bbox=BBox(12, 12, 32, 32)),  # Second shadow offset by 2px
        ... ]
        >>> kept, removed = filter_duplicate_blocks(blocks)
        >>> len(kept)  # Returns 1, keeping only the largest
        1
        >>> len(removed)  # Returns 2, the two smaller blocks
        2
    """
    if not blocks:
        return [], {}

    IOU_THRESHOLD = 0.9

    # Helper function to check if two blocks are similar based on IOU
    def are_similar(block_i: Blocks, block_j: Blocks) -> bool:
        return block_i.bbox.iou(block_j.bbox) >= IOU_THRESHOLD

    # Union-find data structure for grouping similar blocks
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

    # Compare all pairs of blocks (O(n²))
    for i in range(n):
        for j in range(i + 1, n):
            if are_similar(blocks[i], blocks[j]):
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
    removed_mapping: dict[Blocks, Blocks] = {}

    for group_indices in groups.values():
        # Find the block with the largest area in this group
        largest_idx = max(group_indices, key=lambda idx: blocks[idx].bbox.area)
        result_indices.append(largest_idx)

        # Map all other blocks in the group to the kept block
        kept_block = blocks[largest_idx]
        for idx in group_indices:
            if idx != largest_idx:
                removed_mapping[blocks[idx]] = kept_block

    # Return blocks in their original order
    result_indices.sort()
    return [blocks[i] for i in result_indices], removed_mapping


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

    for ele in result.page_data.blocks:
        if ele is target or id(ele) in keep_ids:
            continue
        b = ele.bbox
        if b.fully_inside(target_bbox):
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
