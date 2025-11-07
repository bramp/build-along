"""Filter duplicate/similar blocks from PDF pages.

This module provides utilities to remove duplicate blocks that are often
created for visual effects like drop shadows in PDF rendering.
"""

from __future__ import annotations

from collections.abc import Sequence

from build_a_long.pdf_extract.extractor.page_blocks import Block


def filter_duplicate_blocks(blocks: Sequence[Block]) -> list[Block]:
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
        Filtered list of blocks with duplicates removed, preserving original order.

    Example:
        >>> # Three blocks forming a drop shadow effect
        >>> blocks = [
        ...     Drawing(bbox=BBox(10, 10, 30, 30)),  # Main element
        ...     Drawing(bbox=BBox(11, 11, 31, 31)),  # Shadow offset by 1px
        ...     Drawing(bbox=BBox(12, 12, 32, 32)),  # Second shadow offset by 2px
        ... ]
        >>> result = filter_duplicate_blocks(blocks)
        >>> len(result)  # Returns 1, keeping only the largest
        1
    """
    if not blocks:
        return []

    IOU_THRESHOLD = 0.8

    # Helper function to check if two blocks are similar based on IOU
    def are_similar(block_i: Block, block_j: Block) -> bool:
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
    for group_indices in groups.values():
        # Find the block with the largest area in this group
        largest_idx = max(group_indices, key=lambda idx: blocks[idx].bbox.area)
        result_indices.append(largest_idx)

    # Return blocks in their original order
    result_indices.sort()
    return [blocks[i] for i in result_indices]
