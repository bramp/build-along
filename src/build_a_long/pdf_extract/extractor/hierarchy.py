"""
Utilities to build a hierarchy of page blocks from a flat list of
extracted blocks, using bounding-box containment.

We nest blocks by bbox containment, choosing the smallest containing
ancestor for each child.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from build_a_long.pdf_extract.extractor.page_blocks import Block

logger = logging.getLogger(__name__)


@dataclass
class BlockTree:
    """A tree structure for managing hierarchical relationships between blocks.

    This separates the hierarchy structure from the blocks themselves, keeping
    blocks as pure data holders without circular references.

    Attributes:
        roots: Top-level blocks with no parent
        parent_map: Maps block id to its parent block (None for roots)
        children_map: Maps block id to list of its children blocks
        depth_map: Maps block id to its nesting depth (0 for roots)
    """

    roots: list[Block] = field(default_factory=list)
    parent_map: dict[int, Block | None] = field(default_factory=dict)
    children_map: dict[int, list[Block]] = field(default_factory=dict)
    depth_map: dict[int, int] = field(default_factory=dict)

    def get_children(self, block: Block) -> list[Block]:
        """Get the children of a given block.

        Args:
            block: The block to get children for

        Returns:
            List of child blocks (empty list if no children)
        """
        return self.children_map.get(id(block), [])

    def get_descendants(self, block: Block) -> list[Block]:
        """Get all descendants of a given block (children, grandchildren, etc.).

        Args:
            block: The block to get descendants for

        Returns:
            List of all descendant blocks (empty list if no descendants)
        """
        descendants: list[Block] = []
        children = self.get_children(block)
        for child in children:
            descendants.append(child)
            # Recursively add all descendants of this child
            descendants.extend(self.get_descendants(child))
        return descendants

    def get_parent(self, block: Block) -> Block | None:
        """Get the parent of a given block.

        Args:
            block: The block to get parent for

        Returns:
            Parent block or None if this is a root block
        """
        return self.parent_map.get(id(block))

    def get_depth(self, block: Block) -> int:
        """Get the nesting depth of a block.

        Args:
            block: The block to get depth for

        Returns:
            Nesting depth (0 for root blocks, 1 for their children, etc.)
        """
        return self.depth_map.get(id(block), 0)

    def is_root(self, block: Block) -> bool:
        """Check if a block is a root block.

        Args:
            block: The block to check

        Returns:
            True if the block is a root, False otherwise
        """
        return id(block) not in self.parent_map or self.parent_map[id(block)] is None


def build_hierarchy_from_blocks(
    blocks: Sequence[Block],
) -> BlockTree:
    """Build a containment-based hierarchy from typed blocks.

    Strategy:
    - Sort blocks by area ascending (smallest first) so children attach before parents.
    - For each block, find the smallest containing ancestor and attach as a child.

    Returns:
        BlockTree containing the hierarchy with roots and parent/children mappings.
    """
    converted: list[Block] = list(blocks)

    # Sort indices by area ascending to assign children first
    idxs = sorted(range(len(converted)), key=lambda i: converted[i].bbox.area)

    # Prepare parent mapping: each index maps to parent index or None
    parent: list[int | None] = [None] * len(converted)

    for i in idxs:  # small to large
        bbox_i = converted[i].bbox
        best_parent: int | None = None
        best_parent_area: float = float("inf")
        for j, candidate in enumerate(converted):
            if i == j:
                continue
            if bbox_i.fully_inside(candidate.bbox):
                area = candidate.bbox.area
                if area < best_parent_area:
                    best_parent = j
                    best_parent_area = area
        parent[i] = best_parent

    # Build children arrays
    children_lists: list[list[int]] = [[] for _ in converted]
    roots: list[int] = []
    for i, p in enumerate(parent):
        if p is None:
            roots.append(i)
        else:
            children_lists[p].append(i)

    # Build BlockTree structure
    tree = BlockTree()
    tree.roots = [converted[r] for r in roots]

    for i, block in enumerate(converted):
        parent_idx = parent[i]
        if parent_idx is not None:
            tree.parent_map[id(block)] = converted[parent_idx]
        else:
            tree.parent_map[id(block)] = None

        tree.children_map[id(block)] = [converted[cidx] for cidx in children_lists[i]]

    # Calculate depths by walking from roots - O(n)
    def _calculate_depth(block: Block, depth: int) -> None:
        """Recursively calculate and store depth for block and its descendants."""
        tree.depth_map[id(block)] = depth
        for child in tree.children_map.get(id(block), []):
            _calculate_depth(child, depth + 1)

    for root in tree.roots:
        _calculate_depth(root, 0)

    return tree
