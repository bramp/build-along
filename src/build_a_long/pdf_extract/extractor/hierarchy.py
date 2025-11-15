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
from typing import Protocol

from build_a_long.pdf_extract.extractor.bbox import BBox

logger = logging.getLogger(__name__)


class HasBBox(Protocol):
    """Protocol for objects that have a bbox attribute."""

    @property
    def bbox(self) -> BBox: ...


@dataclass
class BlockTree[T: HasBBox]:
    """A tree structure for managing hierarchical relationships between objects.

    This separates the hierarchy structure from the objects themselves, keeping
    objects as pure data holders without circular references.

    Attributes:
        roots: Top-level objects with no parent
        parent_map: Maps object id to its parent object (None for roots)
        children_map: Maps object id to list of its children objects
        depth_map: Maps object id to its nesting depth (0 for roots)
    """

    roots: list[T] = field(default_factory=list)
    parent_map: dict[int, T | None] = field(default_factory=dict)
    children_map: dict[int, list[T]] = field(default_factory=dict)
    depth_map: dict[int, int] = field(default_factory=dict)

    def get_children(self, obj: T) -> list[T]:
        """Get the children of a given object.

        Args:
            obj: The object to get children for

        Returns:
            List of child objects (empty list if no children)
        """
        return self.children_map.get(id(obj), [])

    def get_descendants(self, obj: T) -> list[T]:
        """Get all descendants of a given object (children, grandchildren, etc.).

        Args:
            obj: The object to get descendants for

        Returns:
            List of all descendant objects (empty list if no descendants)
        """
        descendants: list[T] = []
        children = self.get_children(obj)
        for child in children:
            descendants.append(child)
            # Recursively add all descendants of this child
            descendants.extend(self.get_descendants(child))
        return descendants

    def get_parent(self, obj: T) -> T | None:
        """Get the parent of a given object.

        Args:
            obj: The object to get parent for

        Returns:
            Parent object or None if this is a root object
        """
        return self.parent_map.get(id(obj))

    def get_depth(self, obj: T) -> int:
        """Get the nesting depth of an object.

        Args:
            obj: The object to get depth for

        Returns:
            Nesting depth (0 for root objects, 1 for their children, etc.)
        """
        return self.depth_map.get(id(obj), 0)

    def is_root(self, obj: T) -> bool:
        """Check if an object is a root object.

        Args:
            obj: The object to check

        Returns:
            True if the object is a root, False otherwise
        """
        return id(obj) not in self.parent_map or self.parent_map[id(obj)] is None


def build_hierarchy_from_blocks[T: HasBBox](
    blocks: Sequence[T],
) -> BlockTree[T]:
    """Build a containment-based hierarchy from objects with bbox attributes.

    Strategy:
    - Sort objects by area ascending (smallest first) so children attach before parents.
    - For each object, find the smallest containing ancestor and attach as a child.

    Args:
        blocks: Sequence of objects that have a bbox attribute

    Returns:
        BlockTree containing the hierarchy with roots and parent/children mappings.
    """
    converted: list[T] = list(blocks)

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
            bbox_j = candidate.bbox
            # Skip if bboxes are identical (duplicate blocks at same position)
            # fully_inside uses >=/<= so identical boxes would be "inside" each other
            if bbox_i == bbox_j:
                continue
            if bbox_i.fully_inside(bbox_j):
                area = bbox_j.area
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
    tree: BlockTree[T] = BlockTree()
    tree.roots = [converted[r] for r in roots]

    for i, obj in enumerate(converted):
        parent_idx = parent[i]
        if parent_idx is not None:
            tree.parent_map[id(obj)] = converted[parent_idx]
        else:
            tree.parent_map[id(obj)] = None

        tree.children_map[id(obj)] = [converted[cidx] for cidx in children_lists[i]]

    # Calculate depths by walking from roots - O(n)
    def _calculate_depth(obj: T, depth: int) -> None:
        """Recursively calculate and store depth for object and its descendants."""
        tree.depth_map[id(obj)] = depth
        for child in tree.children_map.get(id(obj), []):
            _calculate_depth(child, depth + 1)

    for root in tree.roots:
        _calculate_depth(root, 0)

    return tree
