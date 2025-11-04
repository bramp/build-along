"""
Utilities to build a hierarchy of page elements from a flat list of
extracted blocks, using bounding-box containment.

We nest elements by bbox containment, choosing the smallest containing
ancestor for each child.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from build_a_long.pdf_extract.extractor.page_elements import Element

logger = logging.getLogger(__name__)


@dataclass
class ElementTree:
    """A tree structure for managing hierarchical relationships between elements.

    This separates the hierarchy structure from the elements themselves, keeping
    elements as pure data holders without circular references.

    Attributes:
        roots: Top-level elements with no parent
        parent_map: Maps element id to its parent element (None for roots)
        children_map: Maps element id to list of its children elements
        depth_map: Maps element id to its nesting depth (0 for roots)
    """

    roots: List[Element] = field(default_factory=list)
    parent_map: Dict[int, Optional[Element]] = field(default_factory=dict)
    children_map: Dict[int, List[Element]] = field(default_factory=dict)
    depth_map: Dict[int, int] = field(default_factory=dict)

    def get_children(self, element: Element) -> List[Element]:
        """Get the children of a given element.

        Args:
            element: The element to get children for

        Returns:
            List of child elements (empty list if no children)
        """
        return self.children_map.get(id(element), [])

    def get_descendants(self, element: Element) -> List[Element]:
        """Get all descendants of a given element (children, grandchildren, etc.).

        Args:
            element: The element to get descendants for

        Returns:
            List of all descendant elements (empty list if no descendants)
        """
        descendants: List[Element] = []
        children = self.get_children(element)
        for child in children:
            descendants.append(child)
            # Recursively add all descendants of this child
            descendants.extend(self.get_descendants(child))
        return descendants

    def get_parent(self, element: Element) -> Optional[Element]:
        """Get the parent of a given element.

        Args:
            element: The element to get parent for

        Returns:
            Parent element or None if this is a root element
        """
        return self.parent_map.get(id(element))

    def get_depth(self, element: Element) -> int:
        """Get the nesting depth of an element.

        Args:
            element: The element to get depth for

        Returns:
            Nesting depth (0 for root elements, 1 for their children, etc.)
        """
        return self.depth_map.get(id(element), 0)

    def is_root(self, element: Element) -> bool:
        """Check if an element is a root element.

        Args:
            element: The element to check

        Returns:
            True if the element is a root, False otherwise
        """
        return (
            id(element) not in self.parent_map or self.parent_map[id(element)] is None
        )


def build_hierarchy_from_elements(
    elements: Sequence[Element],
) -> ElementTree:
    """Build a containment-based hierarchy from typed elements.

    Strategy:
    - Sort elements by area ascending (smallest first) so children attach before parents.
    - For each element, find the smallest containing ancestor and attach as a child.

    Returns:
        ElementTree containing the hierarchy with roots and parent/children mappings.
    """
    converted: List[Element] = list(elements)

    # Sort indices by area ascending to assign children first
    idxs = sorted(range(len(converted)), key=lambda i: converted[i].bbox.area)

    # Prepare parent mapping: each index maps to parent index or None
    parent: List[Optional[int]] = [None] * len(converted)

    for i in idxs:  # small to large
        bbox_i = converted[i].bbox
        best_parent: Optional[int] = None
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
    children_lists: List[List[int]] = [[] for _ in converted]
    roots: List[int] = []
    for i, p in enumerate(parent):
        if p is None:
            roots.append(i)
        else:
            children_lists[p].append(i)

    # Build ElementTree structure
    tree = ElementTree()
    tree.roots = [converted[r] for r in roots]

    for i, element in enumerate(converted):
        parent_idx = parent[i]
        if parent_idx is not None:
            tree.parent_map[id(element)] = converted[parent_idx]
        else:
            tree.parent_map[id(element)] = None

        tree.children_map[id(element)] = [converted[cidx] for cidx in children_lists[i]]

    # Calculate depths by walking from roots - O(n)
    def _calculate_depth(element: Element, depth: int) -> None:
        """Recursively calculate and store depth for element and its descendants."""
        tree.depth_map[id(element)] = depth
        for child in tree.children_map.get(id(element), []):
            _calculate_depth(child, depth + 1)

    for root in tree.roots:
        _calculate_depth(root, 0)

    return tree
