"""
Utilities to build a hierarchy of page elements from a flat list of
extracted blocks, using bounding-box containment.

We nest elements by bbox containment, choosing the smallest containing
ancestor for each child.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Sequence

from build_a_long.pdf_extract.extractor.page_elements import PageElement

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
    """

    roots: List[PageElement] = field(default_factory=list)
    parent_map: Dict[int, Optional[PageElement]] = field(default_factory=dict)
    children_map: Dict[int, List[PageElement]] = field(default_factory=dict)

    def get_children(self, element: PageElement) -> List[PageElement]:
        """Get the children of a given element.

        Args:
            element: The element to get children for

        Returns:
            List of child elements (empty list if no children)
        """
        return self.children_map.get(id(element), [])

    def get_parent(self, element: PageElement) -> Optional[PageElement]:
        """Get the parent of a given element.

        Args:
            element: The element to get parent for

        Returns:
            Parent element or None if this is a root element
        """
        return self.parent_map.get(id(element))

    def is_root(self, element: PageElement) -> bool:
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
    elements: Sequence[PageElement],
) -> ElementTree:
    """Build a containment-based hierarchy from typed elements.

    Strategy:
    - Sort elements by area ascending (smallest first) so children attach before parents.
    - For each element, find the smallest containing ancestor and attach as a child.

    Returns:
        ElementTree containing the hierarchy with roots and parent/children mappings.
    """
    converted: List[PageElement] = list(elements)

    # Sort indices by area ascending to assign children first
    idxs = sorted(range(len(converted)), key=lambda i: converted[i].bbox.area())

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
                area = candidate.bbox.area()
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

    return tree
