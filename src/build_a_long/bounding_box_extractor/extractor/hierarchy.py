"""
Utilities to build a hierarchy of page elements from a flat list of
extracted blocks, using bounding-box containment.

We nest elements by bbox containment, choosing the smallest containing
ancestor for each child.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from build_a_long.bounding_box_extractor.extractor.bbox import BBox
from build_a_long.bounding_box_extractor.extractor.page_elements import Element

logger = logging.getLogger(__name__)


def build_hierarchy_from_elements(
    elements: Sequence[Element],
) -> List[Element]:
    """Build a containment-based hierarchy from typed elements.

    Strategy:
    - Sort elements by area ascending (smallest first) so children attach before parents.
    - For each element, find the smallest containing ancestor and attach as a child.

    Returns:
        List of top-level elements with their children nested in the children field.
    """
    converted: List[Element] = list(elements)

    # TODO Move this to a method on BBox
    def _area(b: BBox) -> float:
        return max(0.0, (b.x1 - b.x0)) * max(0.0, (b.y1 - b.y0))

    # Sort indices by area ascending to assign children first
    idxs = sorted(range(len(converted)), key=lambda i: _area(converted[i].bbox))

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
                area = _area(candidate.bbox)
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

    # Recursively produce Element trees with children attached.
    def build_element(i: int) -> Element:
        ele = converted[i]
        ele.children = [build_element(cidx) for cidx in children_lists[i]]
        return ele

    return [build_element(r) for r in roots]
