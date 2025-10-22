"""
Utilities to build a hierarchy of page elements from a flat list of
extracted blocks, using bounding-box containment.

We convert the extractor's dict-based elements to typed PageElement instances
when possible (StepNumber, Drawing) and fall back to Unknown when the type is
ambiguous. We then nest elements by bbox containment, choosing the smallest
containing ancestor for each child.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from build_a_long.bounding_box_extractor.bbox import BBox
from build_a_long.bounding_box_extractor.page_elements import (
    Element,
    Unknown,
)


@dataclass(frozen=True)
class ElementNode:
    """A tree node holding a PageElement and the nested children."""

    element: Element
    children: Tuple["ElementNode", ...] = ()


def build_hierarchy_from_elements(
    elements: Sequence[Element],
) -> Tuple[ElementNode, ...]:
    """Build a containment-based hierarchy from typed elements.

    Strategy:
    - Sort elements by area ascending (smallest first) so children attach before parents.
    - For each element, find the smallest containing ancestor and attach as a child.
    - Unknown parents mirror their direct children's elements in their `children` tuple.
    """
    converted: List[Element] = list(elements)

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

    # Recursively produce ElementNode trees. For Unknown parents, we also
    # embed children into the Unknown element's `children` tuple for convenience.
    def build_node(i: int) -> ElementNode:
        child_nodes = tuple(build_node(cidx) for cidx in children_lists[i])
        ele = converted[i]
        if isinstance(ele, Unknown):
            ele = Unknown(
                bbox=ele.bbox,
                label=ele.label,
                raw_type=ele.raw_type,
                content=ele.content,
                source_id=ele.source_id,
                children=tuple(c.element for c in child_nodes),
            )
        return ElementNode(element=ele, children=child_nodes)

    return tuple(build_node(r) for r in roots)
