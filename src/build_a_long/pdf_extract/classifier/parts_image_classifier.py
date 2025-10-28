"""
Part image classifier.

Purpose
-------
Associate each part_count text with exactly one image inside the chosen parts list,
using a heuristic: choose the image that is just above and left-aligned with the part
count. Enforce a one-to-one mapping between part counts and images.

Heuristic
---------
- Consider only Image elements fully inside any labeled parts_list Drawing.
- For each part_count Text inside the same parts_list, create candidate edges to
  Images that are above (image.y1 <= count.y0 + VERT_EPS) and roughly left-aligned
  (|image.x0 - count.x0| <= ALIGN_EPS).
- Sort candidates by vertical distance (count.y0 - image.y1), then greedily match
  to enforce one-to-one pairing.

Debugging
---------
Enable with CLASSIFIER_DEBUG=part_image or =all for structured logs.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.types import ClassifierConfig
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import (
    Image,
    Text,
    Drawing,
)

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classifier import Classifier

log = logging.getLogger(__name__)


class PartsImageClassifier(LabelClassifier):
    """Classifier for part images paired with part count texts."""

    outputs = {"part_image"}
    requires = {"parts_list", "part_count"}

    def __init__(self, config: ClassifierConfig, classifier: "Classifier"):
        super().__init__(config, classifier)
        self._debug_enabled = os.getenv("CLASSIFIER_DEBUG", "").lower() in (
            "part_image",
            "all",
        )

    def calculate_scores(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
    ) -> None:
        # Score-free: selection occurs in classify().
        return

    def _match_and_label_parts(
        self,
        edges: List[Tuple[float, Text, Image]],
        part_counts: List[Text],
        images: List[Image],
        labeled_elements: Dict[str, Any],
    ):
        edges.sort(key=lambda t: t[0])
        matched_counts: Set[int] = set()
        matched_images: Set[int] = set()

        if "part_image" not in labeled_elements:
            labeled_elements["part_image"] = []
        if "part_image_pairs" not in labeled_elements:
            labeled_elements["part_image_pairs"] = []

        for _, pc, img in edges:
            if id(pc) in matched_counts or id(img) in matched_images:
                continue
            matched_counts.add(id(pc))
            matched_images.add(id(img))
            img.label = img.label or "part_image"
            if img not in labeled_elements["part_image"]:
                labeled_elements["part_image"].append(img)
            labeled_elements["part_image_pairs"].append((pc, img))

        if self._debug_enabled and log.isEnabledFor(logging.DEBUG):
            unmatched_c = [pc for pc in part_counts if id(pc) not in matched_counts]
            unmatched_i = [im for im in images if id(im) not in matched_images]
            if unmatched_c:
                log.debug("[part_image] unmatched part_counts: %d", len(unmatched_c))
            if unmatched_i:
                log.debug("[part_image] unmatched images: %d", len(unmatched_i))

    def _build_candidate_edges(
        self, part_counts: List[Text], images: List[Image], page_width: float
    ) -> List[Tuple[float, Text, Image]]:
        VERT_EPS = 2.0  # allow minor overlap/touching
        ALIGN_EPS = max(2.0, 0.02 * page_width)

        edges: List[Tuple[float, Text, Image]] = []
        for pc in part_counts:
            cb = pc.bbox
            for img in images:
                ib = img.bbox
                if ib.y1 <= cb.y0 + VERT_EPS and abs(ib.x0 - cb.x0) <= ALIGN_EPS:
                    distance = max(0.0, cb.y0 - ib.y1)
                    edges.append((distance, pc, img))
        return edges

    def _get_images_in_parts_lists(
        self, page_data: PageData, parts_lists: List[Drawing]
    ) -> List[Image]:
        def inside_any_parts_list(img: Image) -> bool:
            return any(img.bbox.fully_inside(pl.bbox) for pl in parts_lists)

        return [
            e
            for e in page_data.elements
            if isinstance(e, Image) and inside_any_parts_list(e)
        ]

    def classify(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
        to_remove: Set[int],
    ) -> None:
        part_counts: List[Text] = labeled_elements.get("part_count", [])
        parts_lists: List[Drawing] = labeled_elements.get("parts_list", [])
        if not part_counts or not parts_lists:
            return

        images = self._get_images_in_parts_lists(page_data, parts_lists)
        if not images:
            return

        page_width = (
            (page_data.bbox.x1 - page_data.bbox.x0) if page_data.bbox else 100.0
        )

        edges = self._build_candidate_edges(part_counts, images, page_width)
        if not edges:
            return

        self._match_and_label_parts(edges, part_counts, images, labeled_elements)
