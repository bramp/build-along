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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.types import (
        Candidate,
        ClassificationHints,
    )
    from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.types import (
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Element,
    Image,
    Text,
)

log = logging.getLogger(__name__)


@dataclass
class _PartImageScore:
    """Internal score representation for part image pairing."""

    distance: float
    """Vertical distance from part count text to image (lower is better)."""

    part_count: Text
    """The part count text element."""

    image: Image
    """The image element."""

    def sort_key(self) -> float:
        """Return sort key for matching (prefer smaller distance)."""
        return self.distance


class PartsImageClassifier(LabelClassifier):
    """Classifier for part images paired with part count texts."""

    outputs = {"part_image"}
    requires = {"parts_list", "part_count"}

    def __init__(self, config: ClassifierConfig, classifier):
        super().__init__(config, classifier)
        self._debug_enabled = os.getenv("CLASSIFIER_DEBUG", "").lower() in (
            "part_image",
            "all",
        )

    def calculate_scores(
        self,
        page_data: PageData,
        scores: Dict[str, Dict[Any, Any]],
        labeled_elements: Dict[Element, str],
    ) -> None:
        """Calculate scores for part image pairings.

        Scores are based on vertical distance and horizontal alignment between
        part count texts and images within parts lists.
        """

        part_counts: List[Text] = [
            e
            for e, label in labeled_elements.items()
            if label == "part_count" and isinstance(e, Text)
        ]
        parts_lists: List[Drawing] = [
            e
            for e, label in labeled_elements.items()
            if label == "parts_list" and isinstance(e, Drawing)
        ]
        if not part_counts or not parts_lists:
            return

        images = self._get_images_in_parts_lists(page_data, parts_lists)
        if not images:
            return

        page_width = (
            (page_data.bbox.x1 - page_data.bbox.x0) if page_data.bbox else 100.0
        )

        # Initialize scores dict for this classifier
        if "part_image" not in scores:
            scores["part_image"] = {}

        # Initialize storage for matched pairs (will be populated in classify())
        if "part_image_pairs" not in scores:
            scores["part_image_pairs"] = {}

        # Build candidate pairings and store them in scores dict
        self._build_candidate_edges(
            part_counts, images, page_width, scores["part_image"]
        )

    def _match_and_label_parts(
        self,
        edges: List[_PartImageScore],
        part_counts: List[Text],
        images: List[Image],
        labeled_elements: Dict[Element, str],
        scores: Dict[str, Dict[Any, Any]],
    ):
        """Match part counts with images using greedy matching based on distance.

        Stores the matched pairs in scores dict under 'part_image_pairs' so they
        can be retrieved by the classifier and passed to the builder.
        """
        edges.sort(key=lambda score: score.sort_key())
        matched_counts: Set[int] = set()
        matched_images: Set[int] = set()

        # Track pairs for later use - store in scores dict so classifier can retrieve them
        part_image_pairs = []

        for score in edges:
            pc = score.part_count
            img = score.image
            if id(pc) in matched_counts or id(img) in matched_images:
                continue
            matched_counts.add(id(pc))
            matched_images.add(id(img))
            # Label the image as part_image (only once per image)
            if labeled_elements.get(img) != "part_image":
                labeled_elements[img] = "part_image"
            part_image_pairs.append((pc, img))

        # CRITICAL: Store the pairs in scores dict so the main classifier
        # can extract them and put them in ClassificationResult
        # We use a special key "pairs" to distinguish from score objects
        scores["part_image_pairs"]["pairs"] = part_image_pairs

        if self._debug_enabled and log.isEnabledFor(logging.DEBUG):
            unmatched_c = [pc for pc in part_counts if id(pc) not in matched_counts]
            unmatched_i = [im for im in images if id(im) not in matched_images]
            if unmatched_c:
                log.debug("[part_image] unmatched part_counts: %d", len(unmatched_c))
            if unmatched_i:
                log.debug("[part_image] unmatched images: %d", len(unmatched_i))
        """Match part counts with images using greedy matching based on distance.
        
        Stores the matched pairs in labeled_elements under the special key
        'part_image_pairs' so they can be retrieved by the classifier and
        passed to the builder.
        """
        edges.sort(key=lambda score: score.sort_key())
        matched_counts: Set[int] = set()
        matched_images: Set[int] = set()

        # Track pairs for later use - store in labeled_elements so classifier can retrieve them
        part_image_pairs = []

        for score in edges:
            pc = score.part_count
            img = score.image
            if id(pc) in matched_counts or id(img) in matched_images:
                continue
            matched_counts.add(id(pc))
            matched_images.add(id(img))
            # Label the image as part_image (only once per image)
            if labeled_elements.get(img) != "part_image":
                labeled_elements[img] = "part_image"
            part_image_pairs.append((pc, img))

        # We use a special key "pairs" to distinguish from score objects
        scores["part_image_pairs"]["pairs"] = part_image_pairs

        if self._debug_enabled and log.isEnabledFor(logging.DEBUG):
            unmatched_c = [pc for pc in part_counts if id(pc) not in matched_counts]
            unmatched_i = [im for im in images if id(im) not in matched_images]
            if unmatched_c:
                log.debug("[part_image] unmatched part_counts: %d", len(unmatched_c))
            if unmatched_i:
                log.debug("[part_image] unmatched images: %d", len(unmatched_i))

    def _build_candidate_edges(
        self,
        part_counts: List[Text],
        images: List[Image],
        page_width: float,
        part_image_scores: Dict[Any, Any],
    ) -> List[_PartImageScore]:
        """Build candidate pairings between part counts and images.

        Returns a list of score objects representing valid pairings.
        """
        VERT_EPS = 2.0  # allow minor overlap/touching
        ALIGN_EPS = max(2.0, 0.02 * page_width)

        edges: List[_PartImageScore] = []
        for pc in part_counts:
            cb = pc.bbox
            for img in images:
                ib = img.bbox
                if ib.y1 <= cb.y0 + VERT_EPS and abs(ib.x0 - cb.x0) <= ALIGN_EPS:
                    distance = max(0.0, cb.y0 - ib.y1)
                    score = _PartImageScore(
                        distance=distance,
                        part_count=pc,
                        image=img,
                    )
                    edges.append(score)
                    # Store in scores dict (using a tuple key since multiple pairings)
                    key = (pc, img)
                    part_image_scores[key] = score
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
        scores: Dict[str, Dict[Any, Any]],
        labeled_elements: Dict[Element, str],
        removal_reasons: Dict[int, RemovalReason],
        hints: Optional["ClassificationHints"] = None,
        constructed_elements: Optional[Dict[Element, "LegoPageElement"]] = None,
        candidates: Optional[Dict[str, List["Candidate"]]] = None,
    ) -> None:
        part_counts: List[Text] = [
            e
            for e, label in labeled_elements.items()
            if label == "part_count" and isinstance(e, Text)
        ]
        parts_lists: List[Drawing] = [
            e
            for e, label in labeled_elements.items()
            if label == "parts_list" and isinstance(e, Drawing)
        ]
        if not part_counts or not parts_lists:
            return

        images = self._get_images_in_parts_lists(page_data, parts_lists)
        if not images:
            return

        # Retrieve pre-computed scores from scores dict (populated in calculate_scores)
        part_image_scores: Dict[Any, Any] = scores.get("part_image", {})
        edges = []
        for pc in part_counts:
            for img in images:
                key = (pc, img)
                score_obj = part_image_scores.get(key)
                if isinstance(score_obj, _PartImageScore):
                    edges.append(score_obj)

        if not edges:
            return

        self._match_and_label_parts(
            edges, part_counts, images, labeled_elements, scores
        )
