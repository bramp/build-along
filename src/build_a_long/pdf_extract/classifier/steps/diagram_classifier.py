"""
Diagram classifier.

Purpose
-------
Identify diagram regions on instruction pages. Diagrams are any images or
drawings on the page, distinct from PartImages (which are single LEGO pieces).
Sometimes a diagram is split into multiple smaller images that are positioned
next to each other, so we cluster them together (similar to how NewBagClassifier
clusters bag graphics).

Heuristic
---------
- Look for Drawing/Image elements on the page
- Filter out very small images (< 3% of page area)
- Filter out full-page images (> 90% of page area, likely backgrounds or borders)
- Cluster adjacent/overlapping images into single diagrams
- Each cluster becomes a diagram candidate

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging

from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    build_all_connected_clusters,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Arrow,
    Diagram,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


class _DiagramScore(Score):
    """Internal score representation for diagram classification."""

    cluster_bbox: BBox
    """Bounding box encompassing the entire diagram cluster."""

    num_images: int
    """Number of images/drawings in this cluster."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        # All diagram clusters get score of 1.0
        # Filtering happens in _score() method
        return 1.0


class DiagramClassifier(LabelClassifier):
    """Classifier for diagram regions on instruction pages."""

    output = "diagram"
    requires = frozenset({"progress_bar", "arrow"})

    # TODO Convert to configurable parameters
    # Area filtering thresholds (as ratio of page area)
    MIN_AREA_RATIO: ClassVar[float] = (
        0.03  # Filter out images < 3% of page (decorative elements)
    )
    MAX_AREA_RATIO: ClassVar[float] = (
        0.90  # Filter out images > 90% of page (backgrounds/borders)
    )

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing/Image clusters and create candidates."""
        page_data = result.page_data
        page_bbox = page_data.bbox
        assert page_bbox is not None

        # Get progress bar bbox to filter out overlapping elements
        progress_bar_bbox = self._get_progress_bar_bbox(result)

        # Get all image/drawing blocks, filtering out full-page images
        image_blocks: list[Drawing | Image] = []
        for block in page_data.blocks:
            # Only consider Drawing and Image elements
            if not isinstance(block, Drawing | Image):
                continue

            # Skip if overlaps with progress bar
            if progress_bar_bbox and block.bbox.overlaps(progress_bar_bbox):
                continue

            # Filter based on area relative to page
            area_ratio = block.bbox.area / page_bbox.area

            # Skip full-page images (> 90% of page area)
            # These are likely borders/backgrounds
            if area_ratio > self.MAX_AREA_RATIO:
                continue

            image_blocks.append(block)

        if not image_blocks:
            log.debug(
                "[diagram] No image/drawing blocks found on page %s",
                page_data.page_number,
            )
            return

        log.debug(
            "[diagram] page=%s image_blocks=%d",
            page_data.page_number,
            len(image_blocks),
        )

        # Build clusters of connected images
        clusters = build_all_connected_clusters(image_blocks)

        log.debug(
            "[diagram] Found %d diagram clusters",
            len(clusters),
        )

        # Create a candidate for each cluster
        for cluster in clusters:
            cluster_bbox = self._calculate_cluster_bbox(cluster)

            # Filter based on area relative to page
            area_ratio = cluster_bbox.area / page_bbox.area

            # Skip very small images (< 3% of page area)
            # These are likely decorative elements or noise
            if area_ratio < self.MIN_AREA_RATIO:
                continue

            # Skip full-page images (> 90% of page area)
            # These are likely borders/backgrounds
            if area_ratio > self.MAX_AREA_RATIO:
                continue

            score_details = _DiagramScore(
                cluster_bbox=cluster_bbox,
                num_images=len(cluster),
            )

            result.add_candidate(
                Candidate(
                    bbox=cluster_bbox,
                    label="diagram",
                    score=score_details.score(),
                    score_details=score_details,
                    source_blocks=list(cluster),
                ),
            )

            log.debug(
                "[diagram] cluster images=%d bbox=%s",
                len(cluster),
                cluster_bbox,
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Diagram:
        """Construct a Diagram element from a single candidate."""
        # Get the cluster bbox from score details
        score_details = candidate.score_details
        assert isinstance(score_details, _DiagramScore)

        diagram_bbox = score_details.cluster_bbox

        # Find arrows that overlap with or are inside this diagram
        arrows: list[Arrow] = []
        for arrow_candidate in result.get_scored_candidates(
            "arrow", valid_only=False, exclude_failed=True
        ):
            # Check if arrow overlaps with or is inside the diagram bbox
            if arrow_candidate.bbox.overlaps(diagram_bbox):
                arrow_elem = result.build(arrow_candidate)
                assert isinstance(arrow_elem, Arrow)
                arrows.append(arrow_elem)

        log.debug(
            "[diagram] Building diagram at %s with %d arrows",
            diagram_bbox,
            len(arrows),
        )

        return Diagram(bbox=diagram_bbox, arrows=arrows)

    def _get_progress_bar_bbox(self, result: ClassificationResult) -> BBox | None:
        """Get the bounding box of the progress bar if present.

        Returns:
            BBox of the progress bar, or None if not found.
        """
        progress_bar_candidates = result.get_scored_candidates(
            "progress_bar", valid_only=False, exclude_failed=True
        )

        # Return the first progress bar candidate's bbox
        if progress_bar_candidates:
            return progress_bar_candidates[0].bbox

        return None

    def _calculate_cluster_bbox(self, cluster: list[Drawing | Image]) -> BBox:
        """Calculate the bounding box encompassing all images in the cluster.

        Args:
            cluster: List of images in the cluster

        Returns:
            Bounding box covering the entire cluster
        """
        if not cluster:
            raise ValueError("Cannot calculate bbox for empty cluster")

        bboxes = [img.bbox for img in cluster]
        return BBox.union_all(bboxes)
