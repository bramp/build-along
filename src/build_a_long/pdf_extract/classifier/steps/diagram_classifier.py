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

Re-scoring
----------
When a diagram's source blocks conflict with another candidate (e.g., an arrow
that claims part of the diagram), the diagram can be re-scored without those
blocks. If the remaining blocks still form a valid diagram (meets minimum area),
a reduced candidate is created instead of failing entirely.

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
    Blocks,
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
    requires = frozenset(
        {
            # Arrows typically overlap diagrams - so we exclude them upfront
            "arrow",
        }
    )

    # TODO Convert to configurable parameters
    # Area filtering thresholds (as ratio of page area)
    MIN_AREA_RATIO: ClassVar[float] = (
        0.03  # Filter out images < 3% of page (decorative elements)
    )
    MAX_AREA_RATIO: ClassVar[float] = (
        0.95  # Filter out images > 95% of page (backgrounds/borders)
    )

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing/Image clusters and create candidates."""
        page_data = result.page_data
        page_bbox = page_data.bbox
        assert page_bbox is not None

        arrow_candidates = result.get_scored_candidates(
            "arrow", valid_only=False, exclude_failed=True
        )

        # Get all image/drawing blocks, filtering out full-page images
        image_blocks: list[Drawing | Image] = []
        for block in page_data.blocks:
            # Only consider Drawing and Image elements
            if not isinstance(block, Image):
                continue

            # Skip if part of an arrow's source blocks
            if any(block in arrow.source_blocks for arrow in arrow_candidates):
                continue

            # Filter based on area relative to page
            area_ratio = block.bbox.area / page_bbox.area

            # Skip full-page images (> 95% of page area)
            # These are likely borders/backgrounds
            # TODO This may not be necessary as we filter out all background
            # blocks a lot earlier
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

        # Create a candidate for each cluster (with no exclusions)
        for cluster in clusters:
            candidate = self._create_candidate_from_blocks(
                list(cluster), excluded_block_ids=set(), result=result
            )
            if candidate:
                result.add_candidate(candidate)

    def rescore_without_blocks(
        self,
        candidate: Candidate,
        excluded_block_ids: set[int],
        result: ClassificationResult,
    ) -> Candidate | None:
        """Create a new diagram candidate excluding specified blocks.

        Args:
            candidate: The original candidate to re-score
            excluded_block_ids: Set of block IDs to exclude
            result: The classification result context

        Returns:
            A new candidate without the excluded blocks, or None if the
            candidate is no longer valid without those blocks.
        """
        # Filter out excluded blocks
        remaining_blocks = [
            b for b in candidate.source_blocks if b.id not in excluded_block_ids
        ]

        if not remaining_blocks:
            return None

        return self._create_candidate_from_blocks(
            remaining_blocks, excluded_block_ids, result
        )

    def _create_candidate_from_blocks(
        self,
        blocks: list[Blocks],
        excluded_block_ids: set[int],
        result: ClassificationResult,
    ) -> Candidate | None:
        """Create a diagram candidate from a list of blocks.

        Args:
            blocks: The blocks to include in the diagram
            excluded_block_ids: Block IDs that were excluded (for logging)
            result: The classification result context

        Returns:
            A Candidate if the blocks form a valid diagram, None otherwise
        """
        if not blocks:
            return None

        page_bbox = result.page_data.bbox
        assert page_bbox is not None

        cluster_bbox = self._calculate_cluster_bbox(blocks)

        # Filter based on area relative to page
        area_ratio = cluster_bbox.area / page_bbox.area

        # Skip very small images (< 3% of page area)
        # These are likely decorative elements or noise
        if area_ratio < self.MIN_AREA_RATIO:
            if excluded_block_ids:
                log.debug(
                    "[diagram] Reduced diagram too small after excluding blocks: "
                    "area_ratio=%.3f < %.3f",
                    area_ratio,
                    self.MIN_AREA_RATIO,
                )
            return None

        # Skip full-page images (> 90% of page area)
        # These are likely borders/backgrounds
        if area_ratio > self.MAX_AREA_RATIO:
            return None

        score_details = _DiagramScore(
            cluster_bbox=cluster_bbox,
            num_images=len(blocks),
        )

        log.debug(
            "[diagram] %s images=%d bbox=%s",
            "Reduced cluster" if excluded_block_ids else "cluster",
            len(blocks),
            cluster_bbox,
        )

        return Candidate(
            bbox=cluster_bbox,
            label="diagram",
            score=score_details.score(),
            score_details=score_details,
            source_blocks=list(blocks),
        )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Diagram:
        """Construct a Diagram element from a single candidate."""
        # Get the cluster bbox from score details
        score_details = candidate.score_details
        assert isinstance(score_details, _DiagramScore)

        # Clip diagram bbox to page bounds (arrows may extend slightly off-page)
        page_bbox = result.page_data.bbox
        diagram_bbox = score_details.cluster_bbox.clip_to(page_bbox)

        # Find arrows that overlap with or are inside this diagram
        arrows: list[Arrow] = []
        for arrow_candidate in result.get_scored_candidates(
            "arrow", valid_only=False, exclude_failed=True
        ):
            # Skip if arrow was marked as failed during iteration
            # (can happen when one arrow's build marks others as conflicting)
            if arrow_candidate.failure_reason:
                log.debug(
                    "[diagram] Skipping failed arrow at %s: %s",
                    arrow_candidate.bbox,
                    arrow_candidate.failure_reason,
                )
                continue

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

    def _calculate_cluster_bbox(self, cluster: list[Blocks]) -> BBox:
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
