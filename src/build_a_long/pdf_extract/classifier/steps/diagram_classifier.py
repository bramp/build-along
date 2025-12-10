"""
Diagram classifier.

Purpose
-------
Identify diagram regions on instruction pages. Diagrams are any images or
drawings on the page, distinct from PartImages (which are single LEGO pieces).
Sometimes a diagram is split into multiple smaller images that are positioned
next to each other, so we cluster them together.

Heuristic
---------
- Look for Image elements on the page
- Filter out full-page images (> 90% of page area, likely backgrounds or borders)
- Each Image becomes a diagram candidate (no clustering during scoring)
- During build(), expand to include adjacent unclaimed images (lazy clustering)

Lazy Clustering
---------------
Clustering is deferred to build() time. This allows other classifiers (like
SubAssemblyClassifier) to claim images first. When build() is called:
1. Start with the candidate's source image
2. Find all adjacent/overlapping unclaimed images
3. Cluster them together into a single Diagram
4. Mark all clustered images as consumed

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
    build_connected_cluster,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
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
    # Area filtering threshold (as ratio of page area)
    MAX_AREA_RATIO: ClassVar[float] = (
        0.95  # Filter out images > 95% of page (backgrounds/borders)
    )

    def _score(self, result: ClassificationResult) -> None:
        """Score Image blocks and create one candidate per image.

        Clustering is deferred to build() time to allow other classifiers
        to claim images first.
        """
        page_data = result.page_data
        page_bbox = page_data.bbox
        assert page_bbox is not None

        arrow_candidates = result.get_scored_candidates(
            "arrow", valid_only=False, exclude_failed=True
        )

        # Get all image blocks, filtering out full-page images
        image_blocks: list[Image] = []
        for block in page_data.blocks:
            # Only consider Image elements
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
                "[diagram] No image blocks found on page %s",
                page_data.page_number,
            )
            return

        log.debug(
            "[diagram] page=%s image_blocks=%d",
            page_data.page_number,
            len(image_blocks),
        )

        # Create one candidate per image (no clustering during scoring)
        for block in image_blocks:
            score_details = _DiagramScore(
                cluster_bbox=block.bbox,
                num_images=1,
            )

            candidate = Candidate(
                bbox=block.bbox,
                label="diagram",
                score=score_details.score(),
                score_details=score_details,
                source_blocks=[block],
            )
            result.add_candidate(candidate)

    def build(
        self,
        candidate: Candidate,
        result: ClassificationResult,
        constraint_bbox: BBox | None = None,
    ) -> Diagram:
        """Construct a Diagram element with lazy clustering.

        Starting from the candidate's source image, expands to include all
        adjacent/overlapping unclaimed images, clustering them into a single
        Diagram.

        Args:
            candidate: The diagram candidate to build
            result: The classification result context
            constraint_bbox: Optional bounding box to constrain clustering.
                If provided, only images fully contained within this bbox
                will be included in the cluster. This is useful when building
                diagrams for subassemblies to prevent clustering beyond the
                subassembly bounds.
        """
        page_bbox = result.page_data.bbox
        assert page_bbox is not None

        # Start with the candidate's source block
        assert len(candidate.source_blocks) == 1
        seed_block = candidate.source_blocks[0]
        assert isinstance(seed_block, Image)

        # Find all unclaimed images that can be clustered with this one
        clustered_blocks = self._expand_cluster(seed_block, result, constraint_bbox)

        # Calculate the combined bbox
        cluster_bbox = BBox.union_all([b.bbox for b in clustered_blocks])

        # Clip diagram bbox to page bounds (and constraint if provided)
        diagram_bbox = cluster_bbox.clip_to(page_bbox)
        if constraint_bbox:
            # TODO Do we need this? this would indicate a problem in clustering
            diagram_bbox = diagram_bbox.clip_to(constraint_bbox)

        # Update the candidate's source_blocks to include all clustered blocks
        # This ensures they all get marked as consumed
        candidate.source_blocks = list(clustered_blocks)

        log.debug(
            "[diagram] Building diagram at %s (clustered %d images%s)",
            diagram_bbox,
            len(clustered_blocks),
            f", constrained to {constraint_bbox}" if constraint_bbox else "",
        )

        return Diagram(bbox=diagram_bbox)

    def _expand_cluster(
        self,
        seed_block: Image,
        result: ClassificationResult,
        constraint_bbox: BBox | None = None,
    ) -> list[Image]:
        """Expand from a seed image to include all adjacent unclaimed images.

        Uses flood-fill to find all images that are adjacent/overlapping
        and not yet consumed by another classifier.

        Args:
            seed_block: The starting image block
            result: Classification result to check consumed blocks
            constraint_bbox: Optional bounding box to constrain clustering.
                If provided, only images fully contained within this bbox
                will be considered for clustering.

        Returns:
            List of all images in the cluster (including seed)
        """
        # Get all unclaimed image blocks on the page
        log.debug(
            "[diagram] _expand_cluster: seed=%d at %s, consumed_blocks=%s%s",
            seed_block.id,
            seed_block.bbox,
            sorted(result._consumed_blocks),
            f", constraint={constraint_bbox}" if constraint_bbox else "",
        )
        unclaimed_images: list[Image] = []
        for block in result.page_data.blocks:
            if not isinstance(block, Image):
                continue
            # Skip if already consumed
            if block.id in result._consumed_blocks:
                log.debug(
                    "[diagram] Skipping consumed image id=%d at %s",
                    block.id,
                    block.bbox,
                )
                continue
            # Skip if outside constraint bbox
            if constraint_bbox and not constraint_bbox.contains(block.bbox):
                log.debug(
                    "[diagram] Skipping image id=%d at %s (outside constraint %s)",
                    block.id,
                    block.bbox,
                    constraint_bbox,
                )
                continue
            unclaimed_images.append(block)

        if seed_block not in unclaimed_images:
            # Seed was already consumed (shouldn't happen, but be safe)
            return [seed_block]

        # Build cluster from unclaimed images starting at seed
        return build_connected_cluster(seed_block, unclaimed_images)
