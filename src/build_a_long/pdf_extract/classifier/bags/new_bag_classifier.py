"""
New bag classifier.

Purpose
-------
Identify "New Bag" elements on LEGO instruction pages. A NewBag element
consists of an optional bag number (large text) surrounded by a cluster
of images forming a bag icon graphic. This typically appears at the
top-left of a page when a new numbered bag of pieces should be opened.

Some sets use a NewBag graphic without a number, indicating that all
bags should be opened.

Heuristic
---------
1. Find all large, square-ish image clusters in the top-left area of the page
2. Score each cluster based on size, aspect ratio, and position
3. Check if any BagNumber candidates are inside the cluster (bonus score)
4. Best-scoring cluster becomes the NewBag candidate

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import BBox, build_connected_cluster
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    BagNumber,
    NewBag,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


class _NewBagScore(Score):
    """Internal score representation for new bag classification."""

    size_score: float
    """Score based on cluster size (0.0-1.0)."""

    aspect_score: float
    """Score based on how square the cluster is (0.0-1.0)."""

    position_score: float
    """Score based on position in top-left area (0.0-1.0)."""

    has_bag_number: bool
    """Whether a BagNumber candidate was found inside the cluster."""

    bag_number_candidate: Candidate | None
    """The bag number candidate for this new bag, or None for numberless bags."""

    cluster_bbox: BBox
    """Bounding box encompassing the entire bag cluster."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        # Base score from cluster properties
        base_score = (self.size_score + self.aspect_score + self.position_score) / 3.0

        # Bonus for having a bag number (increases confidence significantly)
        if self.has_bag_number:
            # Boost score but cap at 1.0
            return min(1.0, base_score + 0.2)
        return base_score


class NewBagClassifier(LabelClassifier):
    """Classifier for new bag elements."""

    output = "new_bag"
    requires = frozenset({"bag_number"})

    def _score(self, result: ClassificationResult) -> None:
        """Find image clusters and score them as potential bag icons."""
        config = self.config
        page_data = result.page_data
        page_bbox = page_data.bbox

        # Get all image/drawing blocks on the page, filtering out large
        # images likely to be backgrounds
        max_image_width = page_bbox.width * 0.5
        max_image_height = page_bbox.height * 0.5
        image_blocks = [
            block
            for block in page_data.blocks
            if isinstance(block, Drawing | Image)
            and block.bbox.width < max_image_width
            and block.bbox.height < max_image_height
        ]

        if not image_blocks:
            return

        # Get bag number candidates
        bag_number_candidates = result.get_scored_candidates(
            "bag_number", valid_only=False, exclude_failed=True
        )

        log.debug(
            "[new_bag] page=%s bag_number_candidates=%d image_blocks=%d",
            page_data.page_number,
            len(bag_number_candidates),
            len(image_blocks),
        )

        # Find potential bag icon clusters
        # Start with large images as seeds and build connected clusters
        processed_images: set[int] = set()

        for seed_block in image_blocks:
            if id(seed_block) in processed_images:
                continue

            # Build a connected cluster starting from this image
            cluster = build_connected_cluster([seed_block], image_blocks)

            # Mark all images in this cluster as processed
            for img in cluster:
                processed_images.add(id(img))

            # Calculate cluster bounding box
            cluster_bbox = BBox.union_all([img.bbox for img in cluster])

            # Score the cluster
            score_details = self._score_cluster(
                cluster_bbox, bag_number_candidates, page_bbox
            )

            if score_details is None:
                continue

            assert (
                score_details.bag_number_candidate is None
                or score_details.bag_number_candidate.source_blocks not in cluster
            ), "NewBag Cluster source blocks should not include the NewBagNumber blocks"

            combined = score_details.score()

            # Skip low-scoring candidates
            if combined < config.new_bag.min_score:
                log.debug(
                    "[new_bag] Skipping cluster score=%.2f (below %.2f) bbox=%s",
                    combined,
                    config.new_bag.min_score,
                    cluster_bbox,
                )
                continue

            result.add_candidate(
                Candidate(
                    bbox=cluster_bbox,
                    label="new_bag",
                    score=combined,
                    score_details=score_details,
                    source_blocks=list(cluster),  # type: ignore[arg-type]
                ),
            )

            log.debug(
                "[new_bag] candidate images=%d size=%.2f aspect=%.2f "
                "position=%.2f has_number=%s score=%.2f bbox=%s",
                len(cluster),
                score_details.size_score,
                score_details.aspect_score,
                score_details.position_score,
                score_details.has_bag_number,
                combined,
                cluster_bbox,
            )

    def _score_cluster(
        self,
        cluster_bbox: BBox,
        bag_number_candidates: list[Candidate],
        page_bbox: BBox,
    ) -> _NewBagScore | None:
        """Score a cluster as a potential bag icon.

        Args:
            cluster_bbox: Bounding box of the image cluster.
            bag_number_candidates: All bag number candidates on the page.
            page_bbox: Page bounding box for position calculations.

        Returns:
            Score details, or None if cluster doesn't meet basic requirements.
        """
        nb_config = self.config.new_bag

        # Check minimum size
        if (
            cluster_bbox.width < nb_config.icon_min_size
            or cluster_bbox.height < nb_config.icon_min_size
        ):
            return None

        # Check position - must be in top-left area
        max_x = page_bbox.width * nb_config.icon_max_x_ratio
        max_y = page_bbox.height * nb_config.icon_max_y_ratio
        if cluster_bbox.x0 > max_x or cluster_bbox.y0 > max_y:
            return None

        # Score size (larger clusters up to ~240 are better)
        ideal_size = 240.0  # Based on observed bag icon sizes
        avg_size = (cluster_bbox.width + cluster_bbox.height) / 2.0
        size_score = min(1.0, avg_size / ideal_size)

        # Score aspect ratio (square is best)
        aspect = cluster_bbox.width / cluster_bbox.height if cluster_bbox.height else 0
        min_aspect = nb_config.icon_min_aspect
        max_aspect = nb_config.icon_max_aspect

        if min_aspect <= aspect <= max_aspect:
            # Within acceptable range - score based on how close to 1.0
            aspect_score = 1.0 - abs(aspect - 1.0) * 2.0
            aspect_score = max(0.0, aspect_score)
        else:
            # Outside acceptable range
            return None

        # Score position (closer to top-left corner is better)
        x_ratio = cluster_bbox.x0 / page_bbox.width if page_bbox.width else 1.0
        y_ratio = cluster_bbox.y0 / page_bbox.height if page_bbox.height else 1.0
        position_score = 1.0 - (x_ratio + y_ratio) / 2.0

        # Check for bag number inside cluster
        bag_number_candidate = self._find_bag_number_in_cluster(
            cluster_bbox, bag_number_candidates
        )

        return _NewBagScore(
            size_score=size_score,
            aspect_score=aspect_score,
            position_score=position_score,
            has_bag_number=bag_number_candidate is not None,
            bag_number_candidate=bag_number_candidate,
            cluster_bbox=cluster_bbox,
        )

    def _find_bag_number_in_cluster(
        self, cluster_bbox: BBox, bag_number_candidates: list[Candidate]
    ) -> Candidate | None:
        """Find a bag number candidate that is inside the cluster bbox.

        Args:
            cluster_bbox: Bounding box of the cluster.
            bag_number_candidates: All bag number candidates on the page.

        Returns:
            The best bag number candidate inside the cluster, or None.
        """
        best_candidate = None
        best_score = 0.0

        for candidate in bag_number_candidates:
            # Check if the bag number is inside the cluster
            if cluster_bbox.contains(candidate.bbox) and candidate.score > best_score:
                best_candidate = candidate
                best_score = candidate.score

        return best_candidate

    def build(self, candidate: Candidate, result: ClassificationResult) -> NewBag:
        """Construct a NewBag element from a single candidate."""
        detail_score = candidate.score_details
        assert isinstance(detail_score, _NewBagScore)

        # Construct bag number if present
        bag_number: BagNumber | None = None
        if detail_score.bag_number_candidate is not None:
            bag_number_elem = result.build(detail_score.bag_number_candidate)
            assert isinstance(bag_number_elem, BagNumber)
            bag_number = bag_number_elem

        return NewBag(bbox=detail_score.cluster_bbox, number=bag_number)
