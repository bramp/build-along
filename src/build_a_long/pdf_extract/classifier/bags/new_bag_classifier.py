"""
New bag classifier.

Purpose
-------
Identify "New Bag" elements on LEGO instruction pages. A NewBag element
consists of a bag number (large text) surrounded by a cluster of images
forming a bag icon graphic. This typically appears at the top-left of
a page when a new numbered bag of pieces should be opened.

Heuristic
---------
- Requires a BagNumber element (classified by BagNumberClassifier)
- Look for clusters of Image/Drawing blocks around the bag number
- The cluster should form a cohesive visual element
- Typically located in upper-left portion of the page

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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

    image_cluster_score: float
    """Score based on number of surrounding images (0.0-1.0)."""

    compactness_score: float
    """Score based on how compact the cluster is (0.0-1.0)."""

    bag_number_candidate: Candidate
    """The bag number candidate for this new bag."""

    cluster_bbox: BBox
    """Bounding box encompassing the entire bag cluster."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        # Equal weighting for both components
        score = (self.image_cluster_score + self.compactness_score) / 2.0
        return score


# TODO The 40573 set, has a new bag graphic that does not include a number.
# Indicating you open all bugs
@dataclass(frozen=True)
class NewBagClassifier(LabelClassifier):
    """Classifier for new bag elements."""

    output = "new_bag"
    requires = frozenset({"bag_number"})

    def _score(self, result: ClassificationResult) -> None:
        """Score bag number + image clusters and create candidates."""
        page_data = result.page_data

        # Get bag number candidates (not constructed elements)
        bag_number_candidates = result.get_scored_candidates(
            "bag_number", valid_only=False, exclude_failed=True
        )
        if not bag_number_candidates:
            return

        # Get all image/drawing blocks on the page, filtering out large
        # images likely to be backgrounds
        max_image_width = page_data.bbox.width * 0.5
        max_image_height = page_data.bbox.height * 0.5
        image_blocks = [
            block
            for block in page_data.blocks
            if isinstance(block, Drawing | Image)
            and (
                block.bbox.width < max_image_width
                and block.bbox.height < max_image_height
            )
        ]

        if not image_blocks:
            return

        log.debug(
            "[new_bag] page=%s bag_number_candidates=%d image_blocks=%d",
            page_data.page_number,
            len(bag_number_candidates),
            len(image_blocks),
        )

        # For each bag number candidate, try to find a surrounding cluster of images
        for bag_number_candidate in bag_number_candidates:
            # Bag number not constructed yet, so we use the candidate info
            bag_bbox = bag_number_candidate.bbox

            # Find nearby images (within a reasonable distance)
            nearby_images = self._find_nearby_images(bag_bbox, image_blocks)

            if not nearby_images:
                # No images nearby, skip this bag number
                log.debug(
                    "[new_bag] No images near bag_number candidate at %s",
                    bag_bbox,
                )
                continue

            # Score the cluster
            cluster_score = self._score_image_cluster(len(nearby_images))
            compactness_score = self._score_compactness(bag_bbox, nearby_images)

            # Calculate bounding box that includes the bag number and all nearby images
            cluster_bbox = self._calculate_cluster_bbox(bag_bbox, nearby_images)

            score_details = _NewBagScore(
                image_cluster_score=cluster_score,
                compactness_score=compactness_score,
                bag_number_candidate=bag_number_candidate,
                cluster_bbox=cluster_bbox,
            )

            combined = score_details.score()

            # Add the candidate
            result.add_candidate(
                Candidate(
                    bbox=cluster_bbox,
                    label="new_bag",
                    score=combined,
                    score_details=score_details,
                    # TODO Update this to reference all the images that make up
                    # the new_bag graphic (but not the text block referenced by BagNumber)
                    source_blocks=[],  # Synthetic element
                ),
            )

            log.debug(
                "[new_bag] candidate images=%d cluster_score=%.2f "
                "compactness_score=%.2f combined=%.2f bbox=%s",
                len(nearby_images),
                cluster_score,
                compactness_score,
                combined,
                cluster_bbox,
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> NewBag:
        """Construct a NewBag element from a single candidate."""
        # Get score details
        detail_score = candidate.score_details
        assert isinstance(detail_score, _NewBagScore)

        # Validate and extract the bag number from parent candidate
        bag_number_candidate = detail_score.bag_number_candidate

        # Construct bag number
        bag_number_elem = result.build(bag_number_candidate)
        assert isinstance(bag_number_elem, BagNumber)
        bag_number = bag_number_elem

        # Construct the NewBag element
        return NewBag(bbox=detail_score.cluster_bbox, number=bag_number)

    def _find_nearby_images(
        self, bag_bbox: BBox, candidate_images: list[Drawing | Image]
    ) -> list[Drawing | Image]:
        """Find image/drawing blocks that form a connected cluster with the bag number.

        The new bag graphic is composed of many small overlapping images.
        This method:
        1. Filters out problematic elements (too large, invalid coordinates)
        2. Finds images that overlap the bag number
        3. Uses build_connected_cluster to find all connected images

        Args:
            bag_bbox: Bounding box of the bag number
            candidate_images: All eligible image/drawing blocks on the page

        Returns:
            List of images in the connected cluster
        """
        if not candidate_images:
            return []

        # Find initial images that overlap the bag number
        seed_images = [img for img in candidate_images if img.bbox.overlaps(bag_bbox)]

        if not seed_images:
            return []

        log.debug(
            "[new_bag] Found %d seed images overlapping bag bbox",
            len(seed_images),
        )

        # Build connected cluster starting from seed images
        cluster = build_connected_cluster(seed_images, candidate_images)

        log.debug(
            "[new_bag] Connected cluster has %d images (from %d seeds)",
            len(cluster),
            len(seed_images),
        )

        return cluster

    def _score_image_cluster(self, num_images: int) -> float:
        """Score based on the number of images in the cluster.

        NewBag elements typically have multiple images forming the bag graphic.

        Args:
            num_images: Number of nearby images

        Returns:
            Score from 0.0 to 1.0
        """
        if num_images == 0:
            return 0.0

        # TODO Re-evaluate this logic. I'm not sure ~8 is a ideal number.
        # Favor clusters with 3-8 images
        # Fewer than 3 might not be a complete bag graphic
        # More than 8 might be overlapping with other elements
        if num_images < 3:
            return 0.3 + (num_images / 3.0) * 0.3  # 0.3-0.6
        elif num_images <= 8:
            return 1.0
        else:
            # Penalize very large clusters
            return max(0.5, 1.0 - (num_images - 8) / 10.0)

    def _score_compactness(
        self, bag_bbox: BBox, images: list[Drawing | Image]
    ) -> float:
        """Score based on how compact the cluster is.

        A good NewBag cluster should have the bag number taking up
        roughly 1/4 of the height of the overall circular area.

        Args:
            bag_bbox: Bounding box of the bag number
            images: List of nearby images

        Returns:
            Score from 0.0 to 1.0
        """
        if not images:
            return 0.0

        cluster_bbox = self._calculate_cluster_bbox(bag_bbox, images)

        # Calculate the ratio of bag number height to cluster height
        bag_height = bag_bbox.height
        cluster_height = cluster_bbox.height

        if cluster_height == 0:
            return 0.0

        # Ideal ratio: bag number is ~25% (1/4) of cluster height
        height_ratio = bag_height / cluster_height

        # Score based on how close we are to the ideal 0.25 ratio
        if 0.2 <= height_ratio <= 0.35:
            # Very close to ideal (20-35%)
            return 1.0
        elif 0.15 <= height_ratio < 0.2 or 0.35 < height_ratio <= 0.45:
            # Somewhat close (15-20% or 35-45%)
            return 0.8
        elif 0.1 <= height_ratio < 0.15 or 0.45 < height_ratio <= 0.6:
            # Getting farther from ideal
            return 0.5
        else:
            # Too far from ideal ratio
            return max(0.0, 0.3 - abs(height_ratio - 0.25))

    def _calculate_cluster_bbox(
        self, bag_bbox: BBox, images: list[Drawing | Image]
    ) -> BBox:
        """Calculate the bounding box encompassing the bag number and images.

        Args:
            bag_bbox: Bounding box of the bag number
            images: List of nearby images

        Returns:
            Bounding box covering the entire cluster
        """
        all_bboxes = [bag_bbox] + [img.bbox for img in images]

        # Calculate the union of all bounding boxes
        min_x = min(bbox.x0 for bbox in all_bboxes)
        min_y = min(bbox.y0 for bbox in all_bboxes)
        max_x = max(bbox.x1 for bbox in all_bboxes)
        max_y = max(bbox.y1 for bbox in all_bboxes)

        return BBox(x0=min_x, y0=min_y, x1=max_x, y1=max_y)
