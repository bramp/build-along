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

TODO The bounding box calculation is currently incorrect and needs fixing.

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    BagNumber,
    LegoPageElements,
    NewBag,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


@dataclass
class _NewBagScore:
    """Internal score representation for new bag classification."""

    image_cluster_score: float
    """Score based on number of surrounding images (0.0-1.0)."""

    compactness_score: float
    """Score based on how compact the cluster is (0.0-1.0)."""

    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components."""
        # Equal weighting for both components
        score = (self.image_cluster_score + self.compactness_score) / 2.0
        return score


@dataclass(frozen=True)
class NewBagClassifier(LabelClassifier):
    """Classifier for new bag elements."""

    outputs = frozenset({"new_bag"})
    requires = frozenset({"bag_number"})

    def score(self, result: ClassificationResult) -> None:
        """Legacy classifier - uses evaluate() instead of score() + construct()."""
        raise NotImplementedError(
            f"{self.__class__.__name__} uses legacy evaluate() method. "
            "Implement score() and construct() to use two-phase classification."
        )

    def construct(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Legacy classifier - uses evaluate() instead of score() + construct()."""
        raise NotImplementedError(
            f"{self.__class__.__name__} uses legacy evaluate() method. "
            "Implement score() and construct() to use two-phase classification."
        )

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for new bag elements.

        This method looks for bag numbers surrounded by clusters of images.
        """
        page_data = result.page_data

        # Get bag number candidates
        bag_numbers = result.get_winners_by_score("bag_number", BagNumber)
        if not bag_numbers:
            return

        # Get all image/drawing blocks on the page
        image_blocks = [
            block for block in page_data.blocks if isinstance(block, Drawing | Image)
        ]

        if not image_blocks:
            return

        log.debug(
            "[new_bag] page=%s bag_numbers=%d image_blocks=%d",
            page_data.page_number,
            len(bag_numbers),
            len(image_blocks),
        )

        # For each bag number, try to find a surrounding cluster of images
        for bag_number in bag_numbers:
            bag_bbox = bag_number.bbox

            # Find nearby images (within a reasonable distance)
            nearby_images = self._find_nearby_images(bag_bbox, image_blocks)

            if not nearby_images:
                # No images nearby, skip this bag number
                log.debug(
                    "[new_bag] No images near bag_number value=%d at %s",
                    bag_number.value,
                    bag_bbox,
                )
                continue

            # Score the cluster
            cluster_score = self._score_image_cluster(len(nearby_images))
            compactness_score = self._score_compactness(bag_bbox, nearby_images)

            score_details = _NewBagScore(
                image_cluster_score=cluster_score,
                compactness_score=compactness_score,
            )

            combined = score_details.combined_score(self.config)

            # Calculate bounding box that includes the bag number and all
            # nearby images
            cluster_bbox = self._calculate_cluster_bbox(bag_bbox, nearby_images)

            # Clip to page bounds to avoid extending beyond the page
            page_bbox = page_data.bbox
            assert page_bbox is not None
            cluster_bbox = cluster_bbox.clip_to(page_bbox)

            # Construct the NewBag element
            constructed_elem = NewBag(bbox=cluster_bbox, number=bag_number)

            # Store candidate
            result.add_candidate(
                "new_bag",
                Candidate(
                    bbox=cluster_bbox,
                    label="new_bag",
                    score=combined,
                    score_details=score_details,
                    constructed=constructed_elem,
                    source_blocks=[],  # Synthetic element
                    failure_reason=None,
                ),
            )

            log.debug(
                "[new_bag] candidate bag=%d images=%d cluster_score=%.2f "
                "compactness_score=%.2f combined=%.2f bbox=%s",
                bag_number.value,
                len(nearby_images),
                cluster_score,
                compactness_score,
                combined,
                cluster_bbox,
            )

    def _find_nearby_images(
        self, bag_bbox: BBox, image_blocks: list[Drawing | Image]
    ) -> list[Drawing | Image]:
        """Find image/drawing blocks near the bag number.

        An image is considered "nearby" if it overlaps with or is within
        a reasonable distance of the bag number's bounding box.

        Args:
            bag_bbox: Bounding box of the bag number
            image_blocks: All image/drawing blocks on the page

        Returns:
            List of nearby image blocks
        """
        # Expand the bag bbox to include nearby images
        # Typical "New Bag" graphics extend significantly around the number
        expansion_factor = 3.0  # Allow images within 3x the bag bbox size
        expansion_width = bag_bbox.width * expansion_factor
        expansion_height = bag_bbox.height * expansion_factor

        # Create expanded search area
        search_bbox = BBox(
            x0=bag_bbox.x0 - expansion_width,
            y0=bag_bbox.y0 - expansion_height,
            x1=bag_bbox.x1 + expansion_width,
            y1=bag_bbox.y1 + expansion_height,
        )

        nearby = []
        for img in image_blocks:
            # Check if image overlaps with or is contained in search area
            if img.bbox.overlaps(search_bbox) or search_bbox.fully_inside(img.bbox):
                nearby.append(img)

        return nearby

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

        A good NewBag cluster should be relatively compact, with images
        clustered around the bag number rather than spread across the page.

        Args:
            bag_bbox: Bounding box of the bag number
            images: List of nearby images

        Returns:
            Score from 0.0 to 1.0
        """
        if not images:
            return 0.0

        cluster_bbox = self._calculate_cluster_bbox(bag_bbox, images)

        # Calculate compactness as the ratio of bag number size to cluster size
        # A compact cluster has the bag number taking up a reasonable portion
        bag_area = bag_bbox.area
        cluster_area = cluster_bbox.area

        if cluster_area == 0:
            return 0.0

        # Ideal ratio: bag number is 10-30% of total cluster
        ratio = bag_area / cluster_area

        if 0.1 <= ratio <= 0.3:
            return 1.0
        elif ratio < 0.1:
            # Cluster too large relative to bag number
            return max(0.0, ratio / 0.1)
        else:
            # Bag number too large relative to cluster (unusual)
            return max(0.5, 1.0 - (ratio - 0.3) / 0.7)

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
