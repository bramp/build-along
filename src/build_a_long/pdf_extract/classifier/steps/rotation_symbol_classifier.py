"""
Rotation symbol classifier.

Purpose
-------
Identify rotation symbols on LEGO instruction pages. These symbols indicate
that the builder should rotate the assembled model. They appear as small,
isolated, square clusters of Drawing elements (~46px).

Heuristic
---------
1. Collect all Drawing blocks on the page
2. Build connected components (clusters) using bbox overlap
3. For each cluster, compute the union bbox
4. Score clusters based on intrinsic properties:
   - Square-ish (aspect ratio ~0.95-1.05)
   - Small (~41-51 pixels per side, ±10% of ideal 46px)
   - Black and white colors only (no colored drawings)


The key insight is that rotation symbols are vector drawings that are ISOLATED -
they don't overlap with nearby diagram elements. Images are excluded because
their bounding boxes may overlap with diagrams even when the visible content
(ignoring transparent areas) appears disconnected.

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.config import RotationSymbolConfig
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    build_all_connected_clusters,
    filter_overlapping,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    RotationSymbol,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


class _RotationSymbolScore(Score):
    """Internal score representation for rotation symbol classification."""

    size_score: float
    """Score based on size being in expected range (0.0-1.0)."""

    aspect_score: float
    """Score based on aspect ratio being square-ish (0.0-1.0)."""

    config: RotationSymbolConfig
    """Configuration containing weights for score calculation."""

    # Cap for intrinsic classifiers (0.8) to allow composites to score higher
    MAX_SCORE: float = 0.8

    def score(self) -> Weight:
        """Calculate final weighted score from components, capped at 0.8."""
        raw_score = (
            self.size_score * self.config.size_weight
            + self.aspect_score * self.config.aspect_weight
        )
        return raw_score * self.MAX_SCORE


class RotationSymbolClassifier(LabelClassifier):
    """Classifier for rotation symbol elements.

    Implementation Pattern: Clustering During Scoring
    --------------------------------------------------
    This classifier clusters Drawing blocks during the _score() phase, which is
    justified because:

    1. **Single Visual Element**: The rotation symbol is a single visual element
       composed of multiple Drawing blocks that must be considered together
    2. **Intrinsic Properties**: Clustering is based on spatial overlap (intrinsic
       geometric property), not relationships with other classified elements
    3. **Build-Time Discovery**: Related Images (shadows, reference diagrams) are
       discovered and claimed during build(), following best practices

    This pattern differs from typical atomic classifiers that score individual blocks,
    but is appropriate when multiple PDF blocks form a single logical element.
    """

    output = "rotation_symbol"
    requires = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        """Score connected clusters of Drawing blocks as rotation symbols."""
        page_data = result.page_data
        config = self.config
        page_bbox = page_data.bbox
        assert page_bbox is not None
        rs_config = config.rotation_symbol

        # Filter out page-spanning drawings (>90% of page width or height).
        # These are typically background/border elements that would connect
        # unrelated symbols together during clustering.
        max_width = page_bbox.width * 0.9
        max_height = page_bbox.height * 0.9

        # Filter out drawings larger than the max rotation symbol size.
        max_drawing_size = rs_config.max_size

        drawings: list[Drawing] = [
            block
            for block in page_data.blocks
            if isinstance(block, Drawing)
            and block.bbox.width <= max_width
            and block.bbox.height <= max_height
            and block.bbox.width <= max_drawing_size
            and block.bbox.height <= max_drawing_size
        ]

        if not drawings:
            return

        # Build connected components using bbox overlap
        clusters = build_all_connected_clusters(drawings)

        log.debug(
            "[rotation_symbol] Found %d clusters from %d drawings",
            len(clusters),
            len(drawings),
        )

        # Score each cluster
        for cluster in clusters:
            cluster_bbox = BBox.union_all([block.bbox for block in cluster])

            # Check that all drawings in the cluster are black/white
            if not self._is_black_and_white_cluster(cluster):
                log.debug(
                    "[rotation_symbol] Rejected cluster at %s "
                    "(%d blocks) - contains non-black/white colors",
                    cluster_bbox,
                    len(cluster),
                )
                continue

            score_details = self._score_bbox(cluster_bbox)
            if score_details is None:
                log.debug(
                    "[rotation_symbol] Rejected cluster at %s "
                    "(%d blocks) size=%.1fx%.1f - outside size/aspect constraints",
                    cluster_bbox,
                    len(cluster),
                    cluster_bbox.width,
                    cluster_bbox.height,
                )
                continue

            if score_details.score() <= config.rotation_symbol.min_score:
                log.debug(
                    "[rotation_symbol] Rejected cluster at %s "
                    "(%d blocks) score=%.2f < min_score=%.2f",
                    cluster_bbox,
                    len(cluster),
                    score_details.score(),
                    config.rotation_symbol.min_score,
                )
                continue

            result.add_candidate(
                Candidate(
                    bbox=cluster_bbox,
                    label="rotation_symbol",
                    score=score_details.score(),
                    score_details=score_details,
                    # Consume the Drawing blocks that make up this rotation symbol
                    source_blocks=list(cluster),
                )
            )
            log.debug(
                "[rotation_symbol] Cluster candidate at %s "
                "(%d blocks) score=%.2f (size=%.2f aspect=%.2f)",
                cluster_bbox,
                len(cluster),
                score_details.score(),
                score_details.size_score,
                score_details.aspect_score,
            )

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> RotationSymbol:
        """Construct a RotationSymbol element from a candidate.

        Also finds and claims small images that overlap or are very close to
        the rotation symbol (e.g., dropshadows or reference diagrams that are
        visually part of the rotation symbol).
        """
        # Find small images that should be claimed as part of the rotation symbol
        claimed_images = self._find_rotation_symbol_images(candidate, result)

        if claimed_images:
            # Update the candidate's bbox to include the claimed images
            all_bboxes = [candidate.bbox] + [img.bbox for img in claimed_images]
            expanded_bbox = BBox.union_all(all_bboxes)

            # Add claimed images to source_blocks so they're marked as consumed
            candidate.source_blocks.extend(claimed_images)

            log.debug(
                "[rotation_symbol] Claimed %d additional images for rotation symbol "
                "at %s, expanded bbox to %s, source_blocks=%s",
                len(claimed_images),
                candidate.bbox,
                expanded_bbox,
                [b.id for b in candidate.source_blocks],
            )

            candidate.bbox = expanded_bbox

        return RotationSymbol(bbox=candidate.bbox)

    def _find_rotation_symbol_images(
        self, candidate: Candidate, result: ClassificationResult
    ) -> list[Image]:
        """Find small images that are part of the rotation symbol.

        These are typically dropshadows or small reference diagrams that visually
        belong to the rotation symbol but are stored as separate Image blocks.

        An image is consumed if:
        1. It overlaps with or is very close to the rotation symbol bbox
        2. It's small enough to plausibly be part of the symbol (not a main diagram)
        3. It hasn't already been consumed by another classifier

        Args:
            candidate: The rotation symbol candidate
            result: Classification result containing page data and consumed blocks

        Returns:
            List of Image blocks that should be claimed as part of the rotation symbol
        """
        page_data = result.page_data
        rs_bbox = candidate.bbox

        # Maximum size for an image to be considered part of the rotation symbol.
        # Images larger than 2x the rotation symbol size are likely main diagrams.
        max_image_dimension = rs_bbox.width * 2.0

        # How close an image must be to be claimed (allow small gap for positioning)
        proximity_threshold = 10.0

        # Expand the rotation symbol bbox slightly for overlap detection
        search_bbox = rs_bbox.expand(proximity_threshold)

        # Filter for available images first
        available_images = [
            block
            for block in page_data.blocks
            if isinstance(block, Image) and block.id not in result._consumed_blocks
        ]

        # TODO Maybe this should be "contains" not "overlaps"?
        # Find images that overlap with the expanded search area
        potential_images = filter_overlapping(available_images, search_bbox)

        claimed: Sequence[Image] = []
        for block in potential_images:
            # Skip if image is too large (likely a main diagram)
            if (
                block.bbox.width > max_image_dimension
                or block.bbox.height > max_image_dimension
            ):
                log.debug(
                    "[rotation_symbol] Skipping large image at %s "
                    "(size %.1fx%.1f > max %.1f)",
                    block.bbox,
                    block.bbox.width,
                    block.bbox.height,
                    max_image_dimension,
                )
                continue

            claimed.append(block)
            log.debug(
                "[rotation_symbol] Found rotation symbol image at %s (size %.1fx%.1f)",
                block.bbox,
                block.bbox.width,
                block.bbox.height,
            )

        return claimed

    def _is_black_and_white_cluster(self, cluster: Sequence[Drawing]) -> bool:
        """Check if all drawings in a cluster are black or white.

        Rotation symbols are typically black drawings on transparent/white
        background. This filter rejects clusters with colored drawings.

        Args:
            cluster: List of Drawing blocks in the cluster

        Returns:
            True if all drawings are black or white, False otherwise
        """
        for drawing in cluster:
            if not self._is_black_or_white_color(drawing.fill_color):
                return False
            if not self._is_black_or_white_color(drawing.stroke_color):
                return False
        return True

    def _is_black_or_white_color(
        self, color: tuple[float, ...] | None, tolerance: float = 0.15
    ) -> bool:
        """Check if a color is black, white, or absent (None).

        Args:
            color: RGB color tuple (values 0.0-1.0), or None for no color
            tolerance: How far from pure black (0,0,0) or white (1,1,1) is allowed

        Returns:
            True if color is None, black, or white within tolerance
        """
        if color is None:
            return True

        # Handle RGB (3 values) or CMYK (4 values)
        if len(color) < 3:
            return True  # Unknown format, allow it

        r, g, b = color[0], color[1], color[2]

        # Check if grayscale (R ≈ G ≈ B) - black, white, or gray are all acceptable
        return abs(r - g) <= tolerance and abs(g - b) <= tolerance

    def _score_bbox(
        self,
        bbox: BBox,
    ) -> _RotationSymbolScore | None:
        """Score a bounding box as a potential rotation symbol.

        Scoring is based purely on intrinsic properties (size and aspect ratio).

        Args:
            bbox: Bounding box to score

        Returns:
            Score details if this could be a rotation symbol, None otherwise
        """
        rs_config = self.config.rotation_symbol
        width = bbox.width
        height = bbox.height

        # Check basic size constraints
        if (
            width < rs_config.min_size
            or width > rs_config.max_size
            or height < rs_config.min_size
            or height > rs_config.max_size
        ):
            return None

        # Score size (prefer images close to ideal size)
        ideal_size = rs_config.ideal_size
        size_diff = abs(width - ideal_size) + abs(height - ideal_size)
        size_score = max(0.0, 1.0 - (size_diff / (ideal_size * 2)))

        # Score aspect ratio (prefer square)
        aspect = width / height if height > 0 else 0
        if aspect < rs_config.min_aspect_ratio or aspect > rs_config.max_aspect_ratio:
            return None

        # Perfect square = 1.0, score decreases linearly to 0 at boundaries
        aspect_diff = abs(aspect - 1.0)
        aspect_tolerance = rs_config.max_aspect_ratio - 1.0
        aspect_score = max(0.0, 1.0 - (aspect_diff / aspect_tolerance))

        return _RotationSymbolScore(
            size_score=size_score,
            aspect_score=aspect_score,
            config=rs_config,
        )
