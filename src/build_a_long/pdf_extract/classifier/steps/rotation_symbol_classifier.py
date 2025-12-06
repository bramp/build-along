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
4. Score clusters that are:
   - Square-ish (aspect ratio ~0.95-1.05)
   - Small (~41-51 pixels per side, ±10% of ideal 46px)
   - Near a Diagram element

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

    proximity_to_diagram: float
    """Score based on proximity to a diagram (0.0-1.0)."""

    # Store weights for score calculation
    size_weight: float = 0.5
    aspect_weight: float = 0.3
    proximity_weight: float = 0.2

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        return (
            self.size_score * self.size_weight
            + self.aspect_score * self.aspect_weight
            + self.proximity_to_diagram * self.proximity_weight
        )


class RotationSymbolClassifier(LabelClassifier):
    """Classifier for rotation symbol elements."""

    output = "rotation_symbol"
    requires = frozenset({"diagram"})

    def _score(self, result: ClassificationResult) -> None:
        """Score connected clusters of Drawing blocks as rotation symbols."""
        page_data = result.page_data
        config = self.config
        page_bbox = page_data.bbox
        assert page_bbox is not None

        # Get diagram candidates to check proximity
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )

        # Filter out page-spanning drawings (>90% of page width or height).
        # These are typically background/border elements that would connect
        # unrelated symbols together during clustering.
        max_width = page_bbox.width * 0.9
        max_height = page_bbox.height * 0.9
        drawings: list[Drawing] = [
            block
            for block in page_data.blocks
            if isinstance(block, Drawing)
            and block.bbox.width <= max_width
            and block.bbox.height <= max_height
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

            score_details = self._score_bbox(cluster_bbox, diagram_candidates)
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
                    # Don't claim source_blocks - rotation symbols
                    # can coexist with diagrams and part images
                    source_blocks=[],
                )
            )
            log.debug(
                "[rotation_symbol] Cluster candidate at %s "
                "(%d blocks) score=%.2f "
                "(size=%.2f aspect=%.2f proximity=%.2f)",
                cluster_bbox,
                len(cluster),
                score_details.score(),
                score_details.size_score,
                score_details.aspect_score,
                score_details.proximity_to_diagram,
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

        An image is claimed if:
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
        search_bbox = BBox(
            x0=rs_bbox.x0 - proximity_threshold,
            y0=rs_bbox.y0 - proximity_threshold,
            x1=rs_bbox.x1 + proximity_threshold,
            y1=rs_bbox.y1 + proximity_threshold,
        )

        claimed: list[Image] = []
        for block in page_data.blocks:
            if not isinstance(block, Image):
                continue

            # Skip if already consumed
            if block.id in result._consumed_blocks:
                continue

            # Check if image overlaps with expanded search area
            if not block.bbox.overlaps(search_bbox):
                continue

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

    def _score_bbox(
        self,
        bbox: BBox,
        diagram_candidates: list[Candidate],
    ) -> _RotationSymbolScore | None:
        """Score a bounding box as a potential rotation symbol.

        Args:
            bbox: Bounding box to score
            diagram_candidates: List of diagram candidates for proximity scoring

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
        if aspect < rs_config.min_aspect or aspect > rs_config.max_aspect:
            return None

        # Perfect square = 1.0, score decreases linearly to 0 at boundaries
        aspect_diff = abs(aspect - 1.0)
        aspect_tolerance = rs_config.max_aspect - 1.0
        aspect_score = max(0.0, 1.0 - (aspect_diff / aspect_tolerance))

        # Score proximity to diagrams
        proximity_score = self._calculate_proximity_to_diagrams(
            bbox, diagram_candidates
        )

        return _RotationSymbolScore(
            size_score=size_score,
            aspect_score=aspect_score,
            proximity_to_diagram=proximity_score,
            size_weight=rs_config.size_weight,
            aspect_weight=rs_config.aspect_weight,
            proximity_weight=rs_config.proximity_weight,
        )

    def _calculate_proximity_to_diagrams(
        self, bbox: BBox, diagram_candidates: list[Candidate]
    ) -> float:
        """Calculate proximity score based on distance to nearest diagram.

        Rotation symbols are typically positioned near diagrams.

        Args:
            bbox: Bounding box of the potential rotation symbol
            diagram_candidates: List of diagram candidates

        Returns:
            Score from 0.0 (far from diagrams) to 1.0 (very close to diagram)
        """
        rs_config = self.config.rotation_symbol
        close_distance = rs_config.proximity_close_distance
        far_distance = rs_config.proximity_far_distance

        if not diagram_candidates:
            # No diagrams on page, give neutral score
            return 0.5

        # Find minimum edge-to-edge distance to any diagram
        min_distance = min(
            bbox.min_distance(diagram_cand.bbox) for diagram_cand in diagram_candidates
        )

        # Score based on distance (closer = better)
        # min_distance returns 0.0 for overlapping bboxes
        if min_distance < close_distance:
            return 1.0
        elif min_distance > far_distance:
            return 0.0
        else:
            return 1.0 - (
                (min_distance - close_distance) / (far_distance - close_distance)
            )
