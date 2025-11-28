"""
Rotation symbol classifier.

Purpose
-------
Identify rotation symbols on LEGO instruction pages. These symbols indicate
that the builder should rotate the assembled model. They can appear as either:
1. Small raster images (most common) - square icons ~40-80px
2. Clusters of vector drawings forming arrow patterns

Heuristic
---------
- Look for square Image blocks (aspect ratio ~0.85-1.15)
- Size range: 35-120 pixels per side
- Often positioned near Diagram elements
- Can also be clusters of 4-25 small Drawing elements forming arrows

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


@dataclass(frozen=True)
class RotationSymbolClassifier(LabelClassifier):
    """Classifier for rotation symbol elements."""

    output = "rotation_symbol"
    requires = frozenset({"diagram"})

    def _score(self, result: ClassificationResult) -> None:
        """Score Image blocks and Drawing clusters as rotation symbol candidates."""
        page_data = result.page_data
        config = self.config

        # Get diagram candidates to check proximity
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )

        # Approach 1: Check raster Images
        for block in page_data.blocks:
            if isinstance(block, Image):
                score_details = self._score_bbox(block.bbox, diagram_candidates)
                if (
                    score_details
                    and score_details.score() > config.rotation_symbol_min_score
                ):
                    result.add_candidate(
                        Candidate(
                            bbox=block.bbox,
                            label="rotation_symbol",
                            score=score_details.score(),
                            score_details=score_details,
                            # Don't claim source_blocks - rotation symbols
                            # can coexist with diagrams and part images
                            source_blocks=[],
                        )
                    )
                    log.debug(
                        "[rotation_symbol] Image candidate at %s score=%.2f "
                        "(size=%.2f aspect=%.2f proximity=%.2f)",
                        block.bbox,
                        score_details.score(),
                        score_details.size_score,
                        score_details.aspect_score,
                        score_details.proximity_to_diagram,
                    )

        # Approach 2: Check Drawing clusters (for vector-based symbols)
        drawings = [
            block
            for block in page_data.blocks
            if isinstance(block, Drawing) and self._is_small_drawing(block)
        ]

        if drawings:
            clusters = self._build_drawing_clusters(drawings)
            for cluster in clusters:
                if (
                    config.rotation_symbol_min_drawings_in_cluster
                    <= len(cluster)
                    <= config.rotation_symbol_max_drawings_in_cluster
                ):
                    cluster_bbox = BBox.union_all([d.bbox for d in cluster])
                    score_details = self._score_bbox(
                        cluster_bbox,
                        diagram_candidates,
                        size_score_override=config.rotation_symbol_cluster_size_score,
                    )
                    if (
                        score_details
                        and score_details.score() > config.rotation_symbol_min_score
                    ):
                        result.add_candidate(
                            Candidate(
                                bbox=cluster_bbox,
                                label="rotation_symbol",
                                score=score_details.score(),
                                score_details=score_details,
                                # Don't claim source_blocks - these drawings
                                # are also part of diagrams and should coexist
                                source_blocks=[],
                            )
                        )
                        log.debug(
                            "[rotation_symbol] Drawing cluster candidate "
                            "at %s with %d drawings score=%.2f",
                            cluster_bbox,
                            len(cluster),
                            score_details.score(),
                        )

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> RotationSymbol:
        """Construct a RotationSymbol element from a candidate."""
        return RotationSymbol(
            bbox=candidate.bbox,
        )

    def _score_bbox(
        self,
        bbox: BBox,
        diagram_candidates: list[Candidate],
        size_score_override: float | None = None,
    ) -> _RotationSymbolScore | None:
        """Score a bounding box as a potential rotation symbol.

        Args:
            bbox: Bounding box to score
            diagram_candidates: List of diagram candidates for proximity scoring
            size_score_override: If provided, use this fixed size score instead
                of calculating from ideal size (used for drawing clusters)

        Returns:
            Score details if this could be a rotation symbol, None otherwise
        """
        config = self.config
        width = bbox.width
        height = bbox.height

        # Check basic size constraints
        if (
            width < config.rotation_symbol_min_size
            or width > config.rotation_symbol_max_size
            or height < config.rotation_symbol_min_size
            or height > config.rotation_symbol_max_size
        ):
            return None

        # Score size (prefer images close to ideal size)
        if size_score_override is not None:
            size_score = size_score_override
        else:
            ideal_size = config.rotation_symbol_ideal_size
            size_diff = abs(width - ideal_size) + abs(height - ideal_size)
            size_score = max(0.0, 1.0 - (size_diff / (ideal_size * 2)))

        # Score aspect ratio (prefer square)
        aspect = width / height if height > 0 else 0
        if (
            aspect < config.rotation_symbol_min_aspect
            or aspect > config.rotation_symbol_max_aspect
        ):
            return None

        # Perfect square = 1.0, score decreases linearly to 0 at boundaries
        aspect_diff = abs(aspect - 1.0)
        aspect_tolerance = config.rotation_symbol_max_aspect - 1.0
        aspect_score = max(0.0, 1.0 - (aspect_diff / aspect_tolerance))

        # Score proximity to diagrams
        proximity_score = self._calculate_proximity_to_diagrams(
            bbox, diagram_candidates
        )

        return _RotationSymbolScore(
            size_score=size_score,
            aspect_score=aspect_score,
            proximity_to_diagram=proximity_score,
            size_weight=config.rotation_symbol_size_weight,
            aspect_weight=config.rotation_symbol_aspect_weight,
            proximity_weight=config.rotation_symbol_proximity_weight,
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
        config = self.config
        close_distance = config.rotation_symbol_proximity_close_distance
        far_distance = config.rotation_symbol_proximity_far_distance

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

    def _is_small_drawing(self, drawing: Drawing) -> bool:
        """Check if a drawing is small enough to be part of a rotation symbol.

        Args:
            drawing: The Drawing block to check

        Returns:
            True if the drawing could be part of a rotation symbol cluster
        """
        max_size = self.config.rotation_symbol_max_size
        return drawing.bbox.width < max_size and drawing.bbox.height < max_size

    def _build_drawing_clusters(self, drawings: list[Drawing]) -> list[list[Drawing]]:
        """Build clusters of connected small drawings.

        Args:
            drawings: List of small drawing blocks

        Returns:
            List of clusters, where each cluster is a list of connected drawings
        """
        if not drawings:
            return []

        remaining = set(range(len(drawings)))
        clusters: list[list[Drawing]] = []

        while remaining:
            seed_idx = min(remaining)
            seed_drawing = drawings[seed_idx]
            remaining.remove(seed_idx)

            # Build cluster using connected drawings
            cluster = build_connected_cluster([seed_drawing], drawings)

            # Remove clustered drawings from remaining set
            for drawing in cluster:
                try:
                    idx = drawings.index(drawing)
                    remaining.discard(idx)
                except ValueError:
                    pass

            clusters.append(cluster)

        return clusters
