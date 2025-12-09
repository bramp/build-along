"""
Preview classifier.

Purpose
-------
Identify preview areas on LEGO instruction pages. Previews are white rectangular
areas containing diagrams that show what the completed model (or a section of it)
will look like. They typically appear on info pages.

Previews are identified by:
- A white/light rectangular box (Drawing element with white fill)
- One or more images inside forming a diagram
- Located outside the main instruction/step areas (not inside steps/subassemblies)

Scoring is based on intrinsic properties of the box:
- Fill color (white/light)
- Size (within min/max thresholds)
- Presence of images inside

Child element discovery (diagram) is deferred to build time per DESIGN.md
principles.

Debugging
---------
Set environment variables to aid investigation without code changes:

- LOG_LEVEL=DEBUG
    Enables DEBUG-level logging (if not already configured by caller).
"""

from __future__ import annotations

import logging
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.config import PreviewConfig
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.score import (
    Score,
    Weight,
)
from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    filter_contained,
    filter_overlapping,
    group_by_similar_bbox,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    Preview,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing, Image

log = logging.getLogger(__name__)


class _PreviewScore(Score):
    """Internal score representation for preview classification.

    Scores based on intrinsic box properties only. Child element discovery
    (diagram) is deferred to build time.
    """

    box_score: float
    """Score based on box having white fill (0.0-1.0)."""

    fill_score: float
    """Score for fill color whiteness (0.0-1.0)."""

    has_images: bool
    """Whether images exist inside (for scoring bonus)."""

    config: PreviewConfig
    """Configuration containing weights for score calculation."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        diagram_score = 1.0 if self.has_images else 0.0

        return (
            self.box_score * self.config.box_shape_weight
            + self.fill_score * self.config.fill_color_weight
            + diagram_score * self.config.diagram_weight
        )


class PreviewClassifier(LabelClassifier):
    """Classifier for preview areas on instruction pages."""

    output: ClassVar[str] = "preview"
    requires: ClassVar[frozenset[str]] = frozenset(
        {
            "diagram",
            # step_count and step_number aren't included in the preview,
            # but reviewed to help avoid false positives.
            "step_count",
            "step_number",
        }
    )

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing blocks as potential preview boxes.

        Previews are white boxes containing diagrams that appear BEFORE steps.
        They are distinguished from subassemblies by:
        - Not containing step_count labels (e.g., "2x")
        - Not overlapping with step_number elements (they're outside step areas)
        """
        page_data = result.page_data
        preview_config = self.config.preview

        # Get diagram candidates for checking what's inside potential previews
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )

        # Get step_count and step_number candidates to distinguish from subassemblies
        # Subassemblies contain step_counts (e.g., "2x") and are within step areas
        step_count_candidates = result.get_scored_candidates(
            "step_count", valid_only=False, exclude_failed=True
        )
        step_number_candidates = result.get_scored_candidates(
            "step_number", valid_only=False, exclude_failed=True
        )

        # Find rectangular drawing blocks that could be preview boxes
        max_width = page_data.bbox.width * preview_config.max_page_width_ratio
        max_height = page_data.bbox.height * preview_config.max_page_height_ratio

        valid_drawings: list[Drawing] = []
        for block in page_data.blocks:
            if not isinstance(block, Drawing):
                continue

            bbox = block.bbox

            # Skip boxes smaller than minimum preview size
            if (
                bbox.width < preview_config.min_width
                or bbox.height < preview_config.min_height
            ):
                continue

            # Skip boxes larger than maximum preview size
            if bbox.width > max_width or bbox.height > max_height:
                log.debug(
                    "[preview] Skipping oversized box at %s "
                    "(%.1f x %.1f > max %.1f x %.1f)",
                    bbox,
                    bbox.width,
                    bbox.height,
                    max_width,
                    max_height,
                )
                continue

            valid_drawings.append(block)

        # Group drawings with similar bboxes (e.g., white-filled box and
        # border box for the same preview)
        groups = group_by_similar_bbox(valid_drawings, tolerance=2.0)

        # Process each group - create one candidate per unique bbox region
        for group in groups:
            # Use union of all grouped drawings' bboxes
            bbox = BBox.union_all([d.bbox for d in group])

            # Reject boxes that contain step_count labels (those are subassemblies)
            # Subassemblies have labels like "2x" indicating how many to build
            step_counts_inside = filter_contained(step_count_candidates, bbox)
            if step_counts_inside:
                log.debug(
                    "[preview] Rejected box at %s: contains step_count "
                    "(likely a subassembly)",
                    bbox,
                )
                continue

            # Reject boxes that overlap with step_numbers
            # Previews appear before steps, so they shouldn't overlap step_numbers
            step_numbers_inside = filter_overlapping(step_number_candidates, bbox)
            if step_numbers_inside:
                log.debug(
                    "[preview] Rejected box at %s: overlaps step_number "
                    "(likely part of a step area)",
                    bbox,
                )
                continue

            # Score each drawing's colors and pick the best
            best_fill_score = 0.0
            best_box_score = 0.0
            for drawing in group:
                fill_score = self._score_fill_color(drawing, preview_config)
                if fill_score > best_fill_score:
                    best_fill_score = fill_score
                    best_box_score = 1.0 if fill_score > 0 else 0.0

            if best_fill_score < 0.3:
                continue

            # Check for images inside the box
            images_inside = self._find_images_inside(bbox, page_data.blocks)
            diagrams_inside = self._find_diagrams_inside(bbox, diagram_candidates)
            has_images = bool(images_inside or diagrams_inside)

            # Create score details
            score_details = _PreviewScore(
                box_score=best_box_score,
                fill_score=best_fill_score,
                has_images=has_images,
                config=preview_config,
            )

            if score_details.score() < preview_config.min_score:
                log.debug(
                    "[preview] Rejected box at %s: score=%.2f < min_score=%.2f",
                    bbox,
                    score_details.score(),
                    preview_config.min_score,
                )
                continue

            result.add_candidate(
                Candidate(
                    bbox=bbox,
                    label="preview",
                    score=score_details.score(),
                    score_details=score_details,
                    source_blocks=list(group),
                )
            )
            log.debug(
                "[preview] Candidate at %s: has_images=%s, score=%.2f",
                bbox,
                has_images,
                score_details.score(),
            )

    def _score_fill_color(self, block: Drawing, config: PreviewConfig) -> float:
        """Score a drawing block based on having white fill.

        Preview boxes typically have a white or light fill color.

        Args:
            block: The Drawing block to analyze
            config: Preview configuration with white threshold

        Returns:
            Score from 0.0 to 1.0 where 1.0 is white fill
        """
        if block.fill_color is not None:
            r, g, b = block.fill_color
            # Check if it's white (all channels above threshold)
            if (
                r >= config.white_threshold
                and g >= config.white_threshold
                and b >= config.white_threshold
            ):
                return 1.0
            # Light gray is also acceptable
            if r > 0.8 and g > 0.8 and b > 0.8:
                return 0.7

        return 0.0

    def _find_images_inside(self, bbox: BBox, blocks: list[Blocks]) -> list[Image]:
        """Find Image blocks that are fully inside the given box.

        Args:
            bbox: The bounding box of the preview container
            blocks: All blocks on the page

        Returns:
            List of Image blocks fully inside the box, sorted by area (largest first)
        """
        min_area = 100.0  # Skip very small images (decorative elements)

        potential_images = [
            b for b in blocks if isinstance(b, Image) and b.bbox.area >= min_area
        ]
        images = filter_contained(potential_images, bbox)

        # Sort by area (largest first)
        images.sort(key=lambda img: img.bbox.area, reverse=True)
        return images

    def _find_diagrams_inside(
        self, bbox: BBox, diagram_candidates: list[Candidate]
    ) -> list[Candidate]:
        """Find diagram candidates that are fully inside the given box.

        Args:
            bbox: The bounding box of the preview container
            diagram_candidates: Diagram candidates to search

        Returns:
            List of diagram candidates inside the box, sorted by area (largest first)
        """
        diagrams = filter_contained(diagram_candidates, bbox)
        # Sort by area (largest first)
        diagrams.sort(key=lambda c: c.bbox.area, reverse=True)
        return diagrams

    def build(self, candidate: Candidate, result: ClassificationResult) -> Preview:
        """Construct a Preview element from a candidate.

        Child element discovery happens here at build time:
        - Find diagrams inside the box
        - Find images inside the box (if no diagram candidates)
        - Build the diagram with constraint bbox
        """
        bbox = candidate.bbox

        # Get diagram candidates for child element discovery
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )

        # Find diagrams inside the box
        diagrams_inside = self._find_diagrams_inside(bbox, diagram_candidates)

        # Build the diagram
        # Pass a slightly inset bbox to constrain diagram clustering
        # to stay within the preview bounds
        inset_bbox = bbox.expand(-3.0)  # Shrink by 3 points on all sides
        diagram: Diagram | None = None

        if diagrams_inside:
            diagram_elem = result.build(diagrams_inside[0], constraint_bbox=inset_bbox)
            assert isinstance(diagram_elem, Diagram)
            diagram = diagram_elem

        return Preview(
            bbox=bbox,
            diagram=diagram,
        )
