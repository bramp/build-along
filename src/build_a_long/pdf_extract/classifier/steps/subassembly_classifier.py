"""
SubAssembly classifier.

Purpose
-------
Identify sub-assembly callout boxes on LEGO instruction pages. SubAssemblies
typically:
- Are white/light-colored rectangular boxes with black borders
- Are larger than individual parts (to contain a small build diagram)
- May contain a count label (e.g., "2x") indicating how many to build
- Contain SubAssemblyStep elements (step number + diagram pairs)
- May have an arrow pointing from them to the main diagram

Architecture
------------
This classifier works with SubStepClassifier:
1. SubStepClassifier independently finds small step number + diagram pairs
2. SubAssemblyClassifier finds white boxes that contain SubStep candidates
3. When built, SubAssemblyClassifier claims the SubSteps inside its box

Scoring is based on:
- Fill color (white/light) - intrinsic box property
- Whether SubAssemblyStep candidates exist inside the box
- Whether a step_count candidate exists inside

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
    CandidateFailedError,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.config import SubAssemblyConfig
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.score import (
    Score,
    Weight,
    find_best_scoring,
)
from build_a_long.pdf_extract.classifier.utils import score_white_fill
from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    filter_contained,
    group_by_similar_bbox,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    StepCount,
    SubAssembly,
    SubStep,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image

log = logging.getLogger(__name__)


class _SubAssemblyScore(Score):
    """Internal score representation for subassembly classification.

    Scores based on:
    - Box having white/light fill (intrinsic property)
    - SubStep candidates inside (from SubStepClassifier)
    - step_count candidate inside (optional)
    """

    box_score: float
    """Score based on box having white/light fill (0.0-1.0)."""

    has_step_count: bool
    """Whether a step_count candidate exists inside (for scoring bonus)."""

    has_substeps: bool
    """Whether SubStep candidates exist inside (strong signal)."""

    has_images: bool
    """Whether Image blocks exist inside (fallback for simple subassemblies)."""

    config: SubAssemblyConfig
    """Configuration containing weights for score calculation."""

    def score(self) -> Weight:
        """Calculate final weighted score from components.

        Scoring logic:
        - SubAssemblySteps inside: strong signal, full content score
        - Images inside with count: standard weighted score
        - Images inside without count: use box + content only (no count penalty)
        - No images: very low score
        """
        # SubSteps are a strong signal, images are a weaker fallback
        if self.has_substeps:
            content_score = 1.0
        elif self.has_images:
            content_score = 0.8
        else:
            content_score = 0.0

        if self.has_step_count:
            # Standard weighted score when count is present
            count_score = 1.0
            return (
                self.box_score * self.config.box_shape_weight
                + count_score * self.config.count_weight
                + content_score * self.config.diagram_weight
            )
        elif self.has_images or self.has_substeps:
            # No count but has content - use box + content only
            # This allows SubAssemblies like callout diagrams that don't have "2x"
            # Normalize to 0-1 scale by using only box and content weights
            box_weight = self.config.box_shape_weight
            diagram_weight = self.config.diagram_weight
            total_weight = box_weight + diagram_weight
            return (
                self.box_score * box_weight / total_weight
                + content_score * diagram_weight / total_weight
            )
        else:
            # No content - very low score
            return self.box_score * self.config.box_shape_weight


class SubAssemblyClassifier(LabelClassifier):
    """Classifier for subassembly callout boxes."""

    output: ClassVar[str] = "subassembly"
    requires: ClassVar[frozenset[str]] = frozenset({"step_count", "substep"})

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing blocks as potential subassembly boxes."""
        page_data = result.page_data
        subassembly_config = self.config.subassembly

        # Get step_count and substep candidates
        step_count_candidates = result.get_scored_candidates(
            "step_count", valid_only=False, exclude_failed=True
        )
        substep_candidates = result.get_scored_candidates(
            "substep", valid_only=False, exclude_failed=True
        )

        # Find rectangular drawing blocks that could be subassembly boxes
        # Filter by size constraints first
        max_width = page_data.bbox.width * subassembly_config.max_page_width_ratio
        max_height = page_data.bbox.height * subassembly_config.max_page_height_ratio

        valid_drawings: list[Drawing] = []
        for block in page_data.blocks:
            if not isinstance(block, Drawing):
                continue

            bbox = block.bbox

            # Skip boxes smaller than minimum subassembly size
            if (
                bbox.width < subassembly_config.min_subassembly_width
                or bbox.height < subassembly_config.min_subassembly_height
            ):
                continue

            # Skip boxes larger than maximum subassembly size
            if bbox.width > max_width or bbox.height > max_height:
                log.debug(
                    "[subassembly] Skipping oversized box at %s "
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
        # black-bordered box for the same subassembly)
        groups = group_by_similar_bbox(
            valid_drawings, tolerance=subassembly_config.bbox_group_tolerance
        )

        # Process each group - create one candidate per unique bbox region
        for group in groups:
            # Use union of all grouped drawings' bboxes
            bbox = BBox.union_all([d.bbox for d in group])

            # Score each drawing's colors and pick the best
            best_box_score = max(score_white_fill(d) for d in group)
            if best_box_score < 0.3:
                continue

            # Check for child elements inside the box (for scoring)
            has_step_count = bool(
                self._find_candidate_inside(bbox, step_count_candidates)
            )
            has_substeps = bool(
                self._find_all_candidates_inside(bbox, substep_candidates)
            )
            images_inside = self._find_images_inside(bbox, page_data.blocks)
            has_images = bool(images_inside)

            # Create score details
            score_details = _SubAssemblyScore(
                box_score=best_box_score,
                has_step_count=has_step_count,
                has_substeps=has_substeps,
                has_images=has_images,
                config=subassembly_config,
            )

            if score_details.score() < subassembly_config.min_score:
                log.debug(
                    "[subassembly] Rejected box at %s: score=%.2f < min_score=%.2f",
                    bbox,
                    score_details.score(),
                    subassembly_config.min_score,
                )
                continue

            result.add_candidate(
                Candidate(
                    bbox=bbox,
                    label="subassembly",
                    score=score_details.score(),
                    score_details=score_details,
                    source_blocks=list(group),  # Border drawing blocks
                )
            )
            log.debug(
                "[subassembly] Candidate at %s: has_count=%s, "
                "has_substeps=%s, has_images=%s, score=%.2f",
                bbox,
                has_step_count,
                has_substeps,
                has_images,
                score_details.score(),
            )

    def _find_candidate_inside(
        self, bbox: BBox, candidates: list[Candidate]
    ) -> Candidate | None:
        """Find the best candidate that is fully inside the given box.

        Args:
            bbox: The bounding box of the subassembly container
            candidates: Candidates to search

        Returns:
            The best candidate inside the box, or None
        """
        return find_best_scoring(filter_contained(candidates, bbox))

    def _find_all_candidates_inside(
        self, bbox: BBox, candidates: list[Candidate]
    ) -> list[Candidate]:
        """Find all candidates that are fully inside the given box.

        Args:
            bbox: The bounding box of the subassembly container
            candidates: Candidates to search

        Returns:
            List of candidates inside the box, sorted by score (highest first)
        """
        inside = filter_contained(candidates, bbox)

        # Sort by score (highest first)
        inside.sort(key=lambda c: c.score, reverse=True)
        return inside

    def _find_images_inside(self, bbox: BBox, blocks: list) -> list[Image]:
        """Find Image blocks that are fully inside the given box.

        Args:
            bbox: The bounding box of the subassembly container
            blocks: All blocks on the page

        Returns:
            List of Image blocks fully inside the box, sorted by area (largest first)
        """
        min_area = 100.0  # Skip very small images (decorative elements)

        potential_images = [
            b for b in blocks if isinstance(b, Image) and b.bbox.area >= min_area
        ]
        images = filter_contained(potential_images, bbox)

        # Sort by area (largest first) - larger images are more likely to be diagrams
        images.sort(key=lambda img: img.bbox.area, reverse=True)
        return images

    def build(self, candidate: Candidate, result: ClassificationResult) -> SubAssembly:
        """Construct a SubAssembly element from a candidate.

        Claims SubStep candidates that are inside the box, and optionally
        a step_count candidate.
        """
        bbox = candidate.bbox

        # Get candidates for child element discovery
        step_count_candidates = result.get_scored_candidates(
            "step_count", valid_only=False, exclude_failed=True
        )
        substep_candidates = result.get_scored_candidates(
            "substep", valid_only=False, exclude_failed=True
        )

        # Find step_count inside the box and build it
        count = None
        step_count_candidate = self._find_candidate_inside(bbox, step_count_candidates)
        if step_count_candidate:
            count_elem = result.build(step_count_candidate)
            assert isinstance(count_elem, StepCount)
            count = count_elem

        # Find and build SubSteps inside the box
        substeps_inside = self._find_all_candidates_inside(bbox, substep_candidates)

        steps: list[SubStep] = []
        for substep_candidate in substeps_inside:
            try:
                # Pass the SubAssembly's bbox as constraint so diagrams inside
                # substeps don't cluster beyond the SubAssembly boundaries
                substep_elem = result.build(substep_candidate, constraint_bbox=bbox)
                assert isinstance(substep_elem, SubStep)
                steps.append(substep_elem)
            except CandidateFailedError:
                # Already claimed by another SubAssembly - skip
                continue

        # Sort steps by step number value
        steps.sort(key=lambda s: s.step_number.value)

        # If no SubSteps found, try to find a diagram candidate inside
        diagram = None
        if not steps:
            # Look for diagram candidates inside this box
            diagram_candidates = result.get_scored_candidates(
                "diagram", valid_only=False, exclude_failed=True
            )
            diagram_inside = self._find_candidate_inside(bbox, diagram_candidates)
            if diagram_inside:
                try:
                    # Pass the SubAssembly's bbox as a constraint to prevent
                    # the diagram from clustering beyond the box boundaries
                    diagram_elem = result.build(diagram_inside, constraint_bbox=bbox)
                    assert isinstance(diagram_elem, Diagram)
                    diagram = diagram_elem
                except CandidateFailedError:
                    # Diagram already claimed - that's okay
                    pass

        # Subassemblies must contain at least one diagram
        # (either in steps or standalone)
        has_diagram = diagram is not None or len(steps) > 0
        if not has_diagram:
            raise ValueError(
                f"SubAssembly at {bbox} has no diagram - "
                "subassemblies must contain at least one diagram"
            )

        log.debug(
            "[subassembly] Built SubAssembly at %s: %d steps, has_diagram=%s, count=%s",
            bbox,
            len(steps),
            diagram is not None,
            count.count if count else None,
        )

        return SubAssembly(
            bbox=bbox,
            steps=steps,
            diagram=diagram,
            count=count,
        )
