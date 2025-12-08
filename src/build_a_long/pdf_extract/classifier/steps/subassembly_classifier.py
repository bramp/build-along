"""
SubAssembly classifier.

Purpose
-------
Identify sub-assembly callout boxes on LEGO instruction pages. SubAssemblies
typically:
- Are white/light-colored rectangular boxes with black borders
- Are larger than individual parts (to contain a small build diagram)
- May contain a count label (e.g., "2x") indicating how many to build
- May contain step numbers for multi-step subassemblies
- May have an arrow pointing from them to the main diagram

Scoring is based on intrinsic properties of the box:
- Fill color (white/light)
- Size (larger than minimum threshold)

Child element discovery (step_count, step_numbers, diagrams, arrows) is
deferred to build time per DESIGN.md principles.

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
from build_a_long.pdf_extract.classifier.text import extract_step_number_value
from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    filter_contained,
    filter_overlapping,
    group_by_similar_bbox,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    StepCount,
    StepNumber,
    SubAssembly,
    SubAssemblyStep,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing, Image, Text

log = logging.getLogger(__name__)


class _SubAssemblyScore(Score):
    """Internal score representation for subassembly classification.

    Scores based on intrinsic box properties only. Child element discovery
    (step_count, step_numbers, diagrams, arrows) is deferred to build time.
    """

    box_score: float
    """Score based on box having white/light fill (0.0-1.0)."""

    has_step_count: bool
    """Whether a step_count candidate exists inside (for scoring bonus)."""

    has_diagram_or_images: bool
    """Whether diagram candidates or images exist inside (for scoring bonus)."""

    has_step_numbers: bool
    """Whether step_number candidates exist inside (for multi-step subassemblies)."""

    config: SubAssemblyConfig
    """Configuration containing weights for score calculation."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        count_score = 1.0 if self.has_step_count else 0.0
        diagram_score = 1.0 if self.has_diagram_or_images else 0.0

        return (
            self.box_score * self.config.box_shape_weight
            + count_score * self.config.count_weight
            + diagram_score * self.config.diagram_weight
        )


class SubAssemblyClassifier(LabelClassifier):
    """Classifier for subassembly callout boxes."""

    output: ClassVar[str] = "subassembly"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"step_count", "step_number", "diagram"}
    )

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing blocks as potential subassembly boxes."""
        page_data = result.page_data
        subassembly_config = self.config.subassembly

        # Get step_count, step_number, and diagram candidates
        step_count_candidates = result.get_scored_candidates(
            "step_count", valid_only=False, exclude_failed=True
        )
        step_number_candidates = result.get_scored_candidates(
            "step_number", valid_only=False, exclude_failed=True
        )
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
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
        groups = group_by_similar_bbox(valid_drawings, tolerance=2.0)

        # Process each group - create one candidate per unique bbox region
        for group in groups:
            # Use union of all grouped drawings' bboxes
            bbox = BBox.union_all([d.bbox for d in group])

            # Score each drawing's colors and pick the best
            best_box_score = max(self._score_box_colors(d) for d in group)
            if best_box_score < 0.3:
                continue

            # Check for child elements inside the box (for scoring only)
            # Actual candidate discovery happens at build time
            has_step_count = bool(
                self._find_candidate_inside(bbox, step_count_candidates)
            )
            has_step_numbers = bool(
                self._find_all_candidates_inside(bbox, step_number_candidates)
            )
            diagrams_inside = self._find_all_diagrams_inside(bbox, diagram_candidates)
            images_inside = self._find_images_inside(bbox, page_data.blocks)
            has_diagram_or_images = bool(diagrams_inside or images_inside)

            # We need at least a box - count and diagram are optional
            score_details = _SubAssemblyScore(
                box_score=best_box_score,
                has_step_count=has_step_count,
                has_diagram_or_images=has_diagram_or_images,
                has_step_numbers=has_step_numbers,
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
                    source_blocks=list(group),
                )
            )
            log.debug(
                "[subassembly] Candidate at %s: has_count=%s, "
                "has_steps=%s, has_diagrams_or_images=%s, score=%.2f",
                bbox,
                has_step_count,
                has_step_numbers,
                has_diagram_or_images,
                score_details.score(),
            )

    def _score_box_colors(self, block: Drawing) -> float:
        """Score a drawing block based on having white fill.

        SubAssembly boxes typically have a white or light fill color.
        The outer black border boxes can be matched separately later.

        Args:
            block: The Drawing block to analyze

        Returns:
            Score from 0.0 to 1.0 where 1.0 is white fill
        """
        # Check fill color (white or light = good)
        if block.fill_color is not None:
            r, g, b = block.fill_color
            # Check if it's white or very light (all channels > 0.9)
            if r > 0.9 and g > 0.9 and b > 0.9:
                return 1.0
            # Light gray is also acceptable
            if r > 0.7 and g > 0.7 and b > 0.7:
                return 0.6

        return 0.0

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

    def _find_diagram_inside(
        self, bbox: BBox, diagram_candidates: list[Candidate]
    ) -> Candidate | None:
        """Find the best diagram candidate that overlaps with the box.

        Args:
            bbox: The bounding box of the subassembly container
            diagram_candidates: Diagram candidates to search

        Returns:
            The best diagram candidate overlapping the box, or None
        """
        best_candidate = None
        best_overlap = 0.0

        # Use filter_overlapping to narrow down candidates
        overlapping_candidates = filter_overlapping(diagram_candidates, bbox)

        for candidate in overlapping_candidates:
            # Calculate overlap area
            # TODO Should this use bbox.intersection_area(candidate.bbox)?
            overlap = bbox.intersect(candidate.bbox)
            overlap_area = overlap.width * overlap.height
            if overlap_area > best_overlap:
                best_candidate = candidate
                best_overlap = overlap_area

        return best_candidate

    def _find_all_diagrams_inside(
        self, bbox: BBox, diagram_candidates: list[Candidate]
    ) -> list[Candidate]:
        """Find all diagram candidates that are fully inside the box.

        Args:
            bbox: The bounding box of the subassembly container
            diagram_candidates: Diagram candidates to search

        Returns:
            List of diagram candidates inside the box, sorted by area (largest first)
        """
        diagrams = filter_contained(diagram_candidates, bbox)
        # Sort by area (largest first)
        diagrams.sort(key=lambda c: c.bbox.area, reverse=True)
        return diagrams

    def _find_images_inside(self, bbox: BBox, blocks: list[Blocks]) -> list[Image]:
        """Find Image blocks that are fully inside the given box.

        This directly looks at Image blocks, bypassing the diagram clustering.
        Images inside subassembly boxes often get clustered with larger diagrams
        outside the box, so we need to find them directly.

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

        Child element discovery happens here at build time:
        - Find step_count inside the box
        - Find step_numbers inside the box
        - Find diagrams and images inside the box
        - Match step_numbers with diagrams/images
        """
        bbox = candidate.bbox
        page_data = result.page_data

        # Get candidates for child element discovery
        step_count_candidates = result.get_scored_candidates(
            "step_count", valid_only=False, exclude_failed=True
        )
        step_number_candidates = result.get_scored_candidates(
            "step_number", valid_only=False, exclude_failed=True
        )
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )

        # Find step_count inside the box and build it
        count = None
        step_count_candidate = self._find_candidate_inside(bbox, step_count_candidates)
        if step_count_candidate:
            count_elem = result.build(step_count_candidate)
            assert isinstance(count_elem, StepCount)
            count = count_elem

        # Find step_numbers inside the box
        step_nums_inside = self._find_all_candidates_inside(
            bbox, step_number_candidates
        )

        # Find diagrams and images inside the box
        diagrams_inside = self._find_all_diagrams_inside(bbox, diagram_candidates)
        images_inside = self._find_images_inside(bbox, page_data.blocks)

        # Build steps if we have step numbers inside
        steps: list[SubAssemblyStep] = []
        if step_nums_inside:
            # Build step numbers and match them with diagrams or images
            # Pass a slightly inset bbox to constrain diagram clustering.
            # The inset avoids capturing the white border of the subassembly
            # box.
            # TODO Turn the -3 into a config
            inset_bbox = bbox.expand(-3.0)  # Shrink by 3 points on all sides
            steps = self._build_subassembly_steps(
                step_nums_inside,
                diagrams_inside,
                images_inside,
                result,
                constraint_bbox=inset_bbox,
            )

        # Build a single diagram if present and no steps were built
        # Pass a slightly inset bbox as a constraint to prevent diagram
        # clustering from expanding beyond the subassembly bounds.
        # The inset avoids capturing the white border of the subassembly box.
        # TODO Turn the -3 into a config
        diagram = None
        if not steps:
            # TODO Check all diagrams are built inside the box
            if diagrams_inside:
                inset_bbox = bbox.expand(-3.0)  # Shrink by 3 points on all sides
                diagram_elem = result.build(
                    diagrams_inside[0], constraint_bbox=inset_bbox
                )
                assert isinstance(diagram_elem, Diagram)
                diagram = diagram_elem
            elif images_inside:
                # Fall back to using an Image directly as the diagram
                diagram = Diagram(bbox=images_inside[0].bbox)

        # Subassemblies must contain at least one diagram
        # (either in steps or standalone)
        has_diagram = diagram is not None or any(s.diagram is not None for s in steps)
        if not has_diagram:
            raise ValueError(
                f"SubAssembly at {bbox} has no diagram - "
                "subassemblies must contain at least one diagram"
            )

        return SubAssembly(
            bbox=bbox,
            steps=steps,
            diagram=diagram,
            count=count,
        )

    def _build_subassembly_steps(
        self,
        step_number_candidates: list[Candidate],
        diagram_candidates: list[Candidate],
        images_inside: list[Image],
        result: ClassificationResult,
        constraint_bbox: BBox,
    ) -> list[SubAssemblyStep]:
        """Build SubAssemblyStep elements by matching step numbers with diagrams.

        Uses a simple heuristic: diagrams are typically to the right of and/or
        below the step number. For each step number, find the best matching
        diagram based on position. If no diagram candidates are available,
        uses Image blocks found directly inside the subassembly box.

        Args:
            step_number_candidates: Step number candidates inside the subassembly
            diagram_candidates: Diagram candidates inside the subassembly
            images_inside: Image blocks found directly inside the subassembly
            result: Classification result for building elements
            constraint_bbox: Bounding box to constrain diagram clustering

        Returns:
            List of SubAssemblyStep elements, sorted by step number value
        """
        steps: list[SubAssemblyStep] = []
        used_diagram_ids: set[int] = set()
        used_image_ids: set[int] = set()

        # Sort step numbers by their value for consistent ordering
        sorted_step_nums = sorted(
            step_number_candidates,
            key=lambda c: self._extract_step_value(c),
        )

        for step_num_candidate in sorted_step_nums:
            # Build the step number element
            step_num_elem = result.build(step_num_candidate)
            assert isinstance(step_num_elem, StepNumber)

            # Find the best matching diagram for this step
            best_diagram: Diagram | None = None
            best_diagram_id: int | None = None
            best_score = -float("inf")

            # First try diagram candidates
            for diagram_candidate in diagram_candidates:
                diagram_id = id(diagram_candidate)
                if diagram_id in used_diagram_ids:
                    continue

                # Skip candidates that are already failed (e.g., from a previous
                # diagram build that clustered and claimed shared images)
                if diagram_candidate.failure_reason:
                    continue

                # Score this diagram for this step
                score = self._score_step_diagram_match(
                    step_num_candidate.bbox, diagram_candidate.bbox
                )
                if score > best_score:
                    # Build the diagram with constraint to prevent clustering
                    # beyond the subassembly bounds
                    # TODO maybe pass images_inside to constrain diagram clustering?
                    try:
                        diagram_elem = result.build(
                            diagram_candidate, constraint_bbox=constraint_bbox
                        )

                        assert isinstance(diagram_elem, Diagram)
                        best_score = score
                        best_diagram_id = diagram_id
                        best_diagram = diagram_elem
                    except CandidateFailedError:
                        # This candidate was claimed by another diagram during
                        # clustering - skip it and try the next one
                        continue

            # If no diagram candidate found, try Image blocks directly
            if best_diagram is None:
                best_image: Image | None = None
                best_image_id: int | None = None
                best_score = -float("inf")

                for image in images_inside:
                    image_id = id(image)
                    if image_id in used_image_ids:
                        continue

                    # Score this image for this step
                    score = self._score_step_diagram_match(
                        step_num_candidate.bbox, image.bbox
                    )
                    if score > best_score:
                        best_score = score
                        best_image_id = image_id
                        best_image = image

                if best_image is not None and best_image_id is not None:
                    used_image_ids.add(best_image_id)
                    # Create a Diagram from the Image
                    best_diagram = Diagram(bbox=best_image.bbox)

            if best_diagram_id is not None:
                used_diagram_ids.add(best_diagram_id)

            # Compute bbox for the step
            step_bbox = step_num_elem.bbox
            if best_diagram:
                step_bbox = step_bbox.union(best_diagram.bbox)

            steps.append(
                SubAssemblyStep(
                    bbox=step_bbox,
                    step_number=step_num_elem,
                    diagram=best_diagram,
                )
            )

        return steps

    def _extract_step_value(self, candidate: Candidate) -> int:
        """Extract the step number value from a candidate.

        Args:
            candidate: A step_number candidate

        Returns:
            The step number value, or 0 if not extractable
        """
        if candidate.source_blocks and isinstance(candidate.source_blocks[0], Text):
            text_block = candidate.source_blocks[0]
            value = extract_step_number_value(text_block.text)
            return value if value is not None else 0
        return 0

    def _score_step_diagram_match(self, step_bbox: BBox, diagram_bbox: BBox) -> float:
        """Score how well a diagram matches a step number in a subassembly.

        In subassemblies, diagrams are typically positioned to the right of
        and/or below the step number.

        Args:
            step_bbox: The step number bounding box
            diagram_bbox: The diagram bounding box

        Returns:
            Score (higher is better match)
        """
        # Prefer diagrams that are:
        # 1. To the right of the step number (positive x_offset)
        # 2. Below or at same level (positive or small negative y_offset)
        # 3. Close by (small distance)

        x_offset = diagram_bbox.x0 - step_bbox.x1
        y_offset = diagram_bbox.y0 - step_bbox.y0

        # X score: prefer diagrams to the right
        if x_offset >= 0:
            x_score = 1.0 - min(x_offset / 200.0, 0.5)
        else:
            x_score = 0.5 + x_offset / 100.0  # Penalize left position

        # Y score: prefer diagrams at same level or below
        if abs(y_offset) < 50:
            y_score = 1.0
        elif y_offset >= 0:
            y_score = 0.8 - min(y_offset / 200.0, 0.3)
        else:
            y_score = 0.5 + y_offset / 100.0  # Penalize above position

        return x_score + y_score
