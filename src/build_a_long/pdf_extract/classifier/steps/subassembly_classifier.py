"""
SubAssembly classifier.

Purpose
-------
Identify sub-assembly callout boxes on LEGO instruction pages. SubAssemblies
typically:
- Are white/light-colored rectangular boxes
- Contain a count label (e.g., "2x") indicating how many to build
- Contain a small diagram/image of the sub-assembly
- Have an arrow pointing from them to the main diagram

Heuristic
---------
1. Find Drawing blocks that form rectangular boxes (potential subassembly containers)
2. Look for step_count candidates inside the boxes
3. Look for diagram candidates inside the boxes
4. Optionally find arrows near the boxes

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
    """Internal score representation for subassembly classification."""

    box_score: float
    """Score based on box having white fill / black border (0.0-1.0)."""

    count_score: float
    """Score for having a valid step_count candidate inside (0.0-1.0)."""

    diagram_score: float
    """Score for having a diagram candidate inside (0.0-1.0)."""

    step_count_candidate: Candidate | None
    """The step_count candidate found inside the box."""

    diagram_candidate: Candidate | None
    """The diagram candidate found inside the box."""

    step_number_candidates: list[Candidate]
    """Step number candidates found inside the box (for multi-step subassemblies)."""

    diagram_candidates: list[Candidate]
    """All diagram candidates inside the box (for multi-step subassemblies)."""

    images_inside: list[Image]
    """Image blocks found directly inside the subassembly box (not from clustering)."""

    arrow_candidate: Candidate | None
    """Arrow candidate pointing from/near this subassembly."""

    config: SubAssemblyConfig
    """Configuration containing weights for score calculation."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        return (
            self.box_score * self.config.box_shape_weight
            + self.count_score * self.config.count_weight
            + self.diagram_score * self.config.diagram_weight
        )


class SubAssemblyClassifier(LabelClassifier):
    """Classifier for subassembly callout boxes."""

    output: ClassVar[str] = "subassembly"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"arrow", "step_count", "step_number", "diagram"}
    )

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing blocks as potential subassembly boxes."""
        page_data = result.page_data
        subassembly_config = self.config.subassembly

        # Get step_count, step_number, diagram, and arrow candidates
        step_count_candidates = result.get_scored_candidates(
            "step_count", valid_only=False, exclude_failed=True
        )
        step_number_candidates = result.get_scored_candidates(
            "step_number", valid_only=False, exclude_failed=True
        )
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )
        arrow_candidates = result.get_scored_candidates(
            "arrow", valid_only=False, exclude_failed=True
        )

        # Collect all potential candidates, then deduplicate by bbox
        # Multiple Drawing blocks can have nearly identical bboxes (e.g.,
        # white-filled box and black-bordered box for the same subassembly)
        found_bboxes: list[BBox] = []

        # Find rectangular drawing blocks that could be subassembly boxes
        for block in page_data.blocks:
            if not isinstance(block, Drawing):
                continue

            bbox = block.bbox

            # Skip boxes smaller than minimum subassembly size
            # (subassemblies must be larger than individual parts)
            if (
                bbox.width < subassembly_config.min_subassembly_width
                or bbox.height < subassembly_config.min_subassembly_height
            ):
                continue

            # Skip if we've already found a similar bbox
            if any(bbox.similar(found, tolerance=2.0) for found in found_bboxes):
                log.debug(
                    "[subassembly] Skipping duplicate bbox at %s",
                    bbox,
                )
                continue

            # Score the box colors (white fill, black border)
            box_score = self._score_box_colors(block)
            if box_score < 0.3:
                continue

            # Find step_count candidate inside the box
            step_count_candidate = self._find_candidate_inside(
                bbox, step_count_candidates
            )
            count_score = 1.0 if step_count_candidate else 0.0

            # Find all step_number candidates inside the box
            step_nums_inside = self._find_all_candidates_inside(
                bbox, step_number_candidates
            )

            # Find all diagram candidates inside/overlapping the box
            diagrams_inside = self._find_all_diagrams_inside(bbox, diagram_candidates)

            # Find Image blocks directly inside the box (not from clustering)
            # This catches images that were absorbed into larger diagram clusters
            images_inside = self._find_images_inside(bbox, page_data.blocks)

            # For scoring, use the best/primary diagram
            diagram_candidate = diagrams_inside[0] if diagrams_inside else None
            # If we have images but no diagram candidates, still give credit
            has_diagram_or_images = bool(diagram_candidate or images_inside)
            diagram_score = 1.0 if has_diagram_or_images else 0.0

            # Find nearby arrow
            arrow_candidate = self._find_arrow_for_subassembly(bbox, arrow_candidates)

            # We need at least a box - count and diagram are optional
            score_details = _SubAssemblyScore(
                box_score=box_score,
                count_score=count_score,
                diagram_score=diagram_score,
                step_count_candidate=step_count_candidate,
                diagram_candidate=diagram_candidate,
                step_number_candidates=step_nums_inside,
                diagram_candidates=diagrams_inside,
                images_inside=images_inside,
                arrow_candidate=arrow_candidate,
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

            # Track this bbox to avoid duplicates
            found_bboxes.append(bbox)

            result.add_candidate(
                Candidate(
                    bbox=bbox,
                    label="subassembly",
                    score=score_details.score(),
                    score_details=score_details,
                    source_blocks=[block],
                )
            )
            log.debug(
                "[subassembly] Candidate at %s: has_count=%s, "
                "has_steps=%d, has_diagrams=%d, has_images=%d, score=%.2f",
                bbox,
                step_count_candidate is not None,
                len(step_nums_inside),
                len(diagrams_inside),
                len(images_inside),
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

        for candidate in diagram_candidates:
            if bbox.overlaps(candidate.bbox):
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
        """Find all diagram candidates that overlap significantly with the box.

        Args:
            bbox: The bounding box of the subassembly container
            diagram_candidates: Diagram candidates to search

        Returns:
            List of diagram candidates overlapping the box, sorted by overlap area
        """
        diagrams: list[tuple[float, Candidate]] = []

        for candidate in diagram_candidates:
            if bbox.overlaps(candidate.bbox):
                # Calculate overlap area
                overlap = bbox.intersect(candidate.bbox)
                overlap_area = overlap.width * overlap.height
                # Only include if significant overlap (at least 50% inside the box)
                candidate_area = candidate.bbox.area
                if candidate_area > 0 and overlap_area / candidate_area >= 0.5:
                    diagrams.append((overlap_area, candidate))

        # Sort by overlap area (largest first)
        diagrams.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in diagrams]

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

    def _find_arrow_for_subassembly(
        self, bbox: BBox, arrow_candidates: list[Candidate]
    ) -> Candidate | None:
        """Find an arrow that points from/near this subassembly box.

        Looks for arrows that are either:
        - Inside the box
        - Adjacent to the box (within a small margin)

        Args:
            bbox: The bounding box of the subassembly container
            arrow_candidates: All arrow candidates on the page

        Returns:
            The best matching arrow candidate, or None
        """
        margin = 20.0  # Points of margin around the box
        expanded_bbox = bbox.expand(margin)
        overlapping = filter_overlapping(arrow_candidates, expanded_bbox)
        return find_best_scoring(overlapping)

    def build(self, candidate: Candidate, result: ClassificationResult) -> SubAssembly:
        """Construct a SubAssembly element from a candidate."""
        score_details = candidate.score_details
        assert isinstance(score_details, _SubAssemblyScore)

        # Build the step_count element if present
        count = None
        if score_details.step_count_candidate:
            count_elem = result.build(score_details.step_count_candidate)
            assert isinstance(count_elem, StepCount)
            count = count_elem

        # Build steps if we have step numbers inside
        steps: list[SubAssemblyStep] = []
        if score_details.step_number_candidates:
            # Build step numbers and match them with diagrams or images
            steps = self._build_subassembly_steps(
                score_details.step_number_candidates,
                score_details.diagram_candidates,
                score_details.images_inside,
                result,
            )

        # Build a single diagram if present and no steps were built
        diagram = None
        if not steps and score_details.diagram_candidate:
            diagram_elem = result.build(score_details.diagram_candidate)
            assert isinstance(diagram_elem, Diagram)
            diagram = diagram_elem

        return SubAssembly(
            bbox=candidate.bbox,
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

                # Score this diagram for this step
                score = self._score_step_diagram_match(
                    step_num_candidate.bbox, diagram_candidate.bbox
                )
                if score > best_score:
                    best_score = score
                    best_diagram_id = diagram_id
                    # Build the diagram
                    diagram_elem = result.build(diagram_candidate)
                    assert isinstance(diagram_elem, Diagram)
                    best_diagram = diagram_elem

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
