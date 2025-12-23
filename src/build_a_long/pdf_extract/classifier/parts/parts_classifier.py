"""
Parts classifier.

Purpose
-------
Assemble Part candidates by chaining spatially constrained searches:
  PartImage (anchor) → PartCount (below) → PartNumber (below count)

The PartImage serves as the anchor because it's the most visually
distinctive element. From each image, we search for a PartCount directly
below and left-aligned. From each count, we optionally search for a
PartNumber directly below. PieceLength is searched near the top-right
of the PartImage.

Spatial Constraints
-------------------
The tight spatial constraints mean that for a page with ~100 parts,
we typically get ~100 Part candidates. Ambiguous cases (where multiple
elements could match) create extra candidates that the constraint solver
resolves.

Search Chain:
1. PartImage → PartCount: Count must be below image, left-aligned
2. PartCount → PartNumber: Number must be below count, left-aligned
3. PartImage → PieceLength: Length must be near top-right of image

Constraint Solver Integration
-----------------------------
The solver enforces one-to-one pairing for:
- Each PartCount used by at most one Part
- Each PartImage used by at most one Part
- Each PartNumber used by at most one Part
- Each PieceLength used by at most one Part

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.constraint_model import ConstraintModel
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.rules.scoring import (
    score_exponential_decay,
    score_triangular,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import BBox, filter_overlapping
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Part,
    PartCount,
    PartImage,
    PartNumber,
    PieceLength,
)

log = logging.getLogger(__name__)


# =============================================================================
# Search Region Definitions
# =============================================================================


@dataclass(frozen=True)
class SearchRegion:
    """Defines a spatial search region relative to an anchor element.

    This encapsulates the search parameters for finding elements
    that are spatially related to an anchor (e.g., PartCount below PartImage).
    """

    bbox: BBox
    """The bounding box defining the search region."""

    vertical_tolerance: float
    """Maximum vertical overlap allowed (negative = overlap)."""

    horizontal_tolerance: float
    """Maximum horizontal alignment offset allowed."""

    max_distance: float
    """Maximum distance between elements."""

    @classmethod
    def below_and_aligned(
        cls,
        anchor: BBox,
        *,
        max_distance: float = 10.0,
        vertical_tolerance: float = 2.0,
        horizontal_tolerance: float = 3.0,
    ) -> SearchRegion:
        """Create a search region for elements directly below an anchor.

        The region extends from the anchor's bottom edge downward,
        constrained by max_distance.

        Args:
            anchor: The anchor element's bounding box
            max_distance: Maximum vertical distance below anchor
            vertical_tolerance: Allow slight overlap (anchor.y1 - tolerance)
            horizontal_tolerance: Allow slight horizontal offset

        Returns:
            SearchRegion configured for below-and-aligned search
        """
        search_bbox = BBox(
            x0=anchor.x0 - horizontal_tolerance,
            y0=anchor.y1 - vertical_tolerance,  # Start from anchor bottom
            x1=anchor.x1 + horizontal_tolerance,
            y1=anchor.y1 + max_distance,  # Extend down by max_distance
        )
        return cls(
            bbox=search_bbox,
            vertical_tolerance=vertical_tolerance,
            horizontal_tolerance=horizontal_tolerance,
            max_distance=max_distance,
        )

    @classmethod
    def above_and_aligned(
        cls,
        anchor: BBox,
        *,
        max_distance: float = 10.0,
        vertical_tolerance: float = 2.0,
        horizontal_tolerance: float = 3.0,
    ) -> SearchRegion:
        """Create a search region for elements directly above an anchor.

        Args:
            anchor: The anchor element's bounding box
            max_distance: Maximum vertical distance above anchor
            vertical_tolerance: Allow slight overlap
            horizontal_tolerance: Allow slight horizontal offset

        Returns:
            SearchRegion configured for above-and-aligned search
        """
        search_bbox = BBox(
            x0=anchor.x0 - horizontal_tolerance,
            y0=max(0, anchor.y0 - max_distance),  # Don't go above page
            x1=anchor.x1 + horizontal_tolerance,
            y1=anchor.y0 + vertical_tolerance,  # Allow slight overlap
        )
        return cls(
            bbox=search_bbox,
            vertical_tolerance=vertical_tolerance,
            horizontal_tolerance=horizontal_tolerance,
            max_distance=max_distance,
        )

    @classmethod
    def nearby(
        cls,
        anchor: BBox,
        *,
        max_distance: float = 10.0,
    ) -> SearchRegion:
        """Create a search region for elements near an anchor.

        Expands the anchor bbox by max_distance in all directions.

        Args:
            anchor: The anchor element's bounding box
            max_distance: Maximum distance from anchor edge

        Returns:
            SearchRegion configured for nearby search
        """
        return cls(
            bbox=anchor.expand(max_distance),
            vertical_tolerance=max_distance,
            horizontal_tolerance=max_distance,
            max_distance=max_distance,
        )

    def find_candidates(
        self,
        candidates: Sequence[Candidate],
        anchor: BBox,
        *,
        require_below: bool = False,
        require_above: bool = False,
        require_aligned: bool = True,
    ) -> Iterable[tuple[Candidate, float, float]]:
        """Find candidates within this region and compute their metrics.

        Args:
            candidates: Candidates to search through
            anchor: The anchor element's bbox for distance/alignment calculation
            require_below: If True, candidate must be below anchor
            require_above: If True, candidate must be above anchor
            require_aligned: If True, candidate must be left-aligned with anchor

        Yields:
            Tuples of (candidate, distance, alignment_offset) for valid matches
        """
        # Fast filter by bbox overlap
        potential = filter_overlapping(candidates, self.bbox)

        for cand in potential:
            cb = cand.bbox

            # Check alignment constraint
            alignment_offset = cb.x0 - anchor.x0
            if require_aligned and abs(alignment_offset) > self.horizontal_tolerance:
                continue

            # Check vertical relationship
            if require_below:
                # Candidate should be below anchor
                if cb.y0 < anchor.y1 - self.vertical_tolerance:
                    continue
                distance = max(0.0, cb.y0 - anchor.y1)
            elif require_above:
                # Candidate should be above anchor
                if cb.y1 > anchor.y0 + self.vertical_tolerance:
                    continue
                distance = max(0.0, anchor.y0 - cb.y1)
            else:
                # Just compute minimum distance
                distance = anchor.min_distance(cb)

            # Check max distance constraint
            if distance > self.max_distance:
                continue

            yield (cand, distance, alignment_offset)


# =============================================================================
# Score Model
# =============================================================================


class _PartPairScore(Score):
    """Internal score representation for part pairing classification.

    Uses generic Candidate[T] types to enable automatic constraint mapping.
    SchemaConstraintGenerator matches:
    - Candidate[PartCount] → Part.count
    - Candidate[PartImage] → Part.diagram
    - Candidate[PartNumber] → Part.number
    - Candidate[PieceLength] → Part.length

    IMPORTANT: The score() method must incorporate all factors that distinguish
    good pairings from bad ones (distance, alignment) so the constraint solver
    can select optimal one-to-one matchings.
    """

    distance: float
    """Vertical distance between image bottom and count top (pixels)."""

    alignment_offset: float
    """Horizontal alignment offset between image and count left edges (pixels)."""

    part_count_candidate: Candidate[PartCount]
    """The part_count candidate (maps to Part.count)."""

    part_image_candidate: Candidate[PartImage]
    """The part_image candidate (maps to Part.diagram)."""

    part_number_candidate: Candidate[PartNumber] | None = None
    """The part_number candidate (optional, maps to Part.number)."""

    piece_length_candidate: Candidate[PieceLength] | None = None
    """The piece_length candidate (optional, maps to Part.length)."""

    def sort_key(self) -> tuple[float, float]:
        """Return a tuple for sorting candidates.

        We prefer:
        1. Smaller vertical distance (closer image)
        2. Lower y-coordinate of image (top-down)
        """
        return (self.distance, self.part_image_candidate.bbox.y0)

    def score(self) -> Weight:
        """Calculate final weighted score for solver optimization.

        The score incorporates multiple factors critical for the constraint
        solver to select optimal pairings:

        1. Distance score (weight: 60%): Exponential decay based on vertical
           distance. Parts are typically close together (image directly above count).

        2. Alignment score (weight: 40%): Triangular score where perfect left
           alignment (offset=0) gives 1.0, degrading to 0 at max tolerance.

        Returns:
            Weight in [0.0, 1.0] where higher is better.
        """
        # Distance score: exponential decay, scale=10 means:
        # distance=0 → 1.0, distance=5 → 0.61, distance=10 → 0.37, distance=20 → 0.14
        distance_score = score_exponential_decay(self.distance, scale=10.0)

        # Alignment score: perfect alignment (0) is best, degrades linearly
        # offset=0 → 1.0, offset>=8 → 0.0 (8 pixels is a generous tolerance)
        alignment_score = score_triangular(
            val=abs(self.alignment_offset),
            min_val=0.0,
            ideal_val=0.0,
            max_val=8.0,
        )

        # Weighted combination: distance is more important than alignment
        base_score = 0.6 * distance_score + 0.4 * alignment_score

        # Small boost for optional components being present
        if self.part_number_candidate:
            base_score = min(1.0, base_score + 0.01)
        if self.piece_length_candidate:
            base_score = min(1.0, base_score + 0.01)

        return base_score


class PartsClassifier(LabelClassifier):
    """Classifier for assembling complete Part elements.

    This classifier combines previously classified components:
    - Part counts (e.g., "2x")
    - Part images (images above the counts)
    - Part numbers (optional element IDs)
    - Piece lengths (optional "1:1" indicators)

    It pairs counts with the closest image directly above them.
    """

    output = "part"
    requires = frozenset({"part_count", "part_image", "part_number", "piece_length"})

    def declare_constraints(
        self, model: ConstraintModel, result: ClassificationResult
    ) -> None:
        """Declare constraints for part candidates.

        Note: Child uniqueness constraints (each part_count, part_image,
        part_number, piece_length can only be used by one part) are handled
        automatically by SchemaConstraintGenerator.add_child_uniqueness_constraints().
        """
        pass  # All constraints now handled by schema generator

    def _score(self, result: ClassificationResult) -> None:
        """Score part pairings using anchor-based spatial chaining.

        Search chain (anchored on PartImage):
        1. For each PartImage, search for PartCounts directly below
        2. For each (PartImage, PartCount) pair, search for PartNumbers below the count
        3. Search for PieceLengths near the PartImage

        Creates Part candidates for ALL valid combinations. The constraint solver
        selects the optimal one-to-one matching based on scores.
        """
        page_data = result.page_data
        page_width = page_data.bbox.width

        # Get all candidates
        part_image_candidates = result.get_scored_candidates("part_image")
        part_count_candidates = result.get_scored_candidates("part_count")
        part_number_candidates = result.get_scored_candidates("part_number")
        piece_length_candidates = result.get_scored_candidates("piece_length")

        if not part_image_candidates:
            log.debug(
                "[parts] No part_image candidates found on page %s",
                page_data.page_number,
            )
            return

        if not part_count_candidates:
            log.debug(
                "[parts] No part_count candidates found on page %s",
                page_data.page_number,
            )
            return

        log.debug(
            "[parts] page=%s part_images=%d part_counts=%d part_numbers=%d "
            "piece_lengths=%d",
            page_data.page_number,
            len(part_image_candidates),
            len(part_count_candidates),
            len(part_number_candidates),
            len(piece_length_candidates),
        )

        # Horizontal alignment tolerance scales with page width
        h_tolerance = max(2.0, 0.02 * page_width)

        # Chain 1: PartImage → PartCount (count is directly below image)
        for img_cand in part_image_candidates:
            # Search for PartCounts below this image
            count_region = SearchRegion.below_and_aligned(
                img_cand.bbox,
                max_distance=10.0,
                vertical_tolerance=2.0,
                horizontal_tolerance=h_tolerance,
            )

            # Get all valid PartCounts below this image
            count_matches = list(
                count_region.find_candidates(
                    part_count_candidates,
                    img_cand.bbox,
                    require_below=True,
                    require_aligned=True,
                )
            )

            if not count_matches:
                continue

            # For each valid (image, count) pair, find optional components
            for count_cand, img_to_count_dist, img_to_count_align in count_matches:
                # Chain 2: PartCount → PartNumber (number is below count)
                number_region = SearchRegion.below_and_aligned(
                    count_cand.bbox,
                    max_distance=15.0,  # Numbers can be a bit further
                    vertical_tolerance=5.0,
                    horizontal_tolerance=3.0,
                )
                number_matches = list(
                    number_region.find_candidates(
                        part_number_candidates,
                        count_cand.bbox,
                        require_below=True,
                        require_aligned=True,
                    )
                )

                # Chain 3: PartImage → PieceLength (near image)
                length_region = SearchRegion.nearby(img_cand.bbox, max_distance=10.0)
                length_matches = list(
                    length_region.find_candidates(
                        piece_length_candidates,
                        img_cand.bbox,
                        require_below=False,
                        require_above=False,
                        require_aligned=False,
                    )
                )

                # Create Part candidates for valid combinations
                # For optional components: if valid matches exist, only use those.
                # If no matches, use None (the component is truly optional).
                # This avoids creating redundant candidates (e.g., with and without
                # part_number when a valid number exists).
                number_options: list[tuple[Candidate, float, float] | None] = (
                    list(number_matches) if number_matches else [None]
                )
                length_options: list[tuple[Candidate, float, float] | None] = (
                    list(length_matches) if length_matches else [None]
                )

                for number_match in number_options:
                    for length_match in length_options:
                        self._create_part_candidate(
                            result,
                            img_cand=img_cand,
                            count_cand=count_cand,
                            img_to_count_distance=img_to_count_dist,
                            img_to_count_alignment=img_to_count_align,
                            number_match=number_match,
                            length_match=length_match,
                        )

        # Log candidate statistics
        part_candidates = result.get_scored_candidates("part")
        log.debug(
            "[parts] Created %d Part candidates from %d images and %d counts",
            len(part_candidates),
            len(part_image_candidates),
            len(part_count_candidates),
        )

    def _create_part_candidate(
        self,
        result: ClassificationResult,
        *,
        img_cand: Candidate,
        count_cand: Candidate,
        img_to_count_distance: float,
        img_to_count_alignment: float,
        number_match: tuple[Candidate, float, float] | None,
        length_match: tuple[Candidate, float, float] | None,
    ) -> None:
        """Create a Part candidate from matched components.

        Args:
            result: Classification result to add candidate to
            img_cand: The PartImage candidate (anchor)
            count_cand: The PartCount candidate
            img_to_count_distance: Vertical distance from image to count
            img_to_count_alignment: Horizontal alignment offset
            number_match: Optional (candidate, distance, alignment) for PartNumber
            length_match: Optional (candidate, distance, alignment) for PieceLength
        """
        number_cand = number_match[0] if number_match else None
        length_cand = length_match[0] if length_match else None

        # Create score details
        score_details = _PartPairScore(
            distance=img_to_count_distance,
            alignment_offset=img_to_count_alignment,
            part_image_candidate=img_cand,
            part_count_candidate=count_cand,
            part_number_candidate=number_cand,
            piece_length_candidate=length_cand,
        )

        # Compute bounding box from all components
        bbox = img_cand.bbox.union(count_cand.bbox)
        if number_cand:
            bbox = bbox.union(number_cand.bbox)
        if length_cand:
            bbox = bbox.union(length_cand.bbox)

        result.add_candidate(
            Candidate(
                bbox=bbox,
                label="part",
                score=score_details.score(),
                score_details=score_details,
                # Part is composite - conflicts handled via constraints
                source_blocks=[],
            )
        )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Part:
        """Construct a Part from a single candidate's score details.

        Validates child candidates and extracts their constructed elements.
        """
        assert isinstance(candidate.score_details, _PartPairScore)
        ps = candidate.score_details

        # Validate and construct part_count from candidate
        try:
            part_count = result.build(ps.part_count_candidate)
            assert isinstance(part_count, PartCount)
        except Exception as e:
            raise ValueError(f"Failed to construct mandatory part_count: {e}") from e

        # Validate and construct part_image from candidate
        try:
            part_image = result.build(ps.part_image_candidate)
            assert isinstance(part_image, PartImage)
        except Exception as e:
            raise ValueError(f"Failed to construct mandatory part_image: {e}") from e

        # Extract optional part_number from candidate
        part_number: PartNumber | None = None
        if ps.part_number_candidate:
            try:
                part_number_elem = result.build(ps.part_number_candidate)
                assert isinstance(part_number_elem, PartNumber)
                part_number = part_number_elem
            except Exception as e:
                log.warning(
                    "Failed to construct optional part_number at %s: %s",
                    ps.part_number_candidate.bbox,
                    e,
                )

        # Extract optional piece_length from candidate
        piece_length: PieceLength | None = None
        if ps.piece_length_candidate:
            try:
                piece_length_elem = result.build(ps.piece_length_candidate)
                assert isinstance(piece_length_elem, PieceLength)
                piece_length = piece_length_elem
            except Exception as e:
                log.warning(
                    "Failed to construct optional piece_length at %s: %s",
                    ps.piece_length_candidate.bbox,
                    e,
                )

        # Compute bbox from built children (not candidates) to include any
        # expansions like PartImage shine.
        bbox = part_count.bbox.union(part_image.bbox)
        if part_number:
            bbox = bbox.union(part_number.bbox)
        if piece_length:
            bbox = bbox.union(piece_length.bbox)

        return Part(
            bbox=bbox,
            count=part_count,
            number=part_number,
            length=piece_length,
            diagram=part_image,
        )
