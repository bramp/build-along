"""
Parts classifier.

Purpose
-------
Associate each part_count text with exactly one image to create Part candidates.
These Part candidates will be consumed by PartsListClassifier to build PartsList
elements.

Heuristic
---------
- For each part_count Text, find candidate Images that are above
  (image.y1 <= count.y0 + VERT_EPS) and roughly left-aligned
  (|image.x0 - count.x0| <= ALIGN_EPS).
- Sort candidates by vertical distance (count.y0 - image.y1), then
  greedily match to enforce one-to-one pairing between part counts and
  images.
- Create Part candidates with the paired PartCount and Drawing (image).

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
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LegoPageElements,
    Part,
    PartCount,
    PartNumber,
    PieceLength,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


@dataclass
class _PartPairScore:
    """Internal score representation for part pairing."""

    distance: float
    """Vertical distance from part count text to image (lower is better)."""

    part_count_candidate: Candidate
    """The part_count candidate (not the constructed element)."""

    image: Image
    """The image element (will become diagram in Part)."""

    part_number_candidate: Candidate | None
    """The associated part_number candidate (if any)."""

    piece_length_candidate: Candidate | None
    """The associated piece_length candidate (if any)."""

    def sort_key(self) -> float:
        """Return sort key for matching (prefer smaller distance)."""
        return self.distance


# TODO Should this be called PartClassifier instead?
@dataclass(frozen=True)
class PartsClassifier(LabelClassifier):
    """Classifier for Part elements (pairs of part_count + image)."""

    output = "part"
    requires = frozenset({"part_count", "part_number", "piece_length"})

    def _score(self, result: ClassificationResult) -> None:
        """Score part pairings and create candidates.

        Creates candidates with score details containing references to parent
        candidates (not constructed elements), following the recommended pattern
        for dependent classifiers.
        """
        page_data = result.page_data

        # Get part_count candidates (not elements!) using new API
        part_count_candidates = result.get_scored_candidates(
            "part_count",
            valid_only=False,
            exclude_failed=True,
        )

        # Get optional part_number candidates
        part_number_candidates = result.get_scored_candidates(
            "part_number",
            valid_only=False,
            exclude_failed=True,
        )

        # Get optional piece_length candidates
        piece_length_candidates = result.get_scored_candidates(
            "piece_length",
            valid_only=False,
            exclude_failed=True,
        )

        images: list[Image] = [e for e in page_data.blocks if isinstance(e, Image)]

        if not part_count_candidates:
            log.debug(
                "[parts] No part_count candidates found on page %s",
                page_data.page_number,
            )
            return

        if not images:
            log.debug(
                "[parts] No images found on page %s",
                page_data.page_number,
            )
            return

        log.debug(
            "[parts] page=%s part_counts=%d images=%d",
            page_data.page_number,
            len(part_count_candidates),
            len(images),
        )

        # Build candidate pairings using existing helper
        candidate_edges = self._build_candidate_edges(
            part_count_candidates,
            images,
            page_data.bbox.width if page_data.bbox else 100.0,
        )

        log.debug(
            "[parts] Built %d candidate edges for %d part_counts",
            len(candidate_edges),
            len(part_count_candidates),
        )

        if not candidate_edges:
            log.debug(
                "[parts] No valid part-image pairs found on page %s",
                page_data.page_number,
            )
            return

        # Sort by distance (prefer closer pairs)
        candidate_edges.sort(key=lambda ps: ps.sort_key())

        # Greedy matching to enforce one-to-one pairing
        used_count_candidates: set[int] = set()
        used_images: set[int] = set()

        for ps in candidate_edges:
            count_cand_id = id(ps.part_count_candidate)
            image_id = id(ps.image)

            if count_cand_id in used_count_candidates or image_id in used_images:
                continue

            # Mark as used
            used_count_candidates.add(count_cand_id)
            used_images.add(image_id)

            # Find associated candidates (not elements!)
            part_number_cand = self._find_part_number_candidate(
                ps.part_count_candidate, part_number_candidates
            )
            piece_length_cand = self._find_piece_length_candidate(
                ps.image, piece_length_candidates
            )

            # Create enhanced score with candidate references
            enhanced_score = _PartPairScore(
                distance=ps.distance,
                part_count_candidate=ps.part_count_candidate,
                image=ps.image,
                part_number_candidate=part_number_cand,
                piece_length_candidate=piece_length_cand,
            )

            # Create bounding box for the Part
            bbox = ps.part_count_candidate.bbox.union(ps.image.bbox)
            if part_number_cand:
                bbox = bbox.union(part_number_cand.bbox)
            if piece_length_cand:
                bbox = bbox.union(piece_length_cand.bbox)

            # Create candidate WITHOUT construction
            candidate = Candidate(
                bbox=bbox,
                label="part",
                score=1.0,
                score_details=enhanced_score,
                source_blocks=[ps.image],
            )

            result.add_candidate(candidate)

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Construct a Part from a single candidate's score details.

        Validates parent candidates and extracts their constructed elements.
        """
        assert isinstance(candidate.score_details, _PartPairScore)
        ps = candidate.score_details

        # Validate and construct part_count from candidate
        try:
            part_count_elem = result.build(ps.part_count_candidate)
            assert isinstance(part_count_elem, PartCount)
            part_count = part_count_elem
        except Exception as e:
            raise ValueError(f"Failed to construct mandatory part_count: {e}") from e

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

        # Wrap Image in Drawing for the diagram field
        diagram = Drawing(bbox=ps.image.bbox, id=ps.image.id)

        return Part(
            bbox=candidate.bbox,
            count=part_count,
            number=part_number,
            length=piece_length,
            diagram=diagram,
        )

    def _build_candidate_edges(
        self,
        part_count_candidates: list[Candidate],
        images: list[Image],
        page_width: float,
    ) -> list[_PartPairScore]:
        """Build candidate pairings between part count candidates and images.

        Returns a list of score objects representing valid pairings.
        """
        VERT_EPS = 2.0  # allow minor overlap/touching
        ALIGN_EPS = max(2.0, 0.02 * page_width)

        edges: list[_PartPairScore] = []
        for pc_cand in part_count_candidates:
            cb = pc_cand.bbox
            for img in images:
                ib = img.bbox
                # Image should be above the count and left-aligned
                if ib.y1 <= cb.y0 + VERT_EPS and abs(ib.x0 - cb.x0) <= ALIGN_EPS:
                    distance = max(0.0, cb.y0 - ib.y1)
                    score = _PartPairScore(
                        distance=distance,
                        part_count_candidate=pc_cand,
                        image=img,
                        part_number_candidate=None,  # Will be filled during matching
                        piece_length_candidate=None,  # Will be filled during matching
                    )
                    edges.append(score)
        return edges

    def _find_part_number_candidate(
        self, part_count_candidate: Candidate, part_number_candidates: list[Candidate]
    ) -> Candidate | None:
        """Find the part_number candidate that belongs to this part_count candidate.

        The part_number should be directly below the part_count, left-aligned.

        Args:
            part_count_candidate: The part_count candidate to find a number for
            part_number_candidates: List of available part_number candidates

        Returns:
            The matching part_number candidate, or None if not found
        """
        VERT_EPS = 5.0  # Small vertical tolerance
        ALIGN_EPS = 3.0  # Horizontal alignment tolerance

        best_candidate = None
        best_distance = float("inf")

        for pn_cand in part_number_candidates:
            # Part number should be below the count
            if (
                pn_cand.bbox.y0 >= part_count_candidate.bbox.y1 - VERT_EPS
                and abs(pn_cand.bbox.x0 - part_count_candidate.bbox.x0) <= ALIGN_EPS
            ):
                # Calculate vertical distance
                distance = pn_cand.bbox.y0 - part_count_candidate.bbox.y1
                if distance < best_distance:
                    best_distance = distance
                    best_candidate = pn_cand

        return best_candidate

    def _find_piece_length_candidate(
        self, image: Image, piece_length_candidates: list[Candidate]
    ) -> Candidate | None:
        """Find the piece_length candidate that belongs to this part image.

        The piece_length should be in the top-right area of the image,
        within a small distance of the image bbox.

        Args:
            image: The Image to find a length for
            piece_length_candidates: List of available piece_length candidates

        Returns:
            The matching piece_length candidate, or None if not found
        """
        # Piece length should be very close to the image (within 10 units)
        # Use minimum distance between bboxes instead of containment check
        MAX_DISTANCE = 10.0

        best_candidate = None
        best_score = float("inf")

        for pl_cand in piece_length_candidates:
            # Calculate minimum distance between piece length and image
            distance = image.bbox.min_distance(pl_cand.bbox)

            # Skip if too far away
            if distance > MAX_DISTANCE:
                continue

            # Prefer piece lengths closer to top-right
            pl_center_x = (pl_cand.bbox.x0 + pl_cand.bbox.x1) / 2
            pl_center_y = (pl_cand.bbox.y0 + pl_cand.bbox.y1) / 2

            # Distance to top-right corner
            dx = image.bbox.x1 - pl_center_x
            dy = pl_center_y - image.bbox.y0

            # Combined score (prefer top-right position)
            score = dx * dx + dy * dy

            if score < best_score:
                best_score = score
                best_candidate = pl_cand

        return best_candidate
