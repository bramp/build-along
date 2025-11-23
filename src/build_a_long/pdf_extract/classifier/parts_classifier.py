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
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LegoPageElements,
    Part,
    PartCount,
    PartNumber,
    PieceLength,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Image,
)

log = logging.getLogger(__name__)


@dataclass
class _PartPairScore:
    """Internal score representation for part pairing."""

    distance: float
    """Vertical distance from part count text to image (lower is better)."""

    part_count: PartCount
    """The constructed PartCount element."""

    image: Image
    """The image element (will become diagram in Part)."""

    def sort_key(self) -> float:
        """Return sort key for matching (prefer smaller distance)."""
        return self.distance


# TODO Should this be called PartClassifier instead?
@dataclass(frozen=True)
class PartsClassifier(LabelClassifier):
    """Classifier for Part elements (pairs of part_count + image)."""

    outputs = frozenset({"part"})
    requires = frozenset({"part_count", "part_number", "piece_length"})

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
        """Evaluate elements and create scores for part pairings.

        Scores are based on vertical distance and horizontal alignment between
        part count elements and images.
        """
        page_data = result.page_data

        # Get part_count candidates with type safety, selecting by score
        part_counts = result.get_winners_by_score("part_count", PartCount)

        log.debug(
            "[parts] Found %d part_counts for pairing",
            len(part_counts),
        )

        if not part_counts:
            return

        # Get part_number candidates (optional, only on catalog pages)
        part_numbers = result.get_winners_by_score("part_number", PartNumber)

        # Get piece_length candidates (optional, can appear on any page type)
        piece_lengths = result.get_winners_by_score("piece_length", PieceLength)

        log.debug(
            "[parts] Retrieved %d piece_lengths from result",
            len(piece_lengths),
        )

        # Get all images on the page
        images: list[Image] = [e for e in page_data.blocks if isinstance(e, Image)]

        log.debug(
            "[parts] Found %d images on page for pairing with %d part_counts",
            len(images),
            len(part_counts),
        )

        if not images:
            return

        # Build candidate pairings and match them directly
        candidate_edges = self._build_candidate_edges(
            part_counts, images, page_data.bbox.width if page_data.bbox else 100.0
        )

        log.debug(
            "[parts] Built %d candidate edges for %d part_counts",
            len(candidate_edges),
            len(part_counts),
        )

        # Match and create Part candidates
        self._match_and_create_parts(
            candidate_edges, part_numbers, piece_lengths, result
        )

    def _build_candidate_edges(
        self,
        part_counts: list[PartCount],
        images: list[Image],
        page_width: float,
    ) -> list[_PartPairScore]:
        """Build candidate pairings between part counts and images.

        Returns a list of score objects representing valid pairings.
        """
        VERT_EPS = 2.0  # allow minor overlap/touching
        ALIGN_EPS = max(2.0, 0.02 * page_width)

        edges: list[_PartPairScore] = []
        for pc in part_counts:
            cb = pc.bbox
            for img in images:
                ib = img.bbox
                # Image should be above the count and left-aligned
                if ib.y1 <= cb.y0 + VERT_EPS and abs(ib.x0 - cb.x0) <= ALIGN_EPS:
                    distance = max(0.0, cb.y0 - ib.y1)
                    score = _PartPairScore(
                        distance=distance,
                        part_count=pc,
                        image=img,
                    )
                    edges.append(score)
        return edges

    def _match_and_create_parts(
        self,
        candidate_edges: list[_PartPairScore],
        part_numbers: list[PartNumber],
        piece_lengths: list[PieceLength],
        result: ClassificationResult,
    ) -> None:
        """Match part counts with images and create Part candidates.

        Args:
            candidate_edges: List of candidate pairings to consider
            part_numbers: List of PartNumber elements (for catalog pages)
            piece_lengths: List of PieceLength elements (for any page)
            result: Classification result to add Part candidates to
        """
        if not candidate_edges:
            log.debug("[parts] No candidate edges to match")
            return

        log.debug(
            "[parts] Matching %d candidate edges to create parts",
            len(candidate_edges),
        )

        # Sort by distance (closest pairs first)
        candidate_edges.sort(key=lambda score: score.sort_key())

        matched_counts: set[int] = set()
        matched_images: set[int] = set()

        for score in candidate_edges:
            pc = score.part_count
            img = score.image

            # Skip if already matched
            if id(pc) in matched_counts or id(img) in matched_images:
                continue

            matched_counts.add(id(pc))
            matched_images.add(id(img))

            # Find matching part_number (if any) - should be below the part_count
            part_number = self._find_part_number(pc, part_numbers)

            # Find matching piece_length (if any) - should be in top-right of image
            piece_length = self._find_piece_length(img, piece_lengths)

            # Create a Part from this pairing
            # The bbox is the union of the part_count, image, and part_number (if present)
            combined_bbox = BBox(
                x0=min(pc.bbox.x0, img.bbox.x0),
                y0=min(pc.bbox.y0, img.bbox.y0),
                x1=max(pc.bbox.x1, img.bbox.x1),
                y1=max(pc.bbox.y1, img.bbox.y1),
            )

            if part_number:
                combined_bbox = BBox(
                    x0=min(combined_bbox.x0, part_number.bbox.x0),
                    y0=min(combined_bbox.y0, part_number.bbox.y0),
                    x1=max(combined_bbox.x1, part_number.bbox.x1),
                    y1=max(combined_bbox.y1, part_number.bbox.y1),
                )

            part = Part(
                bbox=combined_bbox,
                count=pc,
                number=part_number,
                length=piece_length,
                # Note: diagram field is optional and not set here
                # The image is tracked via the score_details
            )

            # Create a candidate for this Part
            result.add_candidate(
                "part",
                Candidate(
                    bbox=combined_bbox,
                    label="part",
                    score=1.0,  # Matched based on distance
                    score_details=score,
                    constructed=part,
                    source_blocks=[],  # Synthetic element, no single source
                    failure_reason=None,
                ),
            )

    def _find_part_number(
        self, part_count: PartCount, part_numbers: list[PartNumber]
    ) -> PartNumber | None:
        """Find the part_number that belongs to this part_count.

        The part_number should be directly below the part_count,
        left-aligned.

        Args:
            part_count: The PartCount to find a number for
            part_numbers: List of available PartNumber elements

        Returns:
            The matching PartNumber, or None if not found
        """
        VERT_EPS = 5.0  # Small vertical tolerance
        ALIGN_EPS = 3.0  # Horizontal alignment tolerance

        best_number = None
        best_distance = float("inf")

        for pn in part_numbers:
            # Part number should be below the count
            if (
                pn.bbox.y0 >= part_count.bbox.y1 - VERT_EPS
                and abs(pn.bbox.x0 - part_count.bbox.x0) <= ALIGN_EPS
            ):
                # Calculate vertical distance
                distance = pn.bbox.y0 - part_count.bbox.y1
                if distance < best_distance:
                    best_distance = distance
                    best_number = pn

        return best_number

    def _find_piece_length(
        self, image: Image, piece_lengths: list[PieceLength]
    ) -> PieceLength | None:
        """Find the piece_length that belongs to this part image.

        The piece_length should be in the top-right area of the image,
        spatially contained within or very close to the image bbox.

        Args:
            image: The Image to find a length for
            piece_lengths: List of available PieceLength elements

        Returns:
            The matching PieceLength, or None if not found
        """
        # Piece length should be near top-right of image
        # Allow some tolerance for being slightly outside image bounds
        TOLERANCE = 5.0

        best_length = None
        best_score = float("inf")

        log.debug(
            f"Matching piece_length to image id={image.id} bbox={image.bbox} "
            f"from {len(piece_lengths)} available"
        )

        for pl in piece_lengths:
            # Check if piece length is in the vicinity of the image
            # (within or slightly outside the image bounds)
            if (
                pl.bbox.x0 >= image.bbox.x0 - TOLERANCE
                and pl.bbox.x1 <= image.bbox.x1 + TOLERANCE
                and pl.bbox.y0 >= image.bbox.y0 - TOLERANCE
                and pl.bbox.y1 <= image.bbox.y1 + TOLERANCE
            ):
                # Prefer piece lengths closer to top-right
                # Calculate distance from piece length center to image top-right
                pl_center_x = (pl.bbox.x0 + pl.bbox.x1) / 2
                pl_center_y = (pl.bbox.y0 + pl.bbox.y1) / 2

                # Distance to top-right corner
                dx = image.bbox.x1 - pl_center_x
                dy = pl_center_y - image.bbox.y0

                # Combined score (prefer top-right position)
                score = dx * dx + dy * dy

                log.debug(
                    f"  Candidate piece_length value={pl.value} bbox={pl.bbox} "
                    f"score={score:.2f}"
                )

                if score < best_score:
                    best_score = score
                    best_length = pl

        if best_length:
            log.debug(
                f"  Selected piece_length value={best_length.value} "
                f"for image id={image.id}"
            )
        else:
            log.debug(f"  No matching piece_length found for image id={image.id}")

        return best_length
