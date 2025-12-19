"""
Scale classifier.

Purpose
-------
Identify 1:1 scale indicators that show the actual printed size of a piece.
These typically appear at the bottom of instruction pages, containing a piece
length indicator (number in a circle) and "1:1" text within a bounding box.

Key Characteristics
-------------------
- Contains "1:1" text
- Enclosed in a white rectangular box (Drawing)
- Has a PieceLength indicator inside (number in circle)
- Usually at the bottom of the page

The Scale element helps builders verify piece lengths by measuring directly
against the printed instruction manual.
"""

import logging
from typing import ClassVar

from build_a_long.pdf_extract.classifier.block_filter import (
    find_text_outline_effects,
)
from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.classifier.text import is_scale_text
from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    filter_contained,
    find_smallest_containing_box,
    group_by_similar_bbox,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PieceLength,
    Scale,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing, Text

log = logging.getLogger(__name__)


class _ScaleScore(Score):
    """Score details for Scale candidates."""

    piece_length_candidate: Candidate | None = None
    """The PieceLength candidate associated with this scale."""

    def score(self) -> float:
        """Calculate overall score."""
        # Base score for having scale text in a bounding box
        base = 0.5

        # Bonus for having piece length inside
        if self.piece_length_candidate:
            base += 0.5

        return base


class ScaleClassifier(LabelClassifier):
    """Classifier for 1:1 scale indicators."""

    output: ClassVar[str] = "scale"
    requires: ClassVar[frozenset[str]] = frozenset({"piece_length"})

    def _score(self, result: ClassificationResult) -> None:
        """Score potential Scale indicators.

        Algorithm:
        1. Find Text blocks with "1:1" pattern
        2. Find the smallest containing Drawing box
        3. Look for PieceLength and PartImage candidates inside that box
        4. Score based on completeness
        """
        page_data = result.page_data

        # Get piece_length candidates
        piece_length_candidates = result.get_scored_candidates("piece_length")

        # Get all Drawing blocks for finding containers
        drawings = [b for b in page_data.blocks if isinstance(b, Drawing)]

        # Find Text blocks with "1:1" pattern
        for block in page_data.blocks:
            if not isinstance(block, Text):
                continue

            # Check for "1:1" pattern using the text extractor
            if not is_scale_text(block.text):
                continue

            log.debug(
                "[scale] Found '1:1' text at %s: '%s'",
                block.bbox,
                block.text,
            )

            # Find the smallest containing Drawing box
            containing_box = find_smallest_containing_box(block.bbox, drawings)

            if not containing_box:
                log.debug("[scale] No containing box found, skipping")
                continue

            log.debug(
                "[scale] Found containing box at %s",
                containing_box.bbox,
            )
            search_bbox = containing_box.bbox

            # Look for PieceLength candidates inside the box
            piece_length_candidate = self._find_candidate_in_box(
                search_bbox, piece_length_candidates
            )

            if piece_length_candidate:
                log.debug(
                    "[scale] Found piece_length at %s inside box",
                    piece_length_candidate.bbox,
                )

            # Create score
            score = _ScaleScore(
                piece_length_candidate=piece_length_candidate,
            )

            # Collect all source blocks:
            # 1. The 1:1 text and its outline effects
            # 2. All drawings with similar bbox to the containing box (borders)
            # 3. All remaining drawings inside the box
            #
            # Note: The "part image" shown in the Scale is often composed of
            # vector Drawing blocks rather than an Image block. Rather than
            # introducing a VectorPartImage type, we simply capture all drawings
            # inside the Scale box as source_blocks. The PieceLength classifier
            # will claim the ruler/circle drawings it needs, and the remaining
            # drawings (which visually represent the part) become part of Scale's
            # source_blocks.
            source_blocks: list[Blocks] = [block]

            # Add text outline effects (shadows, etc.)
            text_effects = find_text_outline_effects(block, page_data.blocks)
            source_blocks.extend(text_effects)

            # Find similar drawings to the containing box (border/shadow effects)
            similar_groups = group_by_similar_bbox(drawings, tolerance=2.0)
            for group in similar_groups:
                if containing_box in group:
                    source_blocks.extend(group)
                    break

            # Capture all drawings inside the box (vector part image, ruler, etc.)
            # PieceLength will claim what it needs; the rest stays with Scale.
            contained_drawings = filter_contained(drawings, search_bbox)
            for drawing in contained_drawings:
                if drawing not in source_blocks:
                    source_blocks.append(drawing)

            # Find similar text blocks (text with same bbox = drop shadow/outline)
            texts = [b for b in page_data.blocks if isinstance(b, Text)]
            for text in texts:
                if text is not block and text.bbox.similar(block.bbox, tolerance=2.0):
                    source_blocks.append(text)

            log.debug(
                "[scale] Collected %d source blocks for scale at %s",
                len(source_blocks),
                containing_box.bbox,
            )

            result.add_candidate(
                Candidate(
                    label=self.output,
                    bbox=containing_box.bbox,
                    score=score.score(),
                    score_details=score,
                    source_blocks=source_blocks,
                )
            )

    def _find_candidate_in_box(
        self,
        box_bbox: BBox,
        candidates: list[Candidate],
    ) -> Candidate | None:
        """Find the best candidate inside the given box.

        Args:
            box_bbox: The bounding box to search within
            candidates: List of candidates to search

        Returns:
            The highest-scoring candidate inside the box, or None
        """
        # Find candidates contained in the box
        contained = filter_contained(candidates, box_bbox)

        if not contained:
            return None

        # Return the highest-scoring one
        return max(contained, key=lambda c: c.score)

    def build(self, candidate: Candidate, result: ClassificationResult) -> Scale:
        """Construct a Scale element from a candidate."""
        score = candidate.score_details
        assert isinstance(score, _ScaleScore)

        # Build the PieceLength if we have one
        piece_length: PieceLength | None = None
        if score.piece_length_candidate:
            pl_elem = result.build(score.piece_length_candidate)
            assert isinstance(pl_elem, PieceLength)
            piece_length = pl_elem

        # If no piece length found, we can't build a valid Scale
        if piece_length is None:
            raise ValueError("Scale requires a PieceLength element")

        return Scale(
            bbox=candidate.bbox,
            length=piece_length,
        )
