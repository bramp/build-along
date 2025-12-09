"""
Piece length classifier.

Purpose
-------
Identify small numbers surrounded by circles/ovals that indicate piece lengths.
These appear in the top-right of part images and use smaller font sizes than
step numbers.

Key Characteristics
-------------------
- Small font size (typically smaller than step numbers)
- Surrounded by a Drawing element (circle/oval)
- Located in top-right area of part image
- Single digit or small number (typically 1-32)
- Can appear on any page type

Distinguishing from Step Numbers
---------------------------------
- Font size: Piece lengths use smaller fonts
- Context: Nested inside Drawing elements (circles)
- Position: Top-right of part image vs. step numbers below parts list
- Value range: Usually 1-32 vs. step numbers can be higher
"""

import logging

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import BBox, filter_contained
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PieceLength,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Text

log = logging.getLogger(__name__)


class _PieceLengthScore(Score):
    """Internal score representation for piece length classification."""

    text_score: float
    """Score based on text matching simple number pattern (0.0-1.0)."""

    context_score: float
    """Score based on being nested in a Drawing (circle) (0.0-1.0)."""

    font_size_score: float
    """Score based on small font size (0.0-1.0)."""

    value: int | None = None
    """The parsed piece length value (1-32)."""

    containing_drawing: Drawing | None = None
    """The Drawing element that contains this piece length text."""

    def score(self) -> Weight:
        """Calculate final score from components.

        All three components are equally weighted.
        """
        return (self.text_score + self.context_score + self.font_size_score) / 3.0


class PieceLengthClassifier(LabelClassifier):
    """Classifier for piece length indicators."""

    output = "piece_length"
    requires = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        """Score text blocks and create candidates.

        Looks for small numbers (1-32) spatially contained within Drawing
        elements (circles/ovals) with small font sizes.
        """
        page_data = result.page_data
        if not page_data.blocks:
            return

        # Find all Drawing elements that could be circles
        drawings = [b for b in page_data.blocks if isinstance(b, Drawing)]

        # Find all Text blocks
        text_blocks = [b for b in page_data.blocks if isinstance(b, Text)]

        log.debug(
            "[piece_length] Found %d drawings and %d text blocks on page",
            len(drawings),
            len(text_blocks),
        )

        # Process each text block to find piece length candidates
        for text in text_blocks:
            # Find the smallest drawing containing this text
            containing_drawing = self._find_smallest_containing_drawing(text, drawings)
            if not containing_drawing:
                continue

            # Try to parse as a valid piece length value
            value = self._parse_piece_length_value(text)
            if value is None:
                continue

            log.debug(
                "[piece_length] Candidate: text='%s' id=%d font_size=%s "
                "in drawing id=%d",
                text.text,
                text.id,
                text.font_size,
                containing_drawing.id,
            )

            # Calculate scores
            text_score = 1.0  # Matched simple number pattern
            context_score = self._score_drawing_fit(text, containing_drawing)
            font_size_score = self._score_piece_length_font_size(text)

            # Piece lengths MUST be in a circle - if context_score is low,
            # heavily penalize. This prevents misclassifying step numbers
            # as piece lengths.
            if context_score < 0.5:
                # Not in a tight circle - unlikely to be a piece length
                context_score *= 0.1  # Reduce to near-zero

            detail_score = _PieceLengthScore(
                text_score=text_score,
                context_score=context_score,
                font_size_score=font_size_score,
                value=value,
                containing_drawing=containing_drawing,
            )

            combined = detail_score.score()

            log.debug(
                "[piece_length] Score for '%s': combined=%.3f "
                "(text=%.3f, context=%.3f, font_size=%.3f)",
                text.text,
                combined,
                text_score,
                context_score,
                font_size_score,
            )

            # Only create candidates with reasonable scores
            if combined < 0.3:
                continue

            # Include text and containing drawing in source blocks
            # Bbox should be the union of both
            bbox = BBox.union(text.bbox, containing_drawing.bbox)

            # Find all drawings that are part of this piece length indicator.
            # Use an expanded bbox to capture outer rings/circles that may
            # extend slightly beyond the core indicator.
            expanded_bbox = bbox.expand(3.0)
            contained = filter_contained(drawings, expanded_bbox)

            source_blocks: list[Drawing | Text] = [text]
            source_blocks.extend(contained)

            # Update bbox to include all contained drawings
            for drawing in contained:
                bbox = BBox.union(bbox, drawing.bbox)

            result.add_candidate(
                Candidate(
                    bbox=bbox,
                    label="piece_length",
                    score=combined,
                    score_details=detail_score,
                    source_blocks=list(source_blocks),
                ),
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> PieceLength:
        """Construct a PieceLength element from a single candidate."""
        # Get score details
        detail_score = candidate.score_details
        assert isinstance(detail_score, _PieceLengthScore)
        assert detail_score.value is not None

        # Use the candidate's bbox (already the union of text and drawing)
        return PieceLength(value=detail_score.value, bbox=candidate.bbox)

    def _create_piece_length_candidate(
        self, text: Text, drawings: list[Drawing]
    ) -> Candidate[PieceLength] | None:
        """Create a piece length candidate from a text block if valid.

        Args:
            text: Text block to evaluate
            drawings: All drawings on the page

        Returns:
            Candidate if text is a valid piece length, None otherwise
        """
        # Find the smallest drawing containing this text
        containing_drawing = self._find_smallest_containing_drawing(text, drawings)
        if not containing_drawing:
            return None

        # Try to parse as a valid piece length value
        value = self._parse_piece_length_value(text)
        if value is None:
            return None

        log.debug(
            "[piece_length] Candidate: text='%s' id=%d font_size=%s in drawing id=%d",
            text.text,
            text.id,
            text.font_size,
            containing_drawing.id,
        )

        # Calculate scores
        text_score = 1.0  # Matched simple number pattern
        context_score = self._score_drawing_fit(text, containing_drawing)
        font_size_score = self._score_piece_length_font_size(text)

        detail_score = _PieceLengthScore(
            text_score=text_score,
            context_score=context_score,
            font_size_score=font_size_score,
        )

        combined = detail_score.score()

        log.debug(
            "[piece_length] Score for '%s': combined=%.3f "
            "(text=%.3f, context=%.3f, font_size=%.3f)",
            text.text,
            combined,
            text_score,
            context_score,
            font_size_score,
        )

        # Only create candidates with reasonable scores
        if combined < 0.3:
            return None

        constructed_elem = PieceLength(
            value=value,
            bbox=text.bbox,
        )

        return Candidate(
            bbox=text.bbox,
            label="piece_length",
            score=combined,
            score_details=detail_score,
            constructed=constructed_elem,
            source_blocks=[text],
        )

    def _find_smallest_containing_drawing(
        self, text: Text, drawings: list[Drawing]
    ) -> Drawing | None:
        """Find the smallest drawing that contains the text.

        This avoids matching page-sized background drawings by preferring
        the tightest-fitting container and filtering out drawings that are
        too large relative to the text.

        Args:
            text: Text block to find container for
            drawings: All drawings on the page

        Returns:
            Smallest containing drawing, or None if not contained
        """
        containing_drawing = None
        smallest_area = float("inf")

        text_area = text.bbox.area
        # Maximum ratio of drawing area to text area
        # A circle around text should be roughly 2-4x the text area,
        # but we want to filter out page-sized backgrounds early.
        # Allow slightly more than the ideal 4x to handle edge cases.
        MAX_AREA_RATIO = 6.0

        for drawing in drawings:
            # Check if text bbox is fully contained in drawing bbox
            if drawing.bbox.contains(text.bbox):
                # Calculate drawing area
                drawing_area = drawing.bbox.area

                # Skip drawings that are way too large (page backgrounds)
                if text_area > 0 and drawing_area / text_area > MAX_AREA_RATIO:
                    continue

                # Keep the smallest containing drawing
                if drawing_area < smallest_area:
                    smallest_area = drawing_area
                    containing_drawing = drawing

        return containing_drawing

    def _parse_piece_length_value(self, text: Text) -> int | None:
        """Parse text as a piece length value.

        Args:
            text: Text block to parse

        Returns:
            Integer value if valid (1-32), None otherwise
        """
        try:
            value = int(text.text.strip())
        except ValueError:
            return None

        # Piece lengths are typically 1-32 (reasonable LEGO piece sizes)
        # TODO There are actually specific acceptable values, maybe later use them.
        if not (1 <= value <= 32):
            return None

        return value

    def _score_piece_length_font_size(self, text: Text) -> float:
        """Score based on font size - should match part_count_size.

        Uses font hints to distinguish piece lengths from other elements.
        Piece lengths typically use the same font size as part counts.

        Args:
            text: Text block to score

        Returns:
            Score from 0.0 to 1.0, where 1.0 is ideal piece length font size
        """
        hints = self.config.font_size_hints

        # Prefer part_count_size, fall back to catalog_part_count_size
        expected_size = hints.part_count_size or hints.catalog_part_count_size

        return self._score_font_size(text, expected_size)

    def _score_drawing_fit(self, text: Text, drawing: Drawing) -> float:
        """Score how well the drawing size fits the text.

        A good piece length indicator has a small circle tightly around
        the number. Score is based on the ratio of drawing area to text area.

        Args:
            text: The text block
            drawing: The containing drawing

        Returns:
            Score from 0.0 to 1.0, where 1.0 means perfect fit
        """
        text_area = text.bbox.area
        drawing_area = drawing.bbox.area

        if text_area <= 0:
            return 0.0

        # Calculate the ratio of drawing area to text area
        # Ideal ratio: 2-4x (circle slightly larger than text)
        # A circle around text would be ~2-3x the text area
        ratio = drawing_area / text_area

        if 2.0 <= ratio <= 4.0:
            # Ideal size - tight circle around text
            return 1.0
        elif 1.0 <= ratio < 2.0:
            # Slightly too small, but acceptable
            return 0.8
        elif 4.0 < ratio <= 10.0:
            # A bit too large, but still reasonable
            return 0.6
        elif 10.0 < ratio <= 50.0:
            # Much too large - probably not a circle indicator
            return 0.3
        else:
            # Way too large (page background) or too small - very unlikely
            return 0.1
