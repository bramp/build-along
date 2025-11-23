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
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import PieceLength
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Text

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PieceLengthScore:
    """Internal score representation for piece length classification."""

    text_score: float
    """Score based on text matching simple number pattern (0.0-1.0)."""

    context_score: float
    """Score based on being nested in a Drawing (circle) (0.0-1.0)."""

    font_size_score: float
    """Score based on small font size (0.0-1.0)."""

    def combined_score(self) -> float:
        """Calculate final score from components.

        All three components are equally weighted.
        """
        return (self.text_score + self.context_score + self.font_size_score) / 3.0


@dataclass(frozen=True)
class PieceLengthClassifier(LabelClassifier):
    """Classifier for piece length indicators."""

    outputs = frozenset({"piece_length"})
    requires = frozenset()

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for piece lengths.

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

        candidates_created = 0

        # Process each text block to find piece length candidates
        for text in text_blocks:
            candidate = self._create_piece_length_candidate(text, drawings)
            if candidate:
                result.add_candidate("piece_length", candidate)
                candidates_created += 1

        log.debug(
            "[piece_length] Created %d piece_length candidates",
            candidates_created,
        )

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

        combined = detail_score.combined_score()

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
            failure_reason=None,
        )

    def _find_smallest_containing_drawing(
        self, text: Text, drawings: list[Drawing]
    ) -> Drawing | None:
        """Find the smallest drawing that contains the text.

        This avoids matching page-sized background drawings by preferring
        the tightest-fitting container.

        Args:
            text: Text block to find container for
            drawings: All drawings on the page

        Returns:
            Smallest containing drawing, or None if not contained
        """
        containing_drawing = None
        smallest_area = float("inf")

        for drawing in drawings:
            # Check if text bbox is fully contained in drawing bbox
            if (
                drawing.bbox.x0 <= text.bbox.x0
                and text.bbox.x1 <= drawing.bbox.x1
                and drawing.bbox.y0 <= text.bbox.y0
                and text.bbox.y1 <= drawing.bbox.y1
            ):
                # Calculate drawing area
                drawing_area = (drawing.bbox.x1 - drawing.bbox.x0) * (
                    drawing.bbox.y1 - drawing.bbox.y0
                )

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
        """Score based on font size - should be >= part count, < step number.

        Uses font hints to distinguish piece lengths from other elements:
        - Piece lengths should be >= part_count_size (or catalog_part_count_size)
        - Piece lengths should be < step_number_size
        - This puts them in the range between part counts and step numbers

        Args:
            text: Text block to score

        Returns:
            Score from 0.0 to 1.0, where 1.0 is ideal piece length font size
        """
        if text.font_size is None:
            return 0.5  # Unknown, neutral score

        hints = self.config.font_size_hints
        font_size = text.font_size

        # Determine the expected range based on hints
        # Lower bound: part count sizes (whichever is available)
        lower_bound = hints.part_count_size or hints.catalog_part_count_size or 4.0

        # Upper bound: step number size
        upper_bound = hints.step_number_size or 10.0

        # Check if font size is in the ideal range
        if lower_bound <= font_size < upper_bound:
            # Perfect range
            return 1.0
        elif font_size < lower_bound:
            # Too small - penalize based on how far below
            diff = lower_bound - font_size
            return max(0.0, 1.0 - (diff / 4.0))
        else:
            # Too large (>= step_number_size) - heavily penalize
            return 0.1

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
