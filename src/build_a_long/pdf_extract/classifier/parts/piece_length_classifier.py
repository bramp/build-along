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
from collections.abc import Sequence

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
)
from build_a_long.pdf_extract.classifier.rules import (
    FontSizeMatch,
    IsInstanceFilter,
    PieceLengthValueRule,
    Rule,
    TextContainerFitRule,
)
from build_a_long.pdf_extract.classifier.rules.scale import LinearScale
from build_a_long.pdf_extract.extractor.bbox import BBox, filter_contained
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PieceLength,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing, Text

log = logging.getLogger(__name__)


class PieceLengthClassifier(RuleBasedClassifier):
    """Classifier for piece length indicators."""

    output = "piece_length"
    requires = frozenset()

    @property
    def rules(self) -> Sequence[Rule]:
        hints = self.config.font_size_hints
        # Prefer part_count_size, fall back to catalog_part_count_size
        expected_size = hints.part_count_size or hints.catalog_part_count_size or 10.0

        return [
            IsInstanceFilter(Text),
            PieceLengthValueRule(
                weight=1.0,
                required=True,
                name="Value",
            ),
            TextContainerFitRule(
                weight=1.0,
                required=True,
                name="ContainerFit",
            ),
            FontSizeMatch(
                scale=LinearScale(
                    {
                        expected_size * 0.5: 0.0,
                        expected_size: 1.0,
                        expected_size * 1.5: 0.0,
                    }
                ),
                weight=1.0,
                name="FontSize",
            ),
        ]

    def _get_additional_source_blocks(
        self, block: Blocks, result: ClassificationResult
    ) -> Sequence[Blocks]:
        """Include containing circle drawings as source blocks.

        This ensures the circle drawings around piece length numbers are
        marked as consumed when the piece_length is built.
        """
        # Start with default effects (e.g. shadows of the text itself)
        additional = list(super()._get_additional_source_blocks(block, result))

        if not isinstance(block, Text):
            return additional

        drawings = [b for b in result.page_data.blocks if isinstance(b, Drawing)]
        containing_drawing = self._find_smallest_containing_drawing(block, drawings)

        if not containing_drawing:
            return additional

        # Add the containing drawing
        if containing_drawing.id not in {b.id for b in additional}:
            additional.append(containing_drawing)

        # Find any other contained drawings (e.g. concentric circles)
        expanded_bbox = BBox.union(block.bbox, containing_drawing.bbox).expand(3.0)
        contained = filter_contained(drawings, expanded_bbox)
        seen_ids = {b.id for b in additional}
        for d in contained:
            if d.id not in seen_ids:
                additional.append(d)

        return additional

    def build(self, candidate: Candidate, result: ClassificationResult) -> PieceLength:
        """Construct a PieceLength element from a single candidate."""
        # Get the text block
        text_block = next(b for b in candidate.source_blocks if isinstance(b, Text))
        assert isinstance(text_block, Text)

        # Parse value
        value = int(text_block.text.strip())

        # Use the candidate's bbox which is already the union of all source_blocks
        # (text + containing drawings) computed by RuleBasedClassifier.
        return PieceLength(value=value, bbox=candidate.bbox)

    def _find_smallest_containing_drawing(
        self, text: Text, drawings: list[Drawing]
    ) -> Drawing | None:
        """Find the smallest drawing that contains the text.

        Re-implemented helper for build time.
        """
        containing_drawing = None
        smallest_area = float("inf")
        text_area = text.bbox.area
        MAX_AREA_RATIO = 6.0

        for drawing in drawings:
            if drawing.bbox.contains(text.bbox):
                drawing_area = drawing.bbox.area
                if text_area > 0 and drawing_area / text_area > MAX_AREA_RATIO:
                    continue
                if drawing_area < smallest_area:
                    smallest_area = drawing_area
                    containing_drawing = drawing

        return containing_drawing
