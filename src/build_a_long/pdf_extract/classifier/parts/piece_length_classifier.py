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
from build_a_long.pdf_extract.extractor.bbox import BBox, filter_contained
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PieceLength,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Text

log = logging.getLogger(__name__)


class PieceLengthClassifier(RuleBasedClassifier):
    """Classifier for piece length indicators."""

    output = "piece_length"
    requires = frozenset()

    @property
    def rules(self) -> list[Rule]:
        hints = self.config.font_size_hints
        # Prefer part_count_size, fall back to catalog_part_count_size
        expected_size = hints.part_count_size or hints.catalog_part_count_size

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
                target_size=expected_size,
                weight=1.0,
                name="FontSize",
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> PieceLength:
        """Construct a PieceLength element from a single candidate."""
        # Get the text block
        text_block = next(b for b in candidate.source_blocks if isinstance(b, Text))
        assert isinstance(text_block, Text)

        # Parse value
        value = int(text_block.text.strip())

        # Use the candidate's bbox (which currently is just the text bbox from RuleBasedClassifier)
        # We need to find the containing drawing again to include it in the final bbox
        # This duplicates logic from TextContainerFitRule, but it's necessary since
        # RuleBasedClassifier doesn't pass rule-internal state to build.
        # Alternatively, we could expand the bbox in _score if we overrode it,
        # but let's stick to standard flow.

        # Find containing drawing to expand bbox
        drawings = [b for b in result.page_data.blocks if isinstance(b, Drawing)]
        containing_drawing = self._find_smallest_containing_drawing(
            text_block, drawings
        )

        bbox = text_block.bbox
        if containing_drawing:
            bbox = BBox.union(text_block.bbox, containing_drawing.bbox)

            # Also include any other contained drawings (e.g. concentric circles)
            expanded_bbox = bbox.expand(3.0)
            contained = filter_contained(drawings, expanded_bbox)
            for d in contained:
                bbox = BBox.union(bbox, d.bbox)

        return PieceLength(value=value, bbox=bbox)

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
