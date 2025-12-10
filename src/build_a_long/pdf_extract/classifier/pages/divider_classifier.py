"""
Divider classifier.

Purpose
-------
Identify visual divider lines that separate sections of a LEGO instruction page.
Dividers are thin lines (typically white strokes) that run vertically or
horizontally across a significant portion of the page (>40% of page
height/width).

Heuristic
---------
- Look for Drawing elements that are thin lines (width or height near 0)
- Must span at least 40% of the page dimension
- Typically have white stroke color (for separating instruction sections)
- Can be vertical (separating left/right columns) or horizontal

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
    RuleScore,
)
from build_a_long.pdf_extract.classifier.rules import (
    IsHorizontalDividerRule,
    IsInstanceFilter,
    IsVerticalDividerRule,
    MaxScoreRule,
    Rule,
    StrokeColorScore,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Divider,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

log = logging.getLogger(__name__)


class DividerClassifier(RuleBasedClassifier):
    """Classifier for divider lines on instruction pages."""

    output: ClassVar[str] = "divider"
    requires: ClassVar[frozenset[str]] = frozenset()  # No dependencies

    @property
    def rules(self) -> list[Rule]:
        config = self.config.divider
        return [
            IsInstanceFilter(Drawing),
            MaxScoreRule(
                rules=[
                    IsVerticalDividerRule(
                        max_thickness=config.max_thickness,
                        min_length_ratio=config.min_length_ratio,
                        edge_margin=config.edge_margin,
                        weight=0.7,
                        name="VerticalDivider",
                        required=False,
                    ),
                    IsHorizontalDividerRule(
                        max_thickness=config.max_thickness,
                        min_length_ratio=config.min_length_ratio,
                        edge_margin=config.edge_margin,
                        weight=0.7,
                        name="HorizontalDivider",
                        required=False,
                    ),
                ],
                weight=0.7,
                required=True,
                name="DividerShape",
            ),
            StrokeColorScore(
                weight=0.3,
                required=False,  # Color is bonus, shape is mandatory
                name="StrokeColor",
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> Divider:
        """Construct a Divider element from a single candidate."""
        detail_score = candidate.score_details
        assert isinstance(detail_score, RuleScore)

        # Infer orientation from score details
        # If "VerticalDivider" score is non-zero, it's vertical
        # Note: RuleScore stores rule scores, but since IsVertical/Horizontal are
        # inside MaxScoreRule, their individual scores might not be directly exposed
        # as top-level keys if not careful.
        # However, `RuleBasedClassifier` calculates weighted sum.
        # Wait, `RuleBasedClassifier` iterates over `self.rules`.
        # MaxScoreRule returns the max score.
        # It doesn't expose WHICH rule won.

        # So we need to re-evaluate or check dimensions to determine orientation
        bbox = candidate.bbox
        if bbox.height > bbox.width:
            orientation = Divider.Orientation.VERTICAL
        else:
            orientation = Divider.Orientation.HORIZONTAL

        return Divider(
            bbox=bbox,
            orientation=orientation,
        )
