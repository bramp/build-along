"""
Background classifier.

Purpose
-------
Aggregate all background-related elements into a single Background element.
This classifier depends on:
- full_page_background: Large drawings covering most of the page
- page_edge: Artifacts at page edges (borders, bleed lines)

The Background element combines all source blocks from these classifiers
into a single element representing the page backdrop.

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
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.rule_based_classifier import RuleScore
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import Background
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing

log = logging.getLogger(__name__)


class BackgroundClassifier(LabelClassifier):
    """Classifier that aggregates background elements.

    This classifier depends on full_page_background and page_edge classifiers,
    collecting all their source blocks into a single Background element.
    """

    output: ClassVar[str] = "background"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"full_page_background", "page_edge"}
    )

    def _score(self, result: ClassificationResult) -> None:
        """Aggregate all background-related candidates into a single candidate."""
        # Collect source blocks from both dependent classifiers
        background_blocks: list[Blocks] = []
        full_page_count = 0
        edge_count = 0

        # Get full-page background candidates
        for candidate in result.get_scored_candidates(
            "full_page_background", valid_only=False, exclude_failed=True
        ):
            background_blocks.extend(candidate.source_blocks)
            full_page_count += 1

        # Get page-edge candidates
        for candidate in result.get_scored_candidates(
            "page_edge", valid_only=False, exclude_failed=True
        ):
            background_blocks.extend(candidate.source_blocks)
            edge_count += 1

        # Create a single candidate with all background blocks
        if background_blocks:
            combined_bbox = BBox.union_all([b.bbox for b in background_blocks])
            result.add_candidate(
                Candidate(
                    bbox=combined_bbox,
                    label=self.output,
                    score=1.0,
                    score_details=RuleScore(
                        components={
                            "full_page_count": float(full_page_count),
                            "edge_count": float(edge_count),
                        },
                        total_score=1.0,
                    ),
                    source_blocks=background_blocks,
                )
            )
            log.debug(
                "[background] Combined %d full-page + %d edge into single background",
                full_page_count,
                edge_count,
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Background:
        """Construct a Background element from the aggregated candidate."""
        # Try to get a Drawing block for fill color extraction
        # Prefer the largest drawing (likely the full-page background)
        drawing_blocks = [b for b in candidate.source_blocks if isinstance(b, Drawing)]
        drawing_block = None
        if drawing_blocks:
            drawing_block = max(drawing_blocks, key=lambda b: b.bbox.area)

        # Extract fill color as RGB tuple (only available from Drawing blocks)
        fill_color: tuple[float, float, float] | None = None
        if (
            drawing_block is not None
            and drawing_block.fill_color is not None
            and len(drawing_block.fill_color) >= 3
        ):
            fill_color = (
                drawing_block.fill_color[0],
                drawing_block.fill_color[1],
                drawing_block.fill_color[2],
            )

        return Background(
            bbox=candidate.bbox,
            fill_color=fill_color,
        )
