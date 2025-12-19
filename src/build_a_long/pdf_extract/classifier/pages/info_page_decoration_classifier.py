"""
Info page decoration classifier.

Purpose
-------
Identify INFO pages (cover, credits, table of contents, etc.) and claim all
their content as a single decorative element. This prevents these blocks from
being left unassigned.

Strategy
--------
1. During scoring: Create a candidate that claims ALL blocks on INFO pages
2. PageClassifier checks for a high-scoring decoration candidate FIRST
3. If found, build just that and skip step/catalog building entirely

This approach is efficient because we don't waste time trying to find
steps/parts on pages that are clearly INFO pages.
"""

import logging
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.lego_page_elements import Decoration, Page
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing, Image, Text

log = logging.getLogger(__name__)


class _DecorationScore(Score):
    """Score for info page decoration candidates."""

    element_count: int
    """Number of elements claimed by this decoration."""

    info_confidence: float
    """Confidence that this is an INFO page (from PageHint)."""

    def score(self) -> Weight:
        """Score based on INFO confidence and element count.

        High confidence INFO pages with many elements score highest.
        """
        # Use INFO confidence directly - we want high confidence pages
        return self.info_confidence


class InfoPageDecorationClassifier(LabelClassifier):
    """Classifier for decorative content on INFO pages.

    This classifier creates a candidate that claims ALL blocks on pages
    identified as INFO pages (via PageHint). PageClassifier will check
    for this candidate first and, if it's high-scoring, skip the normal
    step/catalog building entirely.
    """

    output: ClassVar[str] = "decoration"
    requires: ClassVar[frozenset[str]] = frozenset()  # No dependencies

    def _score(self, result: ClassificationResult) -> None:
        """Create a decoration candidate for INFO pages."""
        page_data = result.page_data
        page_number = page_data.page_number

        # Check page hint for INFO confidence
        page_hint = self.config.page_hints.get_hint(page_number)
        if page_hint is None:
            log.debug(
                "[decoration] Page %d: No page hint available, skipping",
                page_number,
            )
            return

        info_confidence = page_hint.confidences.get(Page.PageType.INFO, 0.0)

        # Only create candidate if there's meaningful INFO confidence
        if info_confidence < 0.5:
            log.debug(
                "[decoration] Page %d: Low INFO confidence (%.2f), skipping",
                page_number,
                info_confidence,
            )
            return

        # Collect all blocks (drawings, images, text)
        source_blocks: list[Blocks] = [
            block
            for block in page_data.blocks
            if isinstance(block, Drawing | Image | Text)
        ]

        if not source_blocks:
            log.debug(
                "[decoration] Page %d: No blocks to claim",
                page_number,
            )
            return

        log.debug(
            "[decoration] Page %d: INFO confidence=%.2f, %d blocks",
            page_number,
            info_confidence,
            len(source_blocks),
        )

        # Create candidate claiming all blocks
        result.add_candidate(
            Candidate(
                bbox=page_data.bbox,
                label="decoration",
                score=info_confidence,
                score_details=_DecorationScore(
                    element_count=len(source_blocks),
                    info_confidence=info_confidence,
                ),
                source_blocks=source_blocks,
            )
        )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Decoration:
        """Construct a Decoration element from a candidate."""
        return Decoration(bbox=candidate.bbox)
