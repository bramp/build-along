"""
Progress bar classifier.

Purpose
-------
Pair progress_bar_bar candidates with progress_bar_indicator candidates to
create complete ProgressBar elements. This follows the composite pattern
where the parent element (ProgressBar) assembles its children (bar + indicator).

Heuristic
---------
- For each progress_bar_bar candidate, find a matching indicator that:
  - Extends beyond the bar's vertical bounds (sticks out above/below)
  - Is horizontally within the bar's extent
  - Is vertically aligned with the bar's center
- Calculate progress based on the indicator's horizontal position within the bar

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    CandidateFailedError,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.config import ProgressBarConfig
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    ProgressBar,
    ProgressBarBar,
    ProgressBarIndicator,
)

log = logging.getLogger(__name__)


class _ProgressBarPairScore(Score):
    """Internal score for pairing bar and indicator candidates."""

    bar_candidate: Candidate
    """The progress_bar_bar candidate."""

    indicator_candidate: Candidate | None = None
    """The progress_bar_indicator candidate (optional)."""

    extension_amount: float = 0.0
    """How much the indicator extends beyond the bar (normalized)."""

    def score(self) -> Weight:
        """Calculate final weighted score.

        Base score is the bar's score. If an indicator is present,
        boost the score based on how much it extends beyond the bar.
        """
        base_score = self.bar_candidate.score

        # Boost score if we have a good indicator match
        if self.indicator_candidate:
            # Add extension bonus (up to 0.2)
            base_score = min(1.0, base_score + self.extension_amount * 0.2)

        return base_score


class ProgressBarClassifier(LabelClassifier):
    """Classifier for complete progress bar elements.

    This classifier assembles ProgressBar elements by pairing:
    - progress_bar_bar candidates (the horizontal bar track)
    - progress_bar_indicator candidates (the circular position marker)

    The indicator must extend beyond the bar's vertical bounds to be
    considered a valid match.
    """

    output = "progress_bar"
    requires = frozenset({"progress_bar_bar", "progress_bar_indicator"})

    def _score(self, result: ClassificationResult) -> None:
        """Create ProgressBar candidates by pairing bar and indicator candidates."""
        config: ProgressBarConfig = self.config.progress_bar

        # Get bar candidates (scoring phase - candidates aren't constructed yet)
        bar_candidates = result.get_scored_candidates("progress_bar_bar")

        # Get indicator candidates (scoring phase)
        indicator_candidates = result.get_scored_candidates("progress_bar_indicator")

        log.debug(
            "[progress_bar] page=%s bar_candidates=%d indicator_candidates=%d",
            result.page_data.page_number,
            len(bar_candidates),
            len(indicator_candidates),
        )

        if not bar_candidates:
            return

        # Track which indicators have been used
        used_indicators: set[int] = set()

        for bar_cand in bar_candidates:
            # Find the best matching indicator for this bar
            best_indicator, extension_amount = self._find_best_indicator(
                bar_cand, indicator_candidates, used_indicators, config
            )

            # Mark indicator as used if found
            if best_indicator:
                used_indicators.add(id(best_indicator))

            # Create the pairing score
            pair_score = _ProgressBarPairScore(
                bar_candidate=bar_cand,
                indicator_candidate=best_indicator,
                extension_amount=extension_amount,
            )

            # Create a candidate for this progress bar
            # The bbox will be computed in build() as union of bar + indicator
            # As a composite element, it has no direct source_blocks - blocks are
            # claimed through the child bar and indicator candidates when built.
            candidate = Candidate(
                label="progress_bar",
                score=pair_score.score(),
                score_details=pair_score,
                bbox=bar_cand.bbox,  # Temporary, will be updated in build
                source_blocks=[],  # Composite element - blocks claimed via children
            )
            result.add_candidate(candidate)

    def _find_best_indicator(
        self,
        bar_cand: Candidate,
        indicator_candidates: Sequence[Candidate],
        used_indicators: set[int],
        config: ProgressBarConfig,
    ) -> tuple[Candidate | None, float]:
        """Find the best matching indicator for a bar candidate.

        Args:
            bar_cand: The bar candidate to find an indicator for
            indicator_candidates: All available indicator candidates
            used_indicators: Set of indicator IDs that have already been used
            config: ProgressBarConfig instance

        Returns:
            Tuple of (best_indicator, extension_amount) where extension_amount
            is normalized by bar height. Returns (None, 0.0) if no match found.
        """
        bar_bbox = bar_cand.bbox
        bar_height = bar_bbox.height
        bar_center_y = (bar_bbox.y0 + bar_bbox.y1) / 2

        # Get the original bar width from the primary source block
        primary_block = bar_cand.source_blocks[0]
        bar_start_x = primary_block.bbox.x0
        bar_full_width = primary_block.bbox.width

        best_candidate: Candidate | None = None
        best_score: float = -1.0
        best_extension: float = 0.0

        for ind_cand in indicator_candidates:
            # Skip already used indicators
            if id(ind_cand) in used_indicators:
                continue

            # Skip if already built (consumed elsewhere)
            if ind_cand.constructed is not None:
                continue

            ind_bbox = ind_cand.bbox

            # Indicator must be at least as tall as the bar
            if ind_bbox.height < bar_height:
                continue

            # Check if indicator extends beyond the bar vertically
            # This is the key criterion - a true indicator "sticks out" from the bar
            extends_above = ind_bbox.y0 < bar_bbox.y0
            extends_below = ind_bbox.y1 > bar_bbox.y1
            if not (extends_above or extends_below):
                continue

            # Check if the indicator's center Y is aligned with the bar's center Y
            ind_center_y = (ind_bbox.y0 + ind_bbox.y1) / 2
            if abs(ind_center_y - bar_center_y) > bar_height:
                continue

            # Must be horizontally within or near the progress bar
            indicator_x = (ind_bbox.x0 + ind_bbox.x1) / 2
            bar_end_x = bar_start_x + bar_full_width
            if (
                indicator_x < bar_start_x - config.indicator_search_margin
                or indicator_x > bar_end_x + config.indicator_search_margin
            ):
                continue

            # Calculate how much the indicator extends beyond the bar
            extension_above = max(0.0, bar_bbox.y0 - ind_bbox.y0)
            extension_below = max(0.0, ind_bbox.y1 - bar_bbox.y1)
            total_extension = extension_above + extension_below
            normalized_extension = total_extension / bar_height if bar_height > 0 else 0

            # Score: combine indicator's base score with extension bonus
            effective_score = ind_cand.score + normalized_extension * 0.5

            if effective_score > best_score:
                best_candidate = ind_cand
                best_score = effective_score
                best_extension = normalized_extension

        return best_candidate, best_extension

    def build(self, candidate: Candidate, result: ClassificationResult) -> ProgressBar:
        """Construct a ProgressBar element from a paired candidate.

        The build order is important:
        1. Build indicator first (claims its shadow blocks)
        2. Build bar (claims remaining shadow blocks)

        This prevents conflicts between indicator and bar over shared blocks
        near the indicator position.
        """
        pair_score = candidate.score_details
        assert isinstance(pair_score, _ProgressBarPairScore)

        bar_cand = pair_score.bar_candidate
        ind_cand = pair_score.indicator_candidate

        # Build the indicator FIRST if present (it claims its shadow blocks)
        indicator: ProgressBarIndicator | None = None

        if ind_cand:
            try:
                built_indicator = result.build(ind_cand)
                assert isinstance(built_indicator, ProgressBarIndicator)
                indicator = built_indicator
            except CandidateFailedError as e:
                log.debug(
                    "[progress_bar] Failed to build indicator at %s: %s",
                    ind_cand.bbox,
                    e,
                )

        # Build the bar element (it claims remaining shadow blocks)
        bar_element = result.build(bar_cand)
        assert isinstance(bar_element, ProgressBarBar)

        # Calculate progress based on indicator position relative to full bar width
        progress: float | None = None
        if indicator:
            bar_start_x = bar_element.bbox.x0
            bar_full_width = bar_element.bbox.width
            indicator_x = (indicator.bbox.x0 + indicator.bbox.x1) / 2
            progress = (indicator_x - bar_start_x) / bar_full_width
            progress = max(0.0, min(1.0, progress))

            log.debug(
                "[progress_bar] indicator_x=%.1f bar_start=%.1f width=%.1f "
                "progress=%.1f%%",
                indicator_x,
                bar_start_x,
                bar_full_width,
                progress * 100,
            )

        # Compute final bbox as union of bar + indicator
        bbox = bar_element.bbox
        if indicator:
            bbox = bbox.union(indicator.bbox)

        # Get full_width from the bar element
        full_width = bar_element.bbox.width

        return ProgressBar(
            bbox=bbox,
            progress=progress,
            full_width=full_width,
            bar=bar_element,
            indicator=indicator,
        )
