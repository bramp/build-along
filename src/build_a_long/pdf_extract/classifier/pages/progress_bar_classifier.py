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
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    CandidateFailedError,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.config import ProgressBarConfig
from build_a_long.pdf_extract.classifier.constraint_model import ConstraintModel
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

    # Score ranges:
    # - 0.9-1.0: Composite with high-confidence children (bar + indicator)
    # - 0.8-0.9: Composite with bar only (no indicator)
    # - 0.6-0.8: Individual intrinsic elements cap here
    COMPOSITE_BONUS: ClassVar[float] = 0.15
    INDICATOR_BONUS: ClassVar[float] = 0.10

    def score(self) -> Weight:
        """Calculate final weighted score.

        Composite elements (like progress_bar) get a bonus on top of their
        children's quality scores. This encourages the solver to prefer
        complete structures while still considering child quality.

        Score = bar_score + COMPOSITE_BONUS + (INDICATOR_BONUS if indicator)
        Capped at 1.0.

        With a good bar (score ~0.7) and indicator:
        - 0.7 + 0.15 + 0.10 = 0.95 (high confidence composite)
        With a good bar only:
        - 0.7 + 0.15 = 0.85 (composite base)
        """
        # Start with the bar's intrinsic quality
        base_score = self.bar_candidate.score

        # Add composite bonus for being a complete structure
        base_score += self.COMPOSITE_BONUS

        # Boost score if we have a good indicator match
        if self.indicator_candidate:
            base_score += self.INDICATOR_BONUS

        return min(1.0, base_score)


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

    def declare_constraints(
        self, model: ConstraintModel, result: ClassificationResult
    ) -> None:
        """Declare constraints for progress_bar candidates.

        Constraints:
        - At most one progress bar per page (singleton element)

        Note: Child uniqueness constraints (each bar/indicator can only be used
        by one progress_bar) are handled automatically by
        SchemaConstraintGenerator.add_child_uniqueness_constraints().
        """
        candidates = list(result.get_scored_candidates("progress_bar"))
        candidates_in_model = [c for c in candidates if model.has_candidate(c)]

        if len(candidates_in_model) <= 1:
            return

        # At most one progress bar per page
        model.at_most_one_of(candidates_in_model)
        log.debug(
            "[progress_bar] Added at_most_one constraint for %d candidates",
            len(candidates_in_model),
        )

    def _score(self, result: ClassificationResult) -> None:
        """Create ProgressBar candidates for all valid bar+indicator pairings."""
        config: ProgressBarConfig = self.config.progress_bar

        bar_candidates = result.get_scored_candidates("progress_bar_bar")
        indicator_candidates = list(
            result.get_scored_candidates("progress_bar_indicator")
        )

        log.debug(
            "[progress_bar] page=%s bar_candidates=%d indicator_candidates=%d",
            result.page_data.page_number,
            len(bar_candidates),
            len(indicator_candidates),
        )

        if not bar_candidates:
            return

        for bar_cand in bar_candidates:
            # Find ALL valid indicators for this bar
            valid_indicators = self._get_valid_indicators(
                bar_cand, indicator_candidates, config
            )

            # Create a candidate for each valid pairing
            for ind_cand, extension_amount in valid_indicators:
                self._create_candidate(result, bar_cand, ind_cand, extension_amount)

            # Also create a candidate with no indicator (bar only)
            if not valid_indicators:
                self._create_candidate(result, bar_cand, None, 0.0)

    def _create_candidate(
        self,
        result: ClassificationResult,
        bar_cand: Candidate,
        ind_cand: Candidate | None,
        extension_amount: float,
    ) -> None:
        """Create a progress_bar candidate for a bar+indicator pairing."""
        pair_score = _ProgressBarPairScore(
            bar_candidate=bar_cand,
            indicator_candidate=ind_cand,
            extension_amount=extension_amount,
        )
        candidate = Candidate(
            label="progress_bar",
            score=pair_score.score(),
            score_details=pair_score,
            bbox=bar_cand.bbox,
            source_blocks=[],
        )
        result.add_candidate(candidate)

    def _get_valid_indicators(
        self,
        bar_cand: Candidate,
        indicator_candidates: Sequence[Candidate],
        config: ProgressBarConfig,
    ) -> list[tuple[Candidate, float]]:
        """Find all valid indicators for a bar candidate.

        Returns:
            List of (indicator_candidate, extension_amount) tuples.
        """
        bar_bbox = bar_cand.bbox
        bar_height = bar_bbox.height
        bar_center_y = (bar_bbox.y0 + bar_bbox.y1) / 2

        primary_block = bar_cand.source_blocks[0]
        bar_start_x = primary_block.bbox.x0
        bar_full_width = primary_block.bbox.width
        bar_end_x = bar_start_x + bar_full_width

        valid: list[tuple[Candidate, float]] = []

        for ind_cand in indicator_candidates:
            ind_bbox = ind_cand.bbox

            # Indicator must be at least as tall as the bar
            if ind_bbox.height < bar_height:
                continue

            # Must extend beyond the bar vertically
            extends_above = ind_bbox.y0 < bar_bbox.y0
            extends_below = ind_bbox.y1 > bar_bbox.y1
            if not (extends_above or extends_below):
                continue

            # Center Y must be aligned with bar's center Y
            ind_center_y = (ind_bbox.y0 + ind_bbox.y1) / 2
            if abs(ind_center_y - bar_center_y) > bar_height:
                continue

            # Must be horizontally within or near the bar
            indicator_x = (ind_bbox.x0 + ind_bbox.x1) / 2
            if (
                indicator_x < bar_start_x - config.indicator_search_margin
                or indicator_x > bar_end_x + config.indicator_search_margin
            ):
                continue

            # Calculate extension amount
            extension_above = max(0.0, bar_bbox.y0 - ind_bbox.y0)
            extension_below = max(0.0, ind_bbox.y1 - bar_bbox.y1)
            total_extension = extension_above + extension_below
            normalized = total_extension / bar_height if bar_height > 0 else 0

            valid.append((ind_cand, normalized))

        return valid

    def build(self, candidate: Candidate, result: ClassificationResult) -> ProgressBar:
        """Construct a ProgressBar element from a paired candidate.

        The build order is important:
        1. Build indicator first (consumes its shadow blocks)
        2. Build bar (consumes remaining shadow blocks)

        This prevents conflicts between indicator and bar over shared blocks
        near the indicator position.
        """
        pair_score = candidate.score_details
        assert isinstance(pair_score, _ProgressBarPairScore)

        bar_cand = pair_score.bar_candidate
        ind_cand = pair_score.indicator_candidate

        # Build the indicator FIRST if present (it consumes its shadow blocks)
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

        # Build the bar element (it consumes remaining shadow blocks)
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
