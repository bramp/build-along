"""
Progress bar bar classifier.

Purpose
-------
Identify the horizontal bar track portion of progress bars at the bottom of
instruction pages. This is the long, thin horizontal element that spans most
of the page width.

Heuristic
---------
- Look for Drawing/Image elements near the bottom of the page
- Must span a significant portion of the page width (e.g., >50%)
- Should be relatively thin vertically (height << width)
- Located near the page number or bottom margin

The ProgressBarClassifier will then pair these bar candidates with
ProgressBarIndicator candidates to form complete ProgressBar elements.

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
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
)
from build_a_long.pdf_extract.classifier.rules import (
    BottomPositionScore,
    ContinuousAspectRatioScore,
    IsInstanceFilter,
    PageNumberProximityScore,
    Rule,
    WidthCoverageScore,
)
from build_a_long.pdf_extract.classifier.rules.scale import (
    LinearScale,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import ProgressBarBar
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image

log = logging.getLogger(__name__)


class ProgressBarBarClassifier(RuleBasedClassifier):
    """Classifier for the bar portion of progress bars.

    This classifier identifies the horizontal bar track that spans the page width.
    The resulting candidates are consumed by ProgressBarClassifier which pairs
    them with indicator candidates.
    """

    output = "progress_bar_bar"
    requires = frozenset({"page_number"})

    @property
    def min_score(self) -> float:
        return self.config.progress_bar.min_score

    @property
    def rules(self) -> Sequence[Rule]:
        config: ProgressBarConfig = self.config.progress_bar
        return [
            IsInstanceFilter((Drawing, Image)),
            BottomPositionScore(
                scale=LinearScale(
                    {0.0: 1.0, config.max_bottom_margin_ratio: 0.0}
                ),  # Closer to bottom scores higher
                weight=1.0,
                name="position_score",
                required=True,  # Progress bar must span most of the page width
            ),
            PageNumberProximityScore(
                proximity_ratio=config.max_page_number_proximity_ratio,
                weight=0.2,
                name="page_number_proximity_score",
            ),
            WidthCoverageScore(
                scale=LinearScale(
                    {config.min_width_ratio: 0.0, config.max_score_width_ratio: 1.0}
                ),
                weight=1.0,
                name="width_score",
                required=True,  # Progress bar must span most of the page width
            ),
            ContinuousAspectRatioScore(
                scale=LinearScale(
                    {config.min_aspect_ratio: 0.0, config.ideal_aspect_ratio: 1.0}
                ),
                weight=1.0,
                name="aspect_ratio_score",
                required=True,  # Progress bar must be wider than tall
            ),
        ]

    # Note: We intentionally do NOT override _get_additional_source_blocks here.
    # Additional blocks are consumed during build() after the indicator has consumed
    # its shadows first, avoiding conflicts over shared blocks.

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> ProgressBarBar:
        """Construct a ProgressBarBar element from a candidate.

        Finds other unconsumed Drawing/Image blocks at approximately the same
        y-position and height, and merges them into a single ProgressBarBar.
        This handles progress bars that are split into two segments
        (left/right of indicator).

        Raises:
            CandidateFailedError: If the merged bar doesn't span enough of the
                page width.
        """
        config = self.config.progress_bar

        # Start with the candidate's bbox and source blocks
        combined_bbox = candidate.bbox
        merged_source_blocks = list(candidate.source_blocks)
        merged_block_ids = {b.id for b in merged_source_blocks}

        # Find other unconsumed blocks at the same approximate y-position and height
        for block in result.get_unconsumed_blocks((Drawing, Image)):
            # Skip blocks we've already included
            if block.id in merged_block_ids:
                continue

            # Check if at approximately the same y-position and height
            if (
                abs(block.bbox.y0 - candidate.bbox.y0) <= config.bar_merge_y_tolerance
                and abs(block.bbox.height - candidate.bbox.height)
                <= config.bar_merge_height_tolerance
            ):
                log.debug(
                    "[%s] Merging block %s at %s with candidate at %s",
                    self.output,
                    block.id,
                    block.bbox,
                    candidate.bbox,
                )
                combined_bbox = combined_bbox.union(block.bbox)
                merged_source_blocks.append(block)
                merged_block_ids.add(block.id)

        # Validate that the merged bar spans enough of the page width
        page_width = result.page_data.bbox.width
        merged_width_ratio = combined_bbox.width / page_width
        if merged_width_ratio < config.min_merged_width_ratio:
            raise CandidateFailedError(
                candidate,
                f"Merged bar width ratio {merged_width_ratio:.2f} is below "
                f"minimum {config.min_merged_width_ratio}",
            )

        # Update the candidate's source_blocks to include merged blocks
        candidate.source_blocks = merged_source_blocks

        return ProgressBarBar(bbox=combined_bbox)
