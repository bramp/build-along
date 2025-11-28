"""Candidate class for classification results."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElements
from build_a_long.pdf_extract.extractor.page_blocks import Blocks


# TODO Change this to be frozen
class Candidate(BaseModel):
    """A candidate block with its score and constructed LegoElement.

    Represents a single block that was considered for a particular label,
    including its score, the constructed LegoPageElement (if successful),
    and information about why it succeeded or failed.

    This enables:
    - Re-evaluation with hints (exclude specific candidates)
    - Debugging (see all candidates and why they won/lost)
    - UI support (show users alternatives)
    """

    bbox: BBox
    """The bounding box for this candidate (from source_block or constructed)"""

    label: str
    """The label this candidate would have (e.g., 'page_number')"""

    # TODO Maybe score is redudant with score_details?
    score: float
    """Combined score (0.0-1.0)"""

    score_details: Score
    """The detailed score object inheriting from Score (e.g., _PageNumberScore)"""

    constructed: LegoPageElements | None = None
    """The constructed LegoElement if parsing succeeded, None if failed"""

    source_blocks: list[Blocks] = []
    """The raw elements that were scored (empty for synthetic elements like Step).
    
    Multiple source blocks indicate the candidate was derived from multiple inputs.
    For example, a PieceLength is derived from both a Text block (the number) and
    a Drawing block (the circle diagram).
    """

    failure_reason: str | None = None
    """Why construction failed, if it did"""

    @property
    def is_valid(self) -> bool:
        """Check if this candidate is valid (constructed and no failure).

        A valid candidate has been successfully constructed and has no failure reason.
        Use this to filter candidates when working with dependencies.

        Returns:
            True if candidate.constructed is not None and failure_reason is None
        """
        return self.constructed is not None and self.failure_reason is None

    @model_validator(mode="after")
    def validate_source_blocks_for_label(self) -> Candidate:
        """Validate that source_blocks is empty for composite-labeled candidates
        and non-empty for non-composite-labeled candidates.
        """
        composite_labels = {
            "page",
            "step",
            "part",
            "new_bag",
        }

        # Non-composite labels are those that correspond to LegoPageElements
        # that are derived directly from Blocks.
        # Examples from lego_page_elements.py: PageNumber, StepNumber, PartCount,
        # PartNumber, PieceLength, PartImage, ProgressBar, BagNumber, Diagram.
        # These should always have source_blocks.

        if self.label in composite_labels:
            assert not self.source_blocks, (
                f"Candidate with label '{self.label}' should have empty source_blocks, "
                f"but got {len(self.source_blocks)}."
            )
        else:
            # If a candidate is not composite, it should have source_blocks.
            # This covers cases like 'part_count', 'page_number', etc.
            assert self.source_blocks, (
                f"Candidate with label '{self.label}' should have non-empty "
                f"source_blocks, but got 0."
            )
        return self
