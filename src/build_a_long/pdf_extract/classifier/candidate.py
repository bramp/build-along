"""Candidate class for classification results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Blocks
from build_a_long.pdf_extract.utils import auto_id_field

if TYPE_CHECKING:
    from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement


class Candidate[T: "LegoPageElement"](BaseModel):
    """A candidate block with its score.

    The generic type parameter T indicates what LegoPageElement type this
    candidate will produce when built. This enables type-safe constraint
    generation by SchemaConstraintGenerator.

    Example:
        # A candidate that produces a Part element
        part_candidate: Candidate[Part]

        # A list of candidates that produce Part elements
        part_candidates: list[Candidate[Part]]

    Represents a single block that was considered for a particular label,
    including its score and score details.

    Mutable state (constructed element, failure reasons) is tracked separately
    in ClassificationResult. Use the result's accessor methods (get_constructed,
    get_failure_reason, is_valid) to check build state.

    Note: source_blocks is mutable because some classifiers add additional
    blocks during build() (e.g., consuming shadow blocks for bag numbers).

    This enables:
    - Re-evaluation with hints (exclude specific candidates)
    - Debugging (see all candidates and why they won/lost)
    - UI support (show users alternatives)
    - Type-safe constraint generation
    """

    id: int = Field(default_factory=auto_id_field)
    """Unique identifier for this candidate.

    This ID is assigned at construction time and is preserved when Pydantic
    deep-copies the object (e.g., when storing in Score objects with generic
    type annotations). Use this for identity comparisons instead of id().
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

    source_blocks: list[Blocks] = Field(default_factory=list)
    """The raw elements that were scored (empty for synthetic elements like Step).
    
    Multiple source blocks indicate the candidate was derived from multiple inputs.
    For example, a PieceLength is derived from both a Text block (the number) and
    a Drawing block (the circle diagram).
    
    This list is mutable because some classifiers add additional blocks during
    build() (e.g., consuming shadow blocks for bag numbers).
    
    Callers should ensure unique blocks.
    """

    @model_validator(mode="after")
    def validate_source_blocks_for_label(self) -> Candidate:
        """Validate that source_blocks is empty for composite-labeled candidates
        and non-empty for non-composite-labeled candidates.
        """
        composite_labels = {
            "page",
            "step",
            "part",
            "substep",
            "progress_bar",  # Made of progress_bar_bar + progress_bar_indicator
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
