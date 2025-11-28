"""RemovalReason class for tracking removed blocks."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.extractor.page_blocks import Blocks


class RemovalReason(BaseModel):
    """Tracks why a block was removed during classification."""

    model_config = {"frozen": True}

    reason_type: str
    """Type of removal: 'duplicate_bbox', 'child_bbox', or 'similar_bbox'"""

    # TODO Should this be updated to the Candidate that caused the removal?
    target_block: Blocks
    """The block that caused this removal"""
