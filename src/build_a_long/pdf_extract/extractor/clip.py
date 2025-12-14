"""Clip path tracking for PDF drawing extraction.

This module provides efficient O(n) clip path computation for PyMuPDF drawings.

In PyMuPDF's drawing hierarchy (with extended=True):
- A clip at level L applies to all subsequent drawings at level > L
- A clip's scope ends when we see any drawing at level <= L

The ClipStackTracker maintains the clip stack as we iterate forward through
drawings, avoiding the O(n²) cost of walking backwards for each drawing.
"""

import logging
from collections.abc import Iterator

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.pymupdf_types import DrawingDict

logger = logging.getLogger(__name__)


class ClipStackTracker:
    """Tracks active clip paths while iterating through drawings.

    This provides O(n) clip computation instead of O(n²) by maintaining
    the clip stack as we iterate forward through drawings.

    Usage:
        tracker = ClipStackTracker()
        for drawing in drawings:
            tracker.update(drawing)
            visible_bbox = tracker.apply_clips(drawing_bbox)
    """

    def __init__(self) -> None:
        """Initialize the clip stack tracker."""
        # Stack of (level, clip_bbox) tuples, sorted by level ascending
        self._clip_stack: list[tuple[int, BBox]] = []

    def update(self, drawing: DrawingDict) -> None:
        """Update clip stack based on the current drawing.

        Call this for EVERY drawing (including clips) before computing
        visible bbox for non-clip drawings.
        """
        level = drawing.get("level", 0)
        drawing_type = drawing.get("type")

        # Pop any clips at level >= current drawing's level
        # (their scope has ended)
        while self._clip_stack and self._clip_stack[-1][0] >= level:
            popped = self._clip_stack.pop()
            logger.debug("  Popped clip at level %d", popped[0])

        # If this is a clip, add it to the stack
        if drawing_type == "clip":
            scissor = drawing.get("scissor")
            if scissor:
                # Skip inverted/invalid clip rectangles
                if scissor.x0 > scissor.x1 or scissor.y0 > scissor.y1:
                    logger.debug(
                        "  Skipping inverted clip at level %d: %s",
                        level,
                        scissor,
                    )
                    return
                clip_bbox = BBox.from_tuple(
                    (scissor.x0, scissor.y0, scissor.x1, scissor.y1)
                )
                self._clip_stack.append((level, clip_bbox))
                logger.debug("  Added clip at level %d: %s", level, clip_bbox)

    def apply_clips(self, bbox: BBox) -> BBox:
        """Apply all active clips to get the visible bbox."""
        visible = bbox
        for _, clip_bbox in self._clip_stack:
            visible = visible.intersect(clip_bbox)
        return visible


def iterate_drawings_with_clips(
    drawings: list[DrawingDict],
) -> Iterator[tuple[int, DrawingDict, BBox | None]]:
    """Iterate through drawings, yielding visible bbox for each non-clip.

    This is the efficient O(n) approach for processing drawings with clips.

    Args:
        drawings: List of drawings from page.get_drawings(extended=True)

    Yields:
        Tuples of (index, drawing, visible_bbox) for non-clip drawings.
        visible_bbox is None if the drawing has no rect.
    """
    tracker = ClipStackTracker()

    for idx, drawing in enumerate(drawings):
        tracker.update(drawing)

        # Skip clip paths - they're not visible drawings
        if drawing.get("type") == "clip":
            continue

        drect = drawing.get("rect")
        if not drect:
            yield idx, drawing, None
            continue

        bbox = BBox.from_rect(drect)
        visible_bbox = tracker.apply_clips(bbox)

        yield idx, drawing, visible_bbox
