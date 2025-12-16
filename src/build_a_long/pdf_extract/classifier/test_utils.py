"""Test utilities for classifier tests.

This module provides helper classes and functions for testing classifiers.
"""

from typing import Any, Self

from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import (
    Blocks,
    Drawing,
    Image,
    Text,
)


class TestScore(Score):
    """Simple Score implementation for testing purposes.

    This provides a minimal Score implementation that can be used in unit tests
    where the actual score calculation logic is not relevant.
    """

    value: float = 1.0
    """The score value (default 1.0, must be in range 0.0-1.0)."""

    def score(self) -> Weight:
        """Return the configured score value (0.0-1.0)."""
        return self.value


class PageBuilder:
    """Builder for creating PageData objects for testing."""

    def __init__(
        self, page_number: int = 1, width: float = 1000.0, height: float = 1000.0
    ) -> None:
        self.page_number = page_number
        self.width = width
        self.height = height
        self.blocks: list[Blocks] = []
        self._next_id = 0

    def add_text(
        self,
        text: str,
        x: float,
        y: float,
        w: float = 10.0,
        h: float = 10.0,
        id: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Add a Text block."""
        block_id = id if id is not None else self._next_id
        self._next_id = max(self._next_id, block_id + 1)

        self.blocks.append(
            Text(
                id=block_id,
                bbox=BBox(x, y, x + w, y + h),
                text=text,
                **kwargs,
            )
        )
        return self

    def add_drawing(
        self,
        x: float,
        y: float,
        w: float = 50.0,
        h: float = 50.0,
        id: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Add a Drawing block."""
        block_id = id if id is not None else self._next_id
        self._next_id = max(self._next_id, block_id + 1)

        self.blocks.append(
            Drawing(
                id=block_id,
                bbox=BBox(x, y, x + w, y + h),
                **kwargs,
            )
        )
        return self

    def add_image(
        self,
        x: float,
        y: float,
        w: float = 50.0,
        h: float = 50.0,
        id: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Add an Image block."""
        block_id = id if id is not None else self._next_id
        self._next_id = max(self._next_id, block_id + 1)

        self.blocks.append(
            Image(
                id=block_id,
                bbox=BBox(x, y, x + w, y + h),
                **kwargs,
            )
        )
        return self

    def build(self) -> PageData:
        """Build the PageData object."""
        return PageData(
            page_number=self.page_number,
            blocks=self.blocks,
            bbox=BBox(0, 0, self.width, self.height),
        )
