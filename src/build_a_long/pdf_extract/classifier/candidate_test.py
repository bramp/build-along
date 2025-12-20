"""Tests for candidate module."""

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Text


class _TestScore(Score):
    """Simple test score implementation."""

    def score(self) -> float:
        return 1.0


class TestCandidateSourceBlocks:
    """Tests for source_blocks behavior."""

    def test_accepts_unique_blocks(self) -> None:
        """List with unique blocks should be accepted."""
        block1 = Text(
            bbox=BBox(x0=0, y0=0, x1=10, y1=10),
            id=1,
            text="a",
            font_size=12.0,
        )
        block2 = Text(
            bbox=BBox(x0=20, y0=0, x1=30, y1=10),
            id=2,
            text="b",
            font_size=12.0,
        )

        candidate = Candidate(
            bbox=BBox(x0=0, y0=0, x1=30, y1=10),
            label="test",
            score=1.0,
            score_details=_TestScore(),
            source_blocks=[block1, block2],
        )

        assert len(candidate.source_blocks) == 2
        assert candidate.source_blocks[0] is block1
        assert candidate.source_blocks[1] is block2

    def test_empty_source_blocks_for_composite_labels(self) -> None:
        """Composite labels should have empty source_blocks."""
        candidate = Candidate(
            bbox=BBox(x0=0, y0=0, x1=10, y1=10),
            label="step",  # composite label
            score=1.0,
            score_details=_TestScore(),
            source_blocks=[],
        )

        assert len(candidate.source_blocks) == 0
