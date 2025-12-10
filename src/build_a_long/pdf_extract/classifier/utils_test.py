from build_a_long.pdf_extract.classifier.utils import (
    colors_match,
    extract_unique_points,
    score_white_fill,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Drawing


class TestScoreWhiteFill:
    def test_white_fill(self):
        """Test score for white fill."""
        bbox = BBox(x0=0, y0=0, x1=10, y1=10)
        drawing = Drawing(bbox=bbox, id=0, fill_color=(1.0, 1.0, 1.0))
        assert score_white_fill(drawing) == 1.0

    def test_light_gray_fill(self):
        """Test score for light gray fill."""
        bbox = BBox(x0=0, y0=0, x1=10, y1=10)
        drawing = Drawing(bbox=bbox, id=0, fill_color=(0.85, 0.85, 0.85))
        score = score_white_fill(drawing)
        assert 0.6 < score < 1.0

    def test_dark_fill(self):
        """Test score for dark fill."""
        bbox = BBox(x0=0, y0=0, x1=10, y1=10)
        drawing = Drawing(bbox=bbox, id=0, fill_color=(0.5, 0.5, 0.5))
        assert score_white_fill(drawing) == 0.0

    def test_no_fill(self):
        """Test score for no fill."""
        bbox = BBox(x0=0, y0=0, x1=10, y1=10)
        drawing = Drawing(bbox=bbox, id=0, fill_color=None)
        assert score_white_fill(drawing) == 0.0


class TestColorsMatch:
    def test_exact_match(self):
        """Test exact color match."""
        assert colors_match((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))

    def test_within_tolerance(self):
        """Test colors within tolerance."""
        assert colors_match((1.0, 1.0, 1.0), (0.95, 1.0, 1.0))

    def test_outside_tolerance(self):
        """Test colors outside tolerance."""
        assert not colors_match((1.0, 1.0, 1.0), (0.5, 1.0, 1.0))

    def test_different_lengths(self):
        """Test colors with different channel counts."""
        assert not colors_match((1.0, 1.0, 1.0), (1.0, 1.0))


class TestExtractUniquePoints:
    def test_extracts_triangle_points(self):
        """Test extracting points from triangle line items."""
        items = [
            ("l", (0.0, 0.0), (10.0, 5.0)),
            ("l", (10.0, 5.0), (0.0, 10.0)),
            ("l", (0.0, 10.0), (0.0, 0.0)),
        ]
        points = extract_unique_points(items)
        assert len(points) == 3
        assert (0.0, 0.0) in points
        assert (10.0, 5.0) in points
        assert (0.0, 10.0) in points

    def test_deduplicates_close_points(self):
        """Test that very close points are deduplicated when rounded."""
        items = [
            ("l", (0.0, 0.0), (10.0, 5.0)),
            ("l", (10.02, 5.04), (0.0, 10.0)),  # Very close to (10.0, 5.0)
        ]
        points = extract_unique_points(items)
        assert len(points) == 3  # (0,0), (10,5), (0,10)
        assert (10.0, 5.0) in points
