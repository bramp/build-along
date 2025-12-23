"""Tests for sorting utilities."""

from dataclasses import dataclass

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.sorting import (
    sort_by_columns,
    sort_by_rows,
    sort_left_to_right,
)


@dataclass
class Item:
    """Simple item with bbox for testing."""

    name: str
    bbox: BBox

    def __repr__(self) -> str:
        return self.name


class TestSortLeftToRight:
    """Tests for sort_left_to_right function."""

    def test_empty_list(self) -> None:
        """Empty list returns empty list."""
        assert sort_left_to_right([]) == []

    def test_single_item(self) -> None:
        """Single item returns list with that item."""
        item = Item("A", BBox(10, 20, 30, 40))
        assert sort_left_to_right([item]) == [item]

    def test_sorts_by_x0(self) -> None:
        """Items are sorted by x0 position."""
        a = Item("A", BBox(100, 0, 120, 20))
        b = Item("B", BBox(50, 0, 70, 20))
        c = Item("C", BBox(200, 0, 220, 20))

        result = sort_left_to_right([a, b, c])
        assert result == [b, a, c]

    def test_y0_as_tiebreaker(self) -> None:
        """When x0 is the same, y0 is used as tiebreaker."""
        a = Item("A", BBox(100, 50, 120, 70))
        b = Item("B", BBox(100, 10, 120, 30))
        c = Item("C", BBox(100, 30, 120, 50))

        result = sort_left_to_right([a, b, c])
        assert result == [b, c, a]


class TestSortByRows:
    """Tests for sort_by_rows function."""

    def test_empty_list(self) -> None:
        """Empty list returns empty list."""
        assert sort_by_rows([]) == []

    def test_single_item(self) -> None:
        """Single item returns list with that item."""
        item = Item("A", BBox(10, 20, 30, 40))
        assert sort_by_rows([item]) == [item]

    def test_single_row_sorted_left_to_right(self) -> None:
        """Items in the same row are sorted left-to-right."""
        # Three items on the same horizontal line (overlapping y ranges)
        a = Item("A", BBox(100, 10, 130, 40))
        b = Item("B", BBox(50, 15, 80, 45))
        c = Item("C", BBox(150, 12, 180, 42))

        result = sort_by_rows([a, b, c])
        assert result == [b, a, c]

    def test_two_rows(self) -> None:
        """Items in different rows are grouped and sorted by row."""
        # Row 1: y = 10-40
        r1_a = Item("R1A", BBox(100, 10, 130, 40))
        r1_b = Item("R1B", BBox(50, 15, 80, 45))

        # Row 2: y = 100-130
        r2_a = Item("R2A", BBox(80, 100, 110, 130))
        r2_b = Item("R2B", BBox(30, 105, 60, 135))

        result = sort_by_rows([r2_a, r1_a, r2_b, r1_b])
        # Row 1 first (lower y), then row 2, each sorted left-to-right
        assert result == [r1_b, r1_a, r2_b, r2_a]

    def test_parts_list_from_golden_file(self) -> None:
        """Test with real parts list data from 6509377_page_011_expected.json.

        This page has Step 3 with parts at:
        - Part 1: x0=76.1, y0=25.3 (left part)
        - Part 2: x0=133.51, y0=32.5 (right part, slightly lower)

        Both should be on the same row since their y ranges overlap.
        """
        part1 = Item("Part1", BBox(76.1, 25.3, 106.09, 52.25))
        part2 = Item("Part2", BBox(133.51, 32.5, 144.98, 52.25))

        # Input in wrong order
        result = sort_by_rows([part2, part1])

        # Should be sorted left-to-right
        assert result == [part1, part2]

    def test_parts_on_different_rows(self) -> None:
        """Test items that don't overlap vertically are on different rows."""
        # Row 1: y = 10-30
        top = Item("Top", BBox(100, 10, 130, 30))
        # Row 2: y = 50-70 (no overlap with row 1)
        bottom = Item("Bottom", BBox(50, 50, 80, 70))

        result = sort_by_rows([bottom, top])
        # Top row first, then bottom row
        assert result == [top, bottom]

    def test_items_with_slight_y_difference_same_row(self) -> None:
        """Items with overlapping y ranges are in the same row even if y0 differs."""
        # These overlap in y range [32.5, 43.28]
        a = Item("A", BBox(76.1, 25.3, 106.09, 43.29))
        b = Item("B", BBox(133.51, 32.5, 144.98, 43.28))

        result = sort_by_rows([b, a])
        assert result == [a, b]


class TestSortByColumns:
    """Tests for sort_by_columns function."""

    def test_empty_list(self) -> None:
        """Empty list returns empty list."""
        assert sort_by_columns([]) == []

    def test_single_item(self) -> None:
        """Single item returns list with that item."""
        item = Item("A", BBox(10, 20, 30, 40))
        assert sort_by_columns([item]) == [item]

    def test_single_column_sorted_top_to_bottom(self) -> None:
        """Items in the same column are sorted top-to-bottom."""
        # Three items in the same vertical column (overlapping x ranges)
        a = Item("A", BBox(10, 100, 40, 130))
        b = Item("B", BBox(15, 50, 45, 80))
        c = Item("C", BBox(12, 150, 42, 180))

        result = sort_by_columns([a, b, c])
        assert result == [b, a, c]

    def test_two_columns(self) -> None:
        """Items in different columns are grouped and sorted by column."""
        # Column 1: x = 10-40
        c1_a = Item("C1A", BBox(10, 100, 40, 130))
        c1_b = Item("C1B", BBox(15, 50, 45, 80))

        # Column 2: x = 100-130
        c2_a = Item("C2A", BBox(100, 80, 130, 110))
        c2_b = Item("C2B", BBox(105, 30, 135, 60))

        result = sort_by_columns([c2_a, c1_a, c2_b, c1_b])
        # Column 1 first (lower x), then column 2, each sorted top-to-bottom
        assert result == [c1_b, c1_a, c2_b, c2_a]

    def test_catalog_parts_from_golden_file(self) -> None:
        """Test with real catalog data from 6509377_page_180_expected.json.

        This catalog page has parts arranged in columns. First column parts:
        - Part at y=13.71 (top)
        - Part at y=42.11
        - Part at y=77.37
        - Part at y=110.1

        All in first column (x around 13-35).
        """
        part1 = Item("Part1", BBox(13.71, 13.71, 30.86, 36.08))
        part2 = Item("Part2", BBox(13.71, 42.11, 34.27, 71.33))
        part3 = Item("Part3", BBox(13.7, 77.37, 33.75, 104.06))
        part4 = Item("Part4", BBox(13.7, 110.1, 36.97, 140.4))

        # Input in scrambled order
        result = sort_by_columns([part3, part1, part4, part2])

        # Should be sorted top-to-bottom within the column
        assert result == [part1, part2, part3, part4]

    def test_multiple_columns_catalog_style(self) -> None:
        """Test multiple columns like a catalog page layout."""
        # Column 1 (x around 13)
        c1_top = Item("C1_Top", BBox(13.71, 13.71, 30.86, 36.08))
        c1_mid = Item("C1_Mid", BBox(13.71, 42.11, 34.27, 71.33))

        # Column 2 (x around 102)
        c2_top = Item("C2_Top", BBox(102.27, 13.71, 126.38, 44.54))
        c2_mid = Item("C2_Mid", BBox(102.74, 50.0, 125.0, 80.0))

        # Column 3 (x around 230)
        c3_top = Item("C3_Top", BBox(230.97, 13.71, 252.77, 44.53))

        # Input in scrambled order
        result = sort_by_columns([c3_top, c1_mid, c2_top, c1_top, c2_mid])

        # Should be: column1 (top, mid), column2 (top, mid), column3 (top)
        assert result == [c1_top, c1_mid, c2_top, c2_mid, c3_top]

    def test_items_not_overlapping_horizontally_different_columns(self) -> None:
        """Items that don't overlap horizontally are in different columns."""
        # Column 1: x = 10-30
        left = Item("Left", BBox(10, 100, 30, 130))
        # Column 2: x = 50-70 (no overlap with column 1)
        right = Item("Right", BBox(50, 50, 70, 80))

        result = sort_by_columns([right, left])
        # Left column first, then right column
        assert result == [left, right]


class TestSortingStability:
    """Tests to ensure sorting is stable and deterministic."""

    def test_identical_positions_stable(self) -> None:
        """Items at identical positions maintain relative order."""
        a = Item("A", BBox(10, 10, 20, 20))
        b = Item("B", BBox(10, 10, 20, 20))

        # sort_left_to_right should be stable
        result = sort_left_to_right([a, b])
        assert result == [a, b]

        result = sort_left_to_right([b, a])
        assert result == [b, a]

    def test_deterministic_output(self) -> None:
        """Same input always produces same output."""
        items = [
            Item("A", BBox(100, 10, 130, 40)),
            Item("B", BBox(50, 15, 80, 45)),
            Item("C", BBox(150, 12, 180, 42)),
        ]

        # Multiple calls should give identical results
        result1 = sort_by_rows(items)
        result2 = sort_by_rows(items)
        result3 = sort_by_rows(items)

        assert result1 == result2 == result3
