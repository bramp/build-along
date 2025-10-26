import pytest

from build_a_long.pdf_extract.parser.parser import parse_page_range


class TestParsePageRange:
    """Test parse_page_range() function with various input formats."""

    def test_single_page(self):
        """Test parsing a single page number."""
        assert parse_page_range("5") == (5, 5)
        assert parse_page_range("1") == (1, 1)
        assert parse_page_range("100") == (100, 100)

    def test_single_page_with_whitespace(self):
        """Test parsing a single page with leading/trailing whitespace."""
        assert parse_page_range("  5  ") == (5, 5)
        assert parse_page_range("\t10\n") == (10, 10)

    def test_explicit_range(self):
        """Test parsing an explicit page range (e.g., '5-10')."""
        assert parse_page_range("5-10") == (5, 10)
        assert parse_page_range("1-3") == (1, 3)
        assert parse_page_range("10-100") == (10, 100)

    def test_explicit_range_with_whitespace(self):
        """Test parsing ranges with whitespace around numbers."""
        assert parse_page_range(" 5 - 10 ") == (5, 10)
        assert parse_page_range("1-  3") == (1, 3)

    def test_same_start_and_end(self):
        """Test range where start equals end."""
        assert parse_page_range("5-5") == (5, 5)

    def test_open_end_range(self):
        """Test 'from page X to end' format (e.g., '10-')."""
        assert parse_page_range("10-") == (10, None)
        assert parse_page_range("1-") == (1, None)
        assert parse_page_range("  5  -  ") == (5, None)

    def test_open_start_range(self):
        """Test 'from start to page X' format (e.g., '-5')."""
        assert parse_page_range("-5") == (None, 5)
        assert parse_page_range("-1") == (None, 1)
        assert parse_page_range("  -  10  ") == (None, 10)

    def test_invalid_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Page range cannot be empty"):
            parse_page_range("")
        with pytest.raises(ValueError, match="Page range cannot be empty"):
            parse_page_range("   ")

    def test_invalid_double_dash(self):
        """Test that '-' alone raises ValueError."""
        with pytest.raises(ValueError, match="At least one page number required"):
            parse_page_range("-")

    def test_invalid_non_numeric(self):
        """Test that non-numeric input raises ValueError."""
        with pytest.raises(ValueError, match="Invalid page number"):
            parse_page_range("abc")
        with pytest.raises(ValueError, match="Invalid end page number"):
            parse_page_range("5-abc")
        with pytest.raises(ValueError, match="Invalid start page number"):
            parse_page_range("abc-10")
        with pytest.raises(ValueError, match="Invalid end page number"):
            parse_page_range("-abc")

    def test_invalid_negative_numbers(self):
        """Test that negative page numbers raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_page_range("0")
        with pytest.raises(ValueError, match="must be >= 1"):
            # Double dash creates empty start_str and "-1" as end_str
            parse_page_range("--1")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_page_range("5-0")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_page_range("0-5")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_page_range("-0")

    def test_invalid_start_greater_than_end(self):
        """Test that start > end raises ValueError."""
        with pytest.raises(ValueError, match="cannot be greater than end page"):
            parse_page_range("10-5")
        with pytest.raises(ValueError, match="cannot be greater than end page"):
            parse_page_range("100-1")

    def test_invalid_float(self):
        """Test that float numbers raise ValueError."""
        with pytest.raises(ValueError, match="Invalid page"):
            parse_page_range("5.5")
        with pytest.raises(ValueError, match="Invalid end page number"):
            parse_page_range("5-10.5")
