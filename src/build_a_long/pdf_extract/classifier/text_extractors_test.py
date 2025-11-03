"""
Tests for text extraction utilities.
"""

from build_a_long.pdf_extract.classifier.text_extractors import (
    extract_page_number_value,
    extract_part_count_value,
    extract_step_number_value,
)


class TestExtractPageNumberValue:
    """Tests for extract_page_number_value function."""

    def test_plain_single_digit(self) -> None:
        assert extract_page_number_value("1") == 1
        assert extract_page_number_value("5") == 5
        assert extract_page_number_value("9") == 9

    def test_plain_multi_digit(self) -> None:
        assert extract_page_number_value("12") == 12
        assert extract_page_number_value("42") == 42
        assert extract_page_number_value("123") == 123

    def test_leading_zeros(self) -> None:
        assert extract_page_number_value("007") == 7
        assert extract_page_number_value("001") == 1
        assert extract_page_number_value("012") == 12
        assert extract_page_number_value("0123") == 123

    def test_with_page_prefix(self) -> None:
        assert extract_page_number_value("page 1") == 1
        assert extract_page_number_value("Page 12") == 12
        assert extract_page_number_value("PAGE 123") == 123
        assert extract_page_number_value("page 007") == 7

    def test_with_p_prefix(self) -> None:
        assert extract_page_number_value("p. 1") == 1
        assert extract_page_number_value("P. 12") == 12
        assert extract_page_number_value("p 5") == 5
        assert extract_page_number_value("P 42") == 42

    def test_with_whitespace(self) -> None:
        assert extract_page_number_value("  42  ") == 42
        assert extract_page_number_value("\t12\t") == 12
        assert extract_page_number_value("  page  5  ") == 5

    def test_invalid_inputs(self) -> None:
        assert extract_page_number_value("abc") is None
        assert extract_page_number_value("") is None
        assert extract_page_number_value("   ") is None
        assert extract_page_number_value("1.5") is None
        assert extract_page_number_value("1a") is None
        assert extract_page_number_value("a1") is None

    def test_too_many_digits(self) -> None:
        # Should reject 4+ digit numbers
        assert extract_page_number_value("1234") is None
        assert extract_page_number_value("12345") is None

    def test_zero_page_number(self) -> None:
        # Zero is a valid page number (though unusual)
        assert extract_page_number_value("0") == 0
        assert extract_page_number_value("00") == 0
        assert extract_page_number_value("000") == 0


class TestExtractStepNumberValue:
    """Tests for extract_step_number_value function."""

    def test_single_digit(self) -> None:
        assert extract_step_number_value("1") == 1
        assert extract_step_number_value("5") == 5
        assert extract_step_number_value("9") == 9

    def test_multi_digit(self) -> None:
        assert extract_step_number_value("10") == 10
        assert extract_step_number_value("42") == 42
        assert extract_step_number_value("123") == 123
        assert extract_step_number_value("9999") == 9999

    def test_with_whitespace(self) -> None:
        assert extract_step_number_value("  42  ") == 42
        assert extract_step_number_value("\t12\t") == 12

    def test_invalid_inputs(self) -> None:
        assert extract_step_number_value("abc") is None
        assert extract_step_number_value("") is None
        assert extract_step_number_value("   ") is None
        assert extract_step_number_value("1.5") is None
        assert extract_step_number_value("1a") is None
        assert extract_step_number_value("a1") is None

    def test_zero_step_number(self) -> None:
        # Step numbers should start at 1, not 0
        assert extract_step_number_value("0") is None
        assert extract_step_number_value("00") is None

    def test_leading_zeros(self) -> None:
        # Step numbers should not have leading zeros
        assert extract_step_number_value("01") is None
        assert extract_step_number_value("007") is None

    def test_too_many_digits(self) -> None:
        # Should reject 5+ digit numbers
        assert extract_step_number_value("10000") is None
        assert extract_step_number_value("12345") is None


class TestExtractPartCountValue:
    """Tests for extract_part_count_value function."""

    def test_lowercase_x(self) -> None:
        assert extract_part_count_value("1x") == 1
        assert extract_part_count_value("2x") == 2
        assert extract_part_count_value("10x") == 10
        assert extract_part_count_value("123x") == 123

    def test_uppercase_x(self) -> None:
        assert extract_part_count_value("1X") == 1
        assert extract_part_count_value("2X") == 2
        assert extract_part_count_value("10X") == 10

    def test_multiplication_sign(self) -> None:
        assert extract_part_count_value("1×") == 1
        assert extract_part_count_value("2×") == 2
        assert extract_part_count_value("10×") == 10

    def test_with_spaces(self) -> None:
        assert extract_part_count_value("1 x") == 1
        assert extract_part_count_value("2 X") == 2
        assert extract_part_count_value("10 ×") == 10
        assert extract_part_count_value("  5  x  ") == 5

    def test_with_whitespace(self) -> None:
        assert extract_part_count_value("  2x  ") == 2
        assert extract_part_count_value("\t3×\t") == 3

    def test_invalid_inputs(self) -> None:
        assert extract_part_count_value("abc") is None
        assert extract_part_count_value("") is None
        assert extract_part_count_value("   ") is None
        assert extract_part_count_value("1") is None  # Missing x
        assert extract_part_count_value("x") is None  # Missing number
        assert extract_part_count_value("1.5x") is None
        assert extract_part_count_value("x2") is None  # Wrong order

    def test_zero_count(self) -> None:
        # Zero count doesn't make sense but should be handled
        assert extract_part_count_value("0x") == 0
        assert extract_part_count_value("00x") == 0

    def test_too_many_digits(self) -> None:
        # Should reject 4+ digit numbers
        assert extract_part_count_value("1234x") is None
        assert extract_part_count_value("12345x") is None
