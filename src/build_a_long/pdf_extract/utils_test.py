"""Tests for PDF extraction utilities."""

from build_a_long.pdf_extract.utils import (
    remove_empty_lists,
    reorder_tag_first,
    round_floats,
)


class TestRoundFloats:
    """Tests for round_floats function."""

    def test_round_float(self) -> None:
        """Test rounding a single float."""
        assert round_floats(3.14159) == 3.14

    def test_round_float_custom_decimals(self) -> None:
        """Test rounding with custom decimal places."""
        assert round_floats(3.14159, decimals=3) == 3.142

    def test_round_dict_values(self) -> None:
        """Test rounding floats in a dict."""
        data = {"x": 1.234, "y": 5.678}
        result = round_floats(data)
        assert result == {"x": 1.23, "y": 5.68}

    def test_round_nested_dict(self) -> None:
        """Test rounding floats in nested dicts."""
        data = {"outer": {"inner": 1.999}}
        result = round_floats(data)
        assert result == {"outer": {"inner": 2.0}}

    def test_round_list_values(self) -> None:
        """Test rounding floats in a list."""
        data = [1.111, 2.222, 3.333]
        result = round_floats(data)
        assert result == [1.11, 2.22, 3.33]

    def test_round_preserves_non_floats(self) -> None:
        """Test that non-floats are preserved."""
        data = {"int": 42, "str": "hello", "float": 1.234}
        result = round_floats(data)
        assert result == {"int": 42, "str": "hello", "float": 1.23}


class TestRemoveEmptyLists:
    """Tests for remove_empty_lists function."""

    def test_remove_empty_list_from_dict(self) -> None:
        """Test removing empty lists from a dict."""
        data = {"items": [], "value": 42}
        result = remove_empty_lists(data)
        assert result == {"value": 42}

    def test_preserve_non_empty_lists(self) -> None:
        """Test that non-empty lists are preserved."""
        data = {"items": [1, 2, 3], "value": 42}
        result = remove_empty_lists(data)
        assert result == {"items": [1, 2, 3], "value": 42}

    def test_nested_empty_lists(self) -> None:
        """Test removing empty lists from nested structures."""
        data = {"outer": {"inner": [], "value": 1}}
        result = remove_empty_lists(data)
        assert result == {"outer": {"value": 1}}


class TestReorderTagFirst:
    """Tests for reorder_tag_first function."""

    def test_tag_moved_to_first_position(self) -> None:
        """Test that __tag__ is moved to first position."""
        data = {"bbox": {"x": 0}, "__tag__": "Test", "value": 42}
        result = reorder_tag_first(data)
        first_key = next(iter(result.keys()))
        assert first_key == "__tag__"

    def test_preserves_all_keys(self) -> None:
        """Test that all keys are preserved after reordering."""
        data = {"bbox": {"x": 0}, "__tag__": "Test", "value": 42}
        result = reorder_tag_first(data)
        assert set(result.keys()) == {"bbox", "__tag__", "value"}
        assert result["__tag__"] == "Test"
        assert result["value"] == 42

    def test_nested_dicts_reordered(self) -> None:
        """Test that __tag__ is moved to first in nested dicts too."""
        data = {
            "bbox": {"x": 0},
            "__tag__": "Outer",
            "child": {"value": 1, "__tag__": "Inner"},
        }
        result = reorder_tag_first(data)

        # Check outer dict
        outer_first_key = next(iter(result.keys()))
        assert outer_first_key == "__tag__"

        # Check nested dict
        inner_first_key = next(iter(result["child"].keys()))
        assert inner_first_key == "__tag__"

    def test_lists_with_dicts_reordered(self) -> None:
        """Test that dicts inside lists are also reordered."""
        data = {
            "__tag__": "Parent",
            "items": [
                {"value": 1, "__tag__": "Item1"},
                {"value": 2, "__tag__": "Item2"},
            ],
        }
        result = reorder_tag_first(data)

        # Check each item in list
        for item in result["items"]:
            first_key = next(iter(item.keys()))
            assert first_key == "__tag__"

    def test_dict_without_tag_unchanged(self) -> None:
        """Test that dicts without __tag__ are unchanged."""
        data = {"x": 0, "y": 1, "z": 2}
        result = reorder_tag_first(data)
        assert result == data

    def test_non_dict_values_unchanged(self) -> None:
        """Test that non-dict values pass through unchanged."""
        assert reorder_tag_first(42) == 42
        assert reorder_tag_first("hello") == "hello"
        assert reorder_tag_first([1, 2, 3]) == [1, 2, 3]
        assert reorder_tag_first(None) is None
