"""
Text extraction utilities for LEGO instruction elements.

This module provides a centralized location for text parsing logic that is
shared between classifiers (for scoring) and builders (for constructing
structured elements).

Design Principles
-----------------
- All extraction functions are pure functions (no side effects)
- All functions are thoroughly type-hinted
- All functions handle edge cases gracefully (return None on failure)
- All functions are independently testable
- Extraction logic is the single source of truth for both classification
  and construction
"""

import re


def extract_page_number_value(text: str) -> int | None:
    """Extract numeric page number value from text.

    Handles various formats:
    - Plain numbers: "1", "12", "123"
    - Leading zeros: "001", "012"
    - With prefix: "page 1", "p. 12", "Page 001"

    Args:
        text: Text potentially containing a page number

    Returns:
        Integer page number if found, None otherwise

    Examples:
        >>> extract_page_number_value("42")
        42
        >>> extract_page_number_value("007")
        7
        >>> extract_page_number_value("page 12")
        12
        >>> extract_page_number_value("P. 5")
        5
        >>> extract_page_number_value("abc")
        None
    """
    t = text.strip()

    # Match plain numbers with optional leading zeros
    m = re.match(r"^0*(\d{1,3})$", t)
    if m:
        return int(m.group(1))

    # Match "page" or "p." prefix with optional leading zeros
    m = re.match(r"^(?:page|p\.?)\s*0*(\d{1,3})$", t, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def extract_step_number_value(text: str) -> int | None:
    """Extract numeric step number value from text.

    Handles plain numeric text representing step numbers (1-9999).

    Args:
        text: Text potentially containing a step number

    Returns:
        Integer step number if found, None otherwise

    Examples:
        >>> extract_step_number_value("1")
        1
        >>> extract_step_number_value("42")
        42
        >>> extract_step_number_value("1234")
        1234
        >>> extract_step_number_value("0")
        None
        >>> extract_step_number_value("abc")
        None
    """
    t = text.strip()

    # Match step numbers: must start with 1-9, can have up to 3 more digits
    if re.fullmatch(r"[1-9]\d{0,3}", t):
        return int(t)

    return None


def extract_part_count_value(text: str) -> int | None:
    """Extract numeric part count value from text.

    Handles part count formats like "2x", "3X", "5×", etc.

    Args:
        text: Text potentially containing a part count

    Returns:
        Integer count if found, None otherwise

    Examples:
        >>> extract_part_count_value("2x")
        2
        >>> extract_part_count_value("3X")
        3
        >>> extract_part_count_value("5×")
        5
        >>> extract_part_count_value("12 x")
        12
        >>> extract_part_count_value("abc")
        None
    """
    t = text.strip()

    # Match count pattern: 1-3 digits followed by x or ×
    m = re.fullmatch(r"(\d{1,3})\s*[x×]", t, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def extract_bag_number_value(text: str) -> int | None:
    """Extract numeric bag number value from text.

    Handles plain numeric text representing bag numbers (1-99).
    Bag numbers are typically small positive integers.

    Args:
        text: Text potentially containing a bag number

    Returns:
        Integer bag number if found, None otherwise

    Examples:
        >>> extract_bag_number_value("1")
        1
        >>> extract_bag_number_value("2")
        2
        >>> extract_bag_number_value("10")
        10
        >>> extract_bag_number_value("0")
        None
        >>> extract_bag_number_value("100")
        None
        >>> extract_bag_number_value("abc")
        None
    """
    t = text.strip()

    # Match bag numbers: must start with 1-9, can have 0-1 more digits (1-99)
    if re.fullmatch(r"[1-9]\d?", t):
        return int(t)

    return None


def extract_element_id(text: str) -> str | None:
    """Extract LEGO element ID from text.

    Element IDs are 4-8 digit numbers that never start with zero.
    Distribution:
    - Length 4: 0.02%
    - Length 5: 0.50%
    - Length 6: 4.98%
    - Length 7: 94.47% (most common)
    - Length 8: 0.03%

    Args:
        text: The text to extract from

    Returns:
        The element ID string if valid, None otherwise
    """
    stripped_text = text.strip()
    if not stripped_text.isdigit():
        return None

    # Element IDs never start with zero
    if stripped_text.startswith("0"):
        return None

    # Element IDs are typically 4-8 digit numbers
    num_digits = len(stripped_text)
    if 4 <= num_digits <= 8:
        return stripped_text

    return None
