"""Validation module for PDF extraction results.

This module provides validation rules to check for common issues that indicate
the extraction may not be working correctly for a particular instruction book.
"""

from .printer import print_validation
from .rules import (
    format_ranges,
    validate_content_no_metadata_overlap,
    validate_elements_within_page,
    validate_first_page_number,
    validate_missing_page_numbers,
    validate_page_number_sequence,
    validate_part_contains_children,
    validate_parts_list_has_parts,
    validate_parts_lists_no_overlap,
    validate_step_sequence,
    validate_steps_have_parts,
    validate_steps_no_significant_overlap,
)
from .runner import validate_page, validate_results
from .types import ValidationIssue, ValidationResult, ValidationSeverity

__all__ = [
    # Types
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
    # Main runners
    "validate_results",
    "validate_page",
    # Printer
    "print_validation",
    # Sequence validation rules (cross-page)
    "format_ranges",
    "validate_first_page_number",
    "validate_missing_page_numbers",
    "validate_page_number_sequence",
    "validate_step_sequence",
    "validate_steps_have_parts",
    # Domain invariant rules (per-page structural)
    "validate_content_no_metadata_overlap",
    "validate_elements_within_page",
    "validate_part_contains_children",
    "validate_parts_list_has_parts",
    "validate_parts_lists_no_overlap",
    "validate_steps_no_significant_overlap",
]
