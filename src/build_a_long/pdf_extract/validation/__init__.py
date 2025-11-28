"""Validation module for PDF extraction results.

This module provides validation rules to check for common issues that indicate
the extraction may not be working correctly for a particular instruction book.
"""

from .printer import print_validation
from .rules import (
    format_ranges,
    validate_first_page_number,
    validate_missing_page_numbers,
    validate_page_number_sequence,
    validate_step_sequence,
    validate_steps_have_parts,
)
from .runner import validate_results
from .types import ValidationIssue, ValidationResult, ValidationSeverity

__all__ = [
    # Types
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
    # Main runner
    "validate_results",
    # Printer
    "print_validation",
    # Individual rules (for testing/customization)
    "format_ranges",
    "validate_first_page_number",
    "validate_missing_page_numbers",
    "validate_page_number_sequence",
    "validate_step_sequence",
    "validate_steps_have_parts",
]
