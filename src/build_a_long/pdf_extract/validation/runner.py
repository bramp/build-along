"""Main validation runner that orchestrates all validation rules."""

from build_a_long.pdf_extract.classifier import ClassificationResult
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import Page

from .rules import (
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
from .types import ValidationResult


def validate_results(
    pages: list[PageData],
    results: list[ClassificationResult],
) -> ValidationResult:
    """Run all validation rules on classification results.

    This function checks for common issues that indicate the extraction
    may not be working correctly for a particular instruction book.

    Validation rules:
    - Each page should have a page number detected
    - Step numbers should form a continuous sequence without gaps
    - Step numbers should not have duplicates
    - Pages with steps should have parts lists
    - The first page number found should be reasonable (typically 1-4)

    Args:
        pages: List of PageData containing extracted elements
        results: List of ClassificationResult with labels

    Returns:
        ValidationResult containing all found issues
    """
    validation = ValidationResult()

    # Collect data for validation
    missing_page_numbers: list[int] = []
    step_numbers_seen: list[tuple[int, int]] = []  # (pdf_page, step_number)
    steps_without_parts: list[tuple[int, int]] = []  # (pdf_page, step_number)
    lego_page_numbers: list[int] = []  # Detected LEGO page numbers

    for page_data, result in zip(pages, results, strict=True):
        page = result.page
        pdf_page = page_data.page_number

        # Check for page number
        if page and page.page_number:
            lego_page_numbers.append(page.page_number.value)
        else:
            missing_page_numbers.append(pdf_page)

        # Collect step numbers
        if page:
            for step in page.steps:
                step_numbers_seen.append((pdf_page, step.step_number.value))

                # Check for steps without parts lists
                if step.parts_list is None or len(step.parts_list.parts) == 0:
                    steps_without_parts.append((pdf_page, step.step_number.value))

    # --- Validation Rules ---

    # Rule 1: Missing page numbers
    validate_missing_page_numbers(validation, missing_page_numbers, len(pages))

    # Rule 2 & 3: Step number sequence validation
    validate_step_sequence(validation, step_numbers_seen)

    # Rule 4: Steps without parts lists
    validate_steps_have_parts(validation, steps_without_parts)

    # Rule 5: First page number validation
    validate_first_page_number(validation, lego_page_numbers)

    # Rule 6: Page number sequence validation
    validate_page_number_sequence(validation, lego_page_numbers)

    return validation


def validate_page(
    page: Page,
    page_data: PageData,
    validation: ValidationResult | None = None,
    *,
    step_overlap_threshold: float = 0.05,
) -> ValidationResult:
    """Run domain invariant validation rules on a single classified page.

    This function checks structural/spatial properties of elements on a page
    to ensure they satisfy LEGO instruction layout invariants.

    Domain invariant rules:
    - Each PartsList should contain at least one Part
    - PartsList bounding boxes should not overlap
    - Step bounding boxes should not significantly overlap
    - Part bbox should contain its count and diagram bboxes
    - All elements should stay within page boundaries
    - Content elements should not overlap page metadata

    Args:
        page: The classified Page object
        page_data: The raw PageData for context (page number, bbox, source)
        validation: Optional existing ValidationResult to add to.
            If None, a new one is created.
        step_overlap_threshold: Maximum allowed IOU for step overlap (default 5%)

    Returns:
        ValidationResult containing all found issues
    """
    if validation is None:
        validation = ValidationResult()

    # Rule 1: Parts lists should have parts
    validate_parts_list_has_parts(validation, page, page_data)

    # Rule 2: Parts lists should not overlap
    validate_parts_lists_no_overlap(validation, page, page_data)

    # Rule 3: Steps should not significantly overlap
    validate_steps_no_significant_overlap(
        validation, page, page_data, step_overlap_threshold
    )

    # Rule 4: Part bbox should contain children
    validate_part_contains_children(validation, page, page_data)

    # Rule 5: Elements should stay within page bounds
    validate_elements_within_page(validation, page, page_data)

    # Rule 6: Content should not overlap metadata
    validate_content_no_metadata_overlap(validation, page, page_data)

    return validation
