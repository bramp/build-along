"""Main validation runner that orchestrates all validation rules."""

from build_a_long.pdf_extract.classifier import BatchClassificationResult
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import Page

from .rules import (
    validate_catalog_coverage,
    validate_content_no_metadata_overlap,
    validate_elements_within_page,
    validate_first_page_number,
    validate_invalid_pages,
    validate_missing_page_numbers,
    validate_page_number_sequence,
    validate_part_contains_children,
    validate_parts_list_has_parts,
    validate_parts_lists_no_overlap,
    validate_progress_bar_sequence,
    validate_skipped_pages,
    validate_step_sequence,
    validate_steps_have_parts,
    validate_steps_no_significant_overlap,
    validate_unassigned_blocks,
)
from .types import ValidationResult


def validate_results(
    batch_result: BatchClassificationResult,
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
        batch_result: The complete batch classification result.

    Returns:
        ValidationResult containing all found issues
    """
    validation = ValidationResult()

    # Collect data for validation
    missing_page_numbers: list[int] = []
    step_numbers_seen: list[tuple[int, int]] = []  # (pdf_page, step_number)
    steps_without_parts: list[tuple[int, int]] = []  # (pdf_page, step_number)
    lego_page_numbers: list[int] = []  # Detected LEGO page numbers
    skipped_pages: list[tuple[int, str]] = []  # (pdf_page, reason)
    invalid_pages: list[int] = []  # Pages where classification produced no Page
    progress_bars: list[tuple[int, float]] = []  # (pdf_page, progress_value)

    for result in batch_result.results:
        page = result.page
        page_data = result.page_data
        pdf_page = page_data.page_number

        # Check for skipped pages
        if result.skipped_reason:
            skipped_pages.append((pdf_page, result.skipped_reason))
            continue  # Don't collect other data for skipped pages

        # Check for unassigned blocks
        validate_unassigned_blocks(validation, result)

        # Check for invalid pages (no Page object but also not skipped)
        if page is None:
            invalid_pages.append(pdf_page)
            continue

        # Check for page number
        if page.page_number:
            lego_page_numbers.append(page.page_number.value)
        else:
            missing_page_numbers.append(pdf_page)

        # Collect progress bar value
        if page.progress_bar and page.progress_bar.progress is not None:
            progress_bars.append((pdf_page, page.progress_bar.progress))

        # Collect step numbers
        if page:
            for step in page.steps:
                step_numbers_seen.append((pdf_page, step.step_number.value))

                # Check for steps without parts lists
                if step.parts_list is None or len(step.parts_list.parts) == 0:
                    steps_without_parts.append((pdf_page, step.step_number.value))

    # --- Validation Rules ---

    # Rule 0: Skipped pages
    validate_skipped_pages(validation, skipped_pages)

    # Rule 0b: Invalid pages (classification failed to produce a Page)
    validate_invalid_pages(validation, invalid_pages)

    # Rule 1: Missing page numbers
    validate_missing_page_numbers(
        validation, missing_page_numbers, len(batch_result.results)
    )

    # Rule 2 & 3: Step number sequence validation
    validate_step_sequence(validation, step_numbers_seen)

    # Rule 4: Steps without parts lists
    validate_steps_have_parts(validation, steps_without_parts)

    # Rule 5: First page number validation
    validate_first_page_number(validation, lego_page_numbers)

    # Rule 6: Page number sequence validation
    validate_page_number_sequence(validation, lego_page_numbers)

    # Rule 7: Progress bar sequence validation
    validate_progress_bar_sequence(validation, progress_bars)

    # Rule 8: Catalog coverage
    validate_catalog_coverage(validation, batch_result.manual, experimental=True)

    return validation

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
