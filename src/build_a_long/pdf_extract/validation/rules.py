"""Individual validation rules for classification results."""

from build_a_long.pdf_extract.classifier import ClassificationResult
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import Page

from .types import ValidationIssue, ValidationResult, ValidationSeverity

# =============================================================================
# Sequence Validation Rules (cross-page)
# =============================================================================


def validate_missing_page_numbers(
    validation: ValidationResult,
    missing_page_numbers: list[int],
    total_pages: int,
) -> None:
    """Validate that pages have detected page numbers.

    Args:
        validation: ValidationResult to add issues to
        missing_page_numbers: List of PDF page numbers missing LEGO page numbers
        total_pages: Total number of pages in the document
    """
    if not missing_page_numbers:
        return

    coverage = (total_pages - len(missing_page_numbers)) / total_pages * 100
    severity = (
        ValidationSeverity.ERROR
        if coverage < 50
        else ValidationSeverity.WARNING
        if coverage < 90
        else ValidationSeverity.INFO
    )

    validation.add(
        ValidationIssue(
            severity=severity,
            rule="missing_page_numbers",
            message=f"{len(missing_page_numbers)} pages missing detected page numbers "
            f"({coverage:.0f}% coverage)",
            pages=missing_page_numbers[:20],  # Limit to first 20
            details=f"Total: {len(missing_page_numbers)} pages"
            if len(missing_page_numbers) > 20
            else None,
        )
    )


def validate_step_sequence(
    validation: ValidationResult,
    step_numbers_seen: list[tuple[int, int]],
) -> None:
    """Validate step numbers form a proper sequence.

    Checks for:
    - Duplicate step numbers
    - Gaps in the step sequence
    - Step sequence not starting at 1

    Args:
        validation: ValidationResult to add issues to
        step_numbers_seen: List of (pdf_page, step_number) tuples
    """
    if not step_numbers_seen:
        return

    # Sort by step number to find gaps
    sorted_steps = sorted(step_numbers_seen, key=lambda x: x[1])
    step_values = [s[1] for s in sorted_steps]

    # Check for duplicates
    duplicates: dict[int, list[int]] = {}
    for pdf_page, step_num in step_numbers_seen:
        if step_values.count(step_num) > 1:
            if step_num not in duplicates:
                duplicates[step_num] = []
            duplicates[step_num].append(pdf_page)

    if duplicates:
        dup_details = ", ".join(
            f"step {s} on pages {sorted(set(p))}" for s, p in sorted(duplicates.items())
        )
        validation.add(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule="duplicate_steps",
                message=f"Found {len(duplicates)} duplicate step numbers",
                pages=sorted(
                    set(p for pages_list in duplicates.values() for p in pages_list)
                ),
                details=dup_details[:200],  # Limit details length
            )
        )

    # Check for gaps (unique step numbers)
    unique_steps = sorted(set(step_values))
    if unique_steps:
        expected_steps = set(range(unique_steps[0], unique_steps[-1] + 1))
        actual_steps = set(unique_steps)
        missing_steps = sorted(expected_steps - actual_steps)

        if missing_steps:
            validation.add(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    rule="step_gaps",
                    message=f"Found {len(missing_steps)} gaps in step sequence",
                    details=f"Missing steps: {format_ranges(missing_steps)}",
                )
            )

        # Check if first step is 1
        if unique_steps[0] != 1:
            validation.add(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    rule="step_start",
                    message=f"Step sequence starts at {unique_steps[0]}, not 1",
                    details="This may indicate the PDF is missing initial pages",
                )
            )


def validate_steps_have_parts(
    validation: ValidationResult,
    steps_without_parts: list[tuple[int, int]],
) -> None:
    """Validate that steps have associated parts lists.

    Args:
        validation: ValidationResult to add issues to
        steps_without_parts: List of (pdf_page, step_number) tuples for steps
            that have no parts list
    """
    if not steps_without_parts:
        return

    # Extract unique page numbers for the pages field
    pages = sorted(set(page for page, _ in steps_without_parts))

    # Format step details (show step numbers with their pages)
    step_details = ", ".join(
        f"step {step} (p.{page})" for page, step in steps_without_parts[:10]
    )
    if len(steps_without_parts) > 10:
        step_details += f" ... ({len(steps_without_parts)} total)"

    validation.add(
        ValidationIssue(
            severity=ValidationSeverity.INFO,
            rule="steps_without_parts",
            message=f"{len(steps_without_parts)} steps without detected parts lists",
            pages=pages[:20],
            details=f"Steps: {step_details}",
        )
    )


def validate_first_page_number(
    validation: ValidationResult,
    lego_page_numbers: list[int],
) -> None:
    """Validate the first detected page number is reasonable.

    Args:
        validation: ValidationResult to add issues to
        lego_page_numbers: List of detected LEGO page numbers in order
    """
    if not lego_page_numbers:
        validation.add(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                rule="no_page_numbers",
                message="No page numbers detected in entire document",
                details="Page number detection may be failing completely",
            )
        )
        return

    first_page = lego_page_numbers[0]
    if first_page > 10:
        validation.add(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule="high_first_page",
                message=f"First detected page number is {first_page}",
                details="This may indicate initial pages are missing or "
                "page number detection is misidentifying numbers",
            )
        )


def validate_page_number_sequence(
    validation: ValidationResult,
    lego_page_numbers: list[int],
) -> None:
    """Validate page numbers form a reasonable sequence.

    Checks for:
    - Page numbers that decrease (should monotonically increase)
    - Gaps in page numbers in the middle (first N and last M pages can be missing,
      but no gaps in between)

    Args:
        validation: ValidationResult to add issues to
        lego_page_numbers: List of detected LEGO page numbers in order
    """
    if len(lego_page_numbers) < 2:
        return

    # Check for non-monotonic sequence (page numbers should generally increase)
    decreases: list[tuple[int, int, int]] = []  # (index, from_value, to_value)
    for i in range(1, len(lego_page_numbers)):
        if lego_page_numbers[i] < lego_page_numbers[i - 1]:
            decreases.append((i, lego_page_numbers[i - 1], lego_page_numbers[i]))

    if decreases:
        validation.add(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule="page_sequence",
                message=f"Page numbers decrease {len(decreases)} time(s)",
                details="Page numbers should generally increase. "
                "This may indicate misdetected page numbers.",
            )
        )

    # Check for gaps in page numbers in the middle of the sequence.
    # We allow:
    # - First N pages to be missing (start at any page number)
    # - Last M pages to be missing (end before the expected last page)
    # But NO gaps in between - pages should be consecutive.
    #
    # To detect this, we look for any gap > 1 between consecutive detected pages.
    # The first and last page numbers are allowed to be anything, but the
    # sequence between them must be complete.
    gaps: list[tuple[int, int, int]] = []  # (pdf_page_index, from_value, to_value)
    for i in range(1, len(lego_page_numbers)):
        gap = lego_page_numbers[i] - lego_page_numbers[i - 1]
        if gap > 1:  # Pages should be consecutive (increment by 1)
            gaps.append((i, lego_page_numbers[i - 1], lego_page_numbers[i]))

    if gaps:
        gap_details = ", ".join(f"{f}->{t}" for _, f, t in gaps[:5])
        validation.add(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule="page_gaps",
                message=f"Found {len(gaps)} gap(s) in page number sequence",
                details=f"Gaps: {gap_details}" + (" ..." if len(gaps) > 5 else ""),
            )
        )


def validate_skipped_pages(
    validation: ValidationResult,
    skipped_pages: list[tuple[int, str]],
) -> None:
    """Validate and report pages that were skipped during classification.

    Args:
        validation: ValidationResult to add issues to
        skipped_pages: List of (pdf_page, reason) tuples for skipped pages
    """
    if not skipped_pages:
        return

    pages = [p for p, _ in skipped_pages]
    # All reasons should be similar, just take the first one for the message
    sample_reason = skipped_pages[0][1] if skipped_pages else ""

    validation.add(
        ValidationIssue(
            severity=ValidationSeverity.WARNING,
            rule="skipped_pages",
            message=f"{len(skipped_pages)} page(s) skipped during classification",
            pages=pages,
            details=sample_reason if len(skipped_pages) == 1 else None,
        )
    )


def validate_invalid_pages(
    validation: ValidationResult,
    invalid_pages: list[int],
) -> None:
    """Validate and report pages where classification failed to produce a Page.

    These are pages that weren't explicitly skipped but still couldn't be
    converted to a valid Page object during classification.

    Args:
        validation: ValidationResult to add issues to
        invalid_pages: List of PDF page numbers that failed classification
    """
    if not invalid_pages:
        return

    validation.add(
        ValidationIssue(
            severity=ValidationSeverity.WARNING,
            rule="invalid_pages",
            message=f"{len(invalid_pages)} page(s) failed to produce valid classification",
            pages=invalid_pages,
        )
    )


def format_ranges(numbers: list[int]) -> str:
    """Format a list of numbers as ranges (e.g., '1-3, 5, 7-9').

    Args:
        numbers: List of integers to format

    Returns:
        Formatted string with ranges collapsed
    """
    if not numbers:
        return ""

    ranges: list[str] = []
    start = numbers[0]
    end = numbers[0]

    for num in numbers[1:]:
        if num == end + 1:
            end = num
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = num

    ranges.append(f"{start}-{end}" if start != end else str(start))

    # Limit output length
    result = ", ".join(ranges)
    if len(result) > 100:
        return result[:97] + "..."
    return result


# =============================================================================
# Domain Invariant Validation Rules (per-page structural checks)
# =============================================================================


def validate_parts_list_has_parts(
    validation: ValidationResult,
    page: Page,
    page_data: PageData,
) -> None:
    """Validate that each PartsList contains at least one Part.

    Domain Invariant: A parts list without parts doesn't make sense in the
    context of LEGO instructions. Each PartsList should contain ≥1 Part objects.

    Args:
        validation: ValidationResult to add issues to
        page: The classified Page object
        page_data: The raw PageData for context (page number, source)
    """
    for step in page.steps:
        if step.parts_list is None:
            continue

        if len(step.parts_list.parts) == 0:
            validation.add(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    rule="empty_parts_list",
                    message=f"Step {step.step_number.value} has empty PartsList",
                    pages=[page_data.page_number],
                    details=f"PartsList at {step.parts_list.bbox} has no parts",
                )
            )


def validate_parts_lists_no_overlap(
    validation: ValidationResult,
    page: Page,
    page_data: PageData,
) -> None:
    """Validate that PartsList bounding boxes do not overlap.

    Domain Invariant: Each parts list occupies a distinct region on the page.
    Overlapping parts lists would indicate a classification error.

    Args:
        validation: ValidationResult to add issues to
        page: The classified Page object
        page_data: The raw PageData for context
    """
    parts_lists = [step.parts_list for step in page.steps if step.parts_list]

    for i, pl1 in enumerate(parts_lists):
        for pl2 in parts_lists[i + 1 :]:
            overlap = pl1.bbox.overlaps(pl2.bbox)
            if overlap > 0.0:
                validation.add(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        rule="overlapping_parts_lists",
                        message="PartsList regions overlap",
                        pages=[page_data.page_number],
                        details=f"{pl1.bbox} and {pl2.bbox} overlap "
                        f"(IOU: {pl1.bbox.iou(pl2.bbox):.3f})",
                    )
                )


def validate_steps_no_significant_overlap(
    validation: ValidationResult,
    page: Page,
    page_data: PageData,
    overlap_threshold: float = 0.05,
) -> None:
    """Validate that Step bounding boxes do not significantly overlap.

    Domain Invariant: Steps should occupy distinct regions on the page.
    Some minor overlap is acceptable (e.g., at boundaries), but significant
    overlap would indicate a classification error.

    Args:
        validation: ValidationResult to add issues to
        page: The classified Page object
        page_data: The raw PageData for context
        overlap_threshold: Maximum allowed IOU (default 5%)
    """
    if len(page.steps) < 2:
        return

    for i, step1 in enumerate(page.steps):
        for step2 in page.steps[i + 1 :]:
            iou = step1.bbox.iou(step2.bbox)
            if iou > overlap_threshold:
                validation.add(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        rule="overlapping_steps",
                        message=f"Steps {step1.step_number.value} and "
                        f"{step2.step_number.value} overlap significantly",
                        pages=[page_data.page_number],
                        details=f"IOU: {iou:.3f} (threshold: {overlap_threshold})",
                    )
                )


def validate_part_contains_children(
    validation: ValidationResult,
    page: Page,
    page_data: PageData,
) -> None:
    """Validate that Part bbox contains its count and diagram bboxes.

    Domain Invariant: A Part represents a cohesive visual unit consisting of
    the part diagram (image) and the count label. The Part's bounding box
    should encompass both elements.

    Args:
        validation: ValidationResult to add issues to
        page: The classified Page object
        page_data: The raw PageData for context
    """
    for step in page.steps:
        if step.parts_list is None:
            continue

        for part in step.parts_list.parts:
            # Count must be inside Part bbox
            if not part.bbox.contains(part.count.bbox):
                validation.add(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        rule="part_count_outside",
                        message="Part count outside Part bbox",
                        pages=[page_data.page_number],
                        details=f"Count {part.count.bbox} not in Part {part.bbox}",
                    )
                )

            # Diagram (if present) must be inside Part bbox
            if part.diagram and not part.bbox.contains(part.diagram.bbox):
                validation.add(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        rule="part_diagram_outside",
                        message="Part diagram outside Part bbox",
                        pages=[page_data.page_number],
                        details=f"Diagram {part.diagram.bbox} not in Part {part.bbox}",
                    )
                )


def validate_elements_within_page(
    validation: ValidationResult,
    page: Page,
    page_data: PageData,
) -> None:
    """Validate all element bboxes stay within page boundaries.

    Domain Invariant: Elements should not extend beyond the page boundaries.
    This would indicate an extraction or classification error.

    Args:
        validation: ValidationResult to add issues to
        page: The classified Page object
        page_data: The raw PageData for context (contains page bbox)
    """
    if page_data.bbox is None:
        validation.add(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                rule="no_page_bbox",
                message="Page has no bounding box defined",
                pages=[page_data.page_number],
            )
        )
        return

    page_bbox = page_data.bbox

    for element in page.iter_elements():
        if not page_bbox.contains(element.bbox):
            validation.add(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    rule="element_outside_page",
                    message=f"{element.__class__.__name__} extends beyond page",
                    pages=[page_data.page_number],
                    details=f"{element.bbox} outside page {page_bbox}",
                )
            )


def validate_content_no_metadata_overlap(
    validation: ValidationResult,
    page: Page,
    page_data: PageData,
) -> None:
    """Validate content elements don't overlap page metadata.

    Domain Invariant: The page number and progress bar are navigation elements
    that should be distinct from actual content (steps, parts, etc.).
    Any overlap indicates a classification error.

    Args:
        validation: ValidationResult to add issues to
        page: The classified Page object
        page_data: The raw PageData for context
    """
    # Define metadata elements
    metadata_elements: list[tuple[str, object]] = []
    if page.page_number:
        metadata_elements.append(("PageNumber", page.page_number))
    if page.progress_bar:
        metadata_elements.append(("ProgressBar", page.progress_bar))

    if not metadata_elements:
        return

    # Collect content elements (top-level only)
    content_elements: list[tuple[str, object]] = []
    for step in page.steps:
        content_elements.append((f"Step {step.step_number.value}", step))
    for bag in page.new_bags:
        content_elements.append(("NewBag", bag))
    for part in page.catalog:
        content_elements.append(("CatalogPart", part))

    # Check for overlaps
    for meta_name, meta_elem in metadata_elements:
        for content_name, content_elem in content_elements:
            # Use intersection area to detect overlap
            intersection = meta_elem.bbox.intersect(content_elem.bbox)  # type: ignore[union-attr]
            if intersection.area > 0:
                validation.add(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        rule="content_metadata_overlap",
                        message=f"{content_name} overlaps with {meta_name}",
                        pages=[page_data.page_number],
                        details=f"{content_elem.bbox} intersects {meta_elem.bbox}",  # type: ignore[union-attr]
                    )
                )


def validate_unassigned_blocks(
    validation: ValidationResult,
    result: ClassificationResult,
) -> None:
    """Validate that all source blocks are assigned to a built candidate or removed.

    Args:
        validation: ValidationResult to add issues to
        result: The ClassificationResult for a page
    """
    if result.skipped_reason:
        return

    unassigned_blocks = []
    for block in result.page_data.blocks:
        # Check if block is assigned to a constructed candidate
        best_candidate = result.get_best_candidate(block)
        if best_candidate:
            continue

        # Check if block was explicitly removed
        if result.is_removed(block):
            continue

        # Block is unassigned and not removed
        unassigned_blocks.append(block)

    if unassigned_blocks:
        block_details = ", ".join(
            f"#{b.id} ({type(b).__name__})" for b in unassigned_blocks[:10]
        )
        if len(unassigned_blocks) > 10:
            block_details += f" ... ({len(unassigned_blocks)} total)"

        validation.add(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule="unassigned_block",
                message=f"{len(unassigned_blocks)} unassigned blocks on page",
                pages=[result.page_data.page_number],
                details=f"Blocks: {block_details}",
            )
        )
