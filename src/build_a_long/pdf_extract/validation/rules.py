"""Individual validation rules for classification results."""

from .types import ValidationIssue, ValidationResult, ValidationSeverity


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
