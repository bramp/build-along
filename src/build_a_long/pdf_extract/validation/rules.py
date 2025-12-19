"""Individual validation rules for classification results."""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING

from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Background,
    Divider,
    LegoPageElement,
    Manual,
    Page,
    Part,
    ProgressBar,
)

from .types import ValidationIssue, ValidationResult, ValidationSeverity

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier import ClassificationResult

# =============================================================================
# Programming Error Detection (assertions)
# =============================================================================


def assert_page_elements_tracked(result: ClassificationResult) -> None:
    """Assert that all page elements are tracked via candidates.

    This validation checks that all elements in the Page hierarchy were
    properly constructed via candidates (using result.build()), rather than
    being created directly. Elements created directly bypass the candidate
    tracking system and won't appear in debugging/visualization tools.

    This is an assertion because untracked elements indicate a programming
    error in the classifier code.

    Args:
        result: The classification result to validate

    Raises:
        AssertionError: If any elements are not tracked via candidates.
    """
    page = result.page
    if page is None:
        return  # No page built yet, nothing to validate

    # Build set of all constructed element ids from candidates
    constructed_ids: set[int] = set()
    for _, candidates in result.candidates.items():
        for candidate in candidates:
            if candidate.constructed is not None:
                constructed_ids.add(id(candidate.constructed))

    # Check all elements in the page hierarchy
    untracked: list[str] = []
    for element in page.iter_elements():
        if id(element) not in constructed_ids:
            untracked.append(f"{element.__class__.__name__} at {element.bbox}")

    if untracked:
        untracked_summary = "; ".join(untracked[:5])
        if len(untracked) > 5:
            untracked_summary += f" ... and {len(untracked) - 5} more"
        raise AssertionError(
            f"Page {result.page_data.page_number}: {len(untracked)} elements not "
            f"constructed via candidates (programming error): {untracked_summary}"
        )


# Labels whose constructed elements may not appear on the final Page.
# These are known edge cases (e.g., tiny diagram fragments from clustering).
# TODO But ideally we should fix these so all constructed elements appear on Page.
WARN_ONLY_LABELS = {"diagram"}


def assert_constructed_elements_on_page(result: ClassificationResult) -> None:
    """Assert that all constructed elements appear in the Page hierarchy.

    This validation checks that every element constructed via result.build()
    is actually included in the final Page structure. Elements that are built
    but not added to the Page are orphaned and won't appear in the output.

    This catches programming errors where a classifier builds an element but
    forgets to include it in the Page (e.g., missing field assignment).

    Args:
        result: The classification result to validate

    Raises:
        AssertionError: If constructed elements are orphaned (not on Page).
    """
    log = logging.getLogger(__name__)

    page = result.page
    if page is None:
        return  # No page built yet, nothing to validate

    # Build set of all element ids that are on the Page
    on_page_ids: set[int] = {id(elem) for elem in page.iter_elements()}

    # Check all constructed elements are on the page
    orphaned: list[tuple[str, str]] = []  # (label, description)
    for label, candidates in result.candidates.items():
        for candidate in candidates:
            if candidate.constructed is None:
                continue
            if id(candidate.constructed) not in on_page_ids:
                desc = f"{candidate.constructed.__class__.__name__} at {candidate.bbox}"
                orphaned.append((label, desc))

    if orphaned:
        # Separate warn-only vs error labels
        errors = [
            (label, desc) for label, desc in orphaned if label not in WARN_ONLY_LABELS
        ]
        warnings = [
            (label, desc) for label, desc in orphaned if label in WARN_ONLY_LABELS
        ]

        # Log warnings for known edge cases
        for label, desc in warnings:
            log.warning(
                "Page %s: Orphaned %s element (known edge case): %s",
                result.page_data.page_number,
                label,
                desc,
            )

        # Raise assertion for unexpected orphans
        if errors:
            error_summary = "; ".join(f"{label}: {desc}" for label, desc in errors[:5])
            if len(errors) > 5:
                error_summary += f" ... and {len(errors) - 5} more"
            raise AssertionError(
                f"Page {result.page_data.page_number}: {len(errors)} constructed "
                f"elements not on Page (programming error): {error_summary}"
            )


def assert_element_bbox_matches_source_and_children(
    result: ClassificationResult,
) -> None:
    """Assert that each element's bbox equals the union of source blocks and children.

    This validation checks that the bounding box of each constructed element
    equals the union of:
    1. The bounding boxes of its source blocks (PDF content)
    2. The bounding boxes of all child elements (via iter_elements)

    This ensures that element bboxes accurately represent both the underlying
    PDF content and any attached child elements (e.g., PartImage with Shine).

    Elements with no source blocks (synthetic/composite elements like Page, Step)
    are skipped since their bbox may be computed differently.

    Args:
        result: The classification result to validate

    Raises:
        AssertionError: If any element's bbox doesn't match the expected union.
    """
    mismatches: list[str] = []

    for label, candidates in result.candidates.items():
        for candidate in candidates:
            # Skip candidates without constructed elements
            if candidate.constructed is None:
                continue

            # Skip candidates without source blocks (synthetic/composite elements)
            if not candidate.source_blocks:
                continue

            # Validate this element
            _validate_element_bbox(
                candidate.constructed,
                [block.bbox for block in candidate.source_blocks],
                label,
                mismatches,
            )

    if mismatches:
        mismatch_summary = "; ".join(mismatches[:5])
        if len(mismatches) > 5:
            mismatch_summary += f" ... and {len(mismatches) - 5} more"
        raise AssertionError(
            f"Page {result.page_data.page_number}: {len(mismatches)} elements have "
            f"bbox not matching source blocks + children union: {mismatch_summary}"
        )


def _validate_element_bbox(
    element: LegoPageElement,
    source_bboxes: list[BBox],
    label: str,
    mismatches: list[str],
) -> None:
    """Validate that an element's bbox equals the union of source blocks and children.

    Args:
        element: The element to validate
        source_bboxes: Bboxes of source blocks for this element
        label: The label of this element (for error messages)
        mismatches: List to append mismatch descriptions to
    """
    # Collect all bboxes that should contribute to the element's bbox
    all_bboxes: list[BBox] = list(source_bboxes)

    # Add child element bboxes (iter_elements yields self first, then all descendants)
    child_iter = element.iter_elements()
    next(child_iter)  # Skip self
    for child in child_iter:
        all_bboxes.append(child.bbox)

    # If we have bboxes to check, validate
    if all_bboxes:
        expected_bbox = BBox.union_all(all_bboxes)
        actual_bbox = element.bbox

        # Check if they match (allowing for small floating point differences)
        if not actual_bbox.similar(expected_bbox, tolerance=0.1):
            mismatches.append(
                f"{label}: {element.__class__.__name__} "
                f"bbox={actual_bbox} != expected={expected_bbox}"
            )


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
            message=(
                f"{len(invalid_pages)} page(s) failed to produce valid classification"
            ),
            pages=invalid_pages,
        )
    )


def validate_progress_bar_sequence(
    validation: ValidationResult,
    progress_bars: list[tuple[int, float]],
) -> None:
    """Validate progress bar values form a reasonable sequence.

    Checks for:
    - Monotonicity: Progress should generally increase (or stay same)
    - Consistency: Progress increments should be relatively steady (linear)

    Args:
        validation: ValidationResult to add issues to
        progress_bars: List of (pdf_page, progress_value) tuples, sorted by page
    """
    if len(progress_bars) < 2:
        return

    # Check 1: Monotonicity (non-decreasing)
    decreases: list[tuple[int, float, float]] = []  # (page, prev_val, curr_val)
    for i in range(1, len(progress_bars)):
        curr_page, curr_val = progress_bars[i]
        _, prev_val = progress_bars[i - 1]

        # Allow small tolerance for float imprecision or minor jitter
        if curr_val < prev_val - 0.001:
            decreases.append((curr_page, prev_val, curr_val))

    if decreases:
        details = ", ".join(
            f"p.{p}: {prev:.1%} -> {curr:.1%}" for p, prev, curr in decreases[:5]
        )
        validation.add(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule="progress_bar_decrease",
                message=f"Progress bar value decreases {len(decreases)} time(s)",
                pages=[p for p, _, _ in decreases],
                details=f"Decreases: {details}"
                + (" ..." if len(decreases) > 5 else ""),
            )
        )

    # Check 2: Consistency (steady rate)
    # Only check if we have enough samples to be statistically meaningful
    if len(progress_bars) > 5:
        increments = []
        for i in range(1, len(progress_bars)):
            curr_val = progress_bars[i][1]
            prev_val = progress_bars[i - 1][1]
            diff = curr_val - prev_val
            # Only consider positive progress for rate check
            if diff >= 0:
                increments.append(diff)

        if len(increments) > 2:
            mean_inc = statistics.mean(increments)
            try:
                stdev_inc = statistics.stdev(increments)
                cv = stdev_inc / mean_inc if mean_inc > 0 else 0.0

                # Coefficient of variation > 1.0 indicates very high variance
                # relative to mean. This suggests progress is not steady
                # (e.g. big jumps vs tiny steps). We flag this as INFO since
                # it's not necessarily an error, but worth noting.
                if cv > 1.0:
                    validation.add(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            rule="progress_bar_inconsistent",
                            message="Progress bar increments are highly inconsistent",
                            details=f"Mean increment: {mean_inc:.1%}, "
                            f"StdDev: {stdev_inc:.1%} (CV={cv:.1f})",
                        )
                    )
            except statistics.StatisticsError:
                pass  # Ignore calculation errors for edge cases


def validate_catalog_coverage(
    validation: ValidationResult,
    manual: Manual,
    experimental: bool = True,
) -> None:
    """Validate that parts used in instructions are present in the catalog.

    This uses `xref` matching, falling back to `digest` if `xref` is not available.
    `xref` is unique within a PDF, while `digest` is a globally unique MD5 hash of
    the image.

    Args:
        validation: ValidationResult to add issues to
        manual: The complete Manual object containing all pages
        experimental: Whether to treat this rule as experimental (INFO severity only)
    """
    if not manual.catalog_pages:
        return  # No catalog to check against

    # 1. Collect all parts from instruction pages
    instruction_parts: list[tuple[int, Part]] = []  # (page_num, Part)
    for page in manual.instruction_pages:
        for step in page.steps:
            if step.parts_list:
                for part in step.parts_list.parts:
                    if part.diagram:  # Ensure there's a diagram
                        instruction_parts.append((page.pdf_page_number, part))

    if not instruction_parts:
        return

    # 2. Collect all unique identifiers from catalog parts (xref or digest)
    catalog_identifiers: set[int | bytes] = set()
    for part in manual.catalog_parts:
        if part.diagram:
            if part.diagram.xref is not None:
                catalog_identifiers.add(part.diagram.xref)
            if part.diagram.digest is not None:
                catalog_identifiers.add(part.diagram.digest)

    if not catalog_identifiers:
        return

    # 3. Check coverage
    matched_count = 0
    unmatched_parts: list[
        tuple[int, str]
    ] = []  # (page_num, identifier_type + id_value)

    for page_num, part in instruction_parts:
        is_matched = False
        if part.diagram and (
            (part.diagram.xref is not None and part.diagram.xref in catalog_identifiers)
            or (
                part.diagram.digest is not None
                and part.diagram.digest in catalog_identifiers
            )
        ):
            matched_count += 1
            is_matched = True

        if not is_matched:
            # Prepare identifier for unmatched parts
            identifier = ""
            if part.diagram and part.diagram.xref is not None:
                identifier = f"xref:{part.diagram.xref}"
            elif part.diagram and part.diagram.digest is not None:
                identifier = f"digest:{part.diagram.digest.hex()}"
            else:
                identifier = "unknown_id"
            unmatched_parts.append((page_num, identifier))

    # Report stats
    coverage_pct = matched_count / len(instruction_parts) * 100
    msg_prefix = "[EXPERIMENTAL] " if experimental else ""

    validation.add(
        ValidationIssue(
            severity=ValidationSeverity.INFO,
            rule="catalog_coverage",
            message=(
                f"{msg_prefix}Catalog coverage: "
                f"{matched_count}/{len(instruction_parts)} "
                f"parts matched ({coverage_pct:.1f}%)"
            ),
            details=f"Catalog has {len(catalog_identifiers)} unique image identifiers. "
            f"Instructions use {len(instruction_parts)} part instances.",
        )
    )

    # If coverage is low but non-zero, it suggests image reuse is happening
    # but we're missing some. If coverage is 0%, maybe image reuse isn't used.
    if 0 < coverage_pct < 80 and unmatched_parts:
        # Report a sample of unmatched parts
        unmatched_summary = ", ".join(
            f"p.{p} ({iid})" for p, iid in unmatched_parts[:5]
        )
        if len(unmatched_parts) > 5:
            unmatched_summary += " ..."

        severity = (
            ValidationSeverity.INFO if experimental else ValidationSeverity.WARNING
        )
        validation.add(
            ValidationIssue(
                severity=severity,
                rule="missing_from_catalog",
                message=(
                    f"{msg_prefix}{len(unmatched_parts)} instruction parts "
                    f"not found in catalog"
                ),
                details=f"Unmatched identifiers: {unmatched_summary}",
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

    Note: For steps, we check the core structural components (step_number,
    parts_list) but NOT diagrams, because diagrams are large visual elements
    that may legitimately extend into the page metadata area. Subassemblies
    are also excluded since they may contain large diagrams.

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
    # For steps, check structural components (step_number, parts_list) but not
    # diagrams or subassemblies, as those are large visual elements that may
    # legitimately extend into the metadata area
    content_elements: list[tuple[str, object]] = []
    for step in page.steps:
        # Check step_number (should never overlap metadata)
        content_elements.append(
            (f"Step {step.step_number.value} number", step.step_number)
        )
        # Check parts_list if present (should never overlap metadata)
        if step.parts_list:
            content_elements.append(
                (f"Step {step.step_number.value} parts_list", step.parts_list)
            )
        # Diagrams are intentionally NOT checked - they may extend into metadata area
    for bag in page.open_bags:
        content_elements.append(("OpenBag", bag))
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


def validate_no_divider_intersection(
    validation: ValidationResult,
    page: Page,
    page_data: PageData,
) -> None:
    """Validate that content elements do not intersect with dividers.

    Domain Invariant: Dividers separate content sections. Elements like steps,
    parts, diagrams, etc. should not cross or touch divider lines.
    Background and ProgressBar are exceptions as they span the page.

    Args:
        validation: ValidationResult to add issues to
        page: The classified Page object
        page_data: The raw PageData for context
    """
    if not page.dividers:
        return

    # Elements to exclude from checking
    excluded_types = (Page, Background, ProgressBar, Divider)

    for element in page.iter_elements():
        if isinstance(element, excluded_types):
            continue

        for divider in page.dividers:
            if element.bbox.overlaps(divider.bbox):
                validation.add(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        rule="divider_intersection",
                        message=f"{type(element).__name__} intersects with divider",
                        pages=[page_data.page_number],
                        details=f"{type(element).__name__} {element.bbox} intersects "
                        f"Divider {divider.bbox}",
                    )
                )
