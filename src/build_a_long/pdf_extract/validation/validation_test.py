"""Tests for validation module."""

from build_a_long.pdf_extract.classifier import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.test_utils import TestScore
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Page,
    PageNumber,
    Part,
    PartCount,
    PartsList,
    Step,
    StepNumber,
)

from .printer import print_validation
from .rules import (
    format_ranges,
    validate_elements_within_page,
    validate_first_page_number,
    validate_missing_page_numbers,
    validate_page_number_sequence,
    validate_parts_list_has_parts,
    validate_parts_lists_no_overlap,
    validate_step_sequence,
    validate_steps_have_parts,
    validate_steps_no_significant_overlap,
)
from .runner import validate_results
from .types import ValidationIssue, ValidationResult, ValidationSeverity


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_empty_result(self) -> None:
        """Test empty validation result."""
        result = ValidationResult()
        assert result.error_count == 0
        assert result.warning_count == 0
        assert result.info_count == 0
        assert not result.has_issues()

    def test_add_issue(self) -> None:
        """Test adding issues to result."""
        result = ValidationResult()
        result.add(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                rule="test",
                message="test error",
            )
        )
        result.add(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule="test",
                message="test warning",
            )
        )
        result.add(
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                rule="test",
                message="test info",
            )
        )

        assert result.error_count == 1
        assert result.warning_count == 1
        assert result.info_count == 1
        assert result.has_issues()

    def test_frozen_issue(self) -> None:
        """Test that ValidationIssue is immutable."""
        import pydantic

        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            rule="test",
            message="test",
        )
        # Should not be able to modify (Pydantic frozen model raises ValidationError)
        try:
            issue.message = "new message"  # type: ignore[misc]
            raise AssertionError("Expected frozen model to raise")
        except pydantic.ValidationError:
            pass  # Expected


class TestFormatRanges:
    """Tests for format_ranges helper function."""

    def test_empty_list(self) -> None:
        """Test empty list."""
        assert format_ranges([]) == ""

    def test_single_number(self) -> None:
        """Test single number."""
        assert format_ranges([5]) == "5"

    def test_consecutive_range(self) -> None:
        """Test consecutive numbers form a range."""
        assert format_ranges([1, 2, 3, 4, 5]) == "1-5"

    def test_separate_numbers(self) -> None:
        """Test non-consecutive numbers."""
        assert format_ranges([1, 3, 5]) == "1, 3, 5"

    def test_mixed_ranges(self) -> None:
        """Test mixed ranges and single numbers."""
        assert format_ranges([1, 2, 3, 5, 7, 8, 9]) == "1-3, 5, 7-9"

    def test_long_list_truncation(self) -> None:
        """Test that very long output is truncated."""
        # Create a list that would produce a very long string
        numbers = list(range(1, 200, 2))  # Odd numbers 1-199
        result = format_ranges(numbers)
        assert len(result) <= 100
        assert result.endswith("...")


class TestValidateMissingPageNumbers:
    """Tests for validate_missing_page_numbers rule."""

    def test_no_missing_pages(self) -> None:
        """Test when all pages have page numbers."""
        validation = ValidationResult()
        validate_missing_page_numbers(validation, [], 10)
        assert not validation.has_issues()

    def test_high_coverage(self) -> None:
        """Test >90% coverage produces INFO."""
        validation = ValidationResult()
        validate_missing_page_numbers(validation, [1], 20)  # 95% coverage
        assert validation.info_count == 1
        assert validation.issues[0].severity == ValidationSeverity.INFO

    def test_medium_coverage(self) -> None:
        """Test 50-90% coverage produces WARNING."""
        validation = ValidationResult()
        validate_missing_page_numbers(validation, [1, 2, 3], 10)  # 70% coverage
        assert validation.warning_count == 1

    def test_low_coverage(self) -> None:
        """Test <50% coverage produces ERROR."""
        validation = ValidationResult()
        validate_missing_page_numbers(validation, list(range(1, 8)), 10)  # 30% coverage
        assert validation.error_count == 1


class TestValidateStepSequence:
    """Tests for validate_step_sequence rule."""

    def test_empty_steps(self) -> None:
        """Test empty step list."""
        validation = ValidationResult()
        validate_step_sequence(validation, [])
        assert not validation.has_issues()

    def test_valid_sequence(self) -> None:
        """Test valid step sequence starting at 1."""
        validation = ValidationResult()
        validate_step_sequence(validation, [(1, 1), (2, 2), (3, 3)])
        assert not validation.has_issues()

    def test_duplicate_steps(self) -> None:
        """Test duplicate step numbers."""
        validation = ValidationResult()
        validate_step_sequence(validation, [(1, 1), (2, 1), (3, 2)])  # Step 1 twice
        # Should have warning about duplicates
        assert any(i.rule == "duplicate_steps" for i in validation.issues)

    def test_step_gaps(self) -> None:
        """Test gaps in step sequence."""
        validation = ValidationResult()
        validate_step_sequence(validation, [(1, 1), (2, 3)])  # Missing step 2
        assert any(i.rule == "step_gaps" for i in validation.issues)

    def test_step_not_starting_at_one(self) -> None:
        """Test sequence not starting at 1."""
        validation = ValidationResult()
        validate_step_sequence(validation, [(1, 5), (2, 6), (3, 7)])  # Starts at 5
        assert any(i.rule == "step_start" for i in validation.issues)


class TestValidateFirstPageNumber:
    """Tests for validate_first_page_number rule."""

    def test_no_page_numbers(self) -> None:
        """Test when no page numbers detected."""
        validation = ValidationResult()
        validate_first_page_number(validation, [])
        assert validation.error_count == 1
        assert validation.issues[0].rule == "no_page_numbers"

    def test_reasonable_first_page(self) -> None:
        """Test reasonable first page number."""
        validation = ValidationResult()
        validate_first_page_number(validation, [1, 2, 3])
        assert not validation.has_issues()

    def test_high_first_page(self) -> None:
        """Test high first page number."""
        validation = ValidationResult()
        validate_first_page_number(validation, [15, 16, 17])
        assert any(i.rule == "high_first_page" for i in validation.issues)


class TestValidatePageNumberSequence:
    """Tests for validate_page_number_sequence rule."""

    def test_single_page(self) -> None:
        """Test single page number."""
        validation = ValidationResult()
        validate_page_number_sequence(validation, [1])
        assert not validation.has_issues()

    def test_valid_sequence(self) -> None:
        """Test valid consecutive sequence."""
        validation = ValidationResult()
        validate_page_number_sequence(validation, [1, 2, 3, 4, 5])
        assert not validation.has_issues()

    def test_valid_sequence_starting_later(self) -> None:
        """Test valid consecutive sequence that doesn't start at 1.

        First few pages missing is OK (e.g., cover pages without page numbers).
        """
        validation = ValidationResult()
        validate_page_number_sequence(validation, [5, 6, 7, 8, 9])
        assert not validation.has_issues()

    def test_valid_sequence_ending_early(self) -> None:
        """Test valid consecutive sequence that might end before the last page.

        Last few pages missing is OK (e.g., back cover without page numbers).
        This tests the sequence is consecutive - we don't know total pages here.
        """
        validation = ValidationResult()
        # Sequence 10-14 is consecutive, even if there could be more pages
        validate_page_number_sequence(validation, [10, 11, 12, 13, 14])
        assert not validation.has_issues()

    def test_valid_sequence_starting_later_and_ending_early(self) -> None:
        """Test consecutive sequence with both start and end pages missing.

        Both first N and last M pages can be missing, as long as there are no
        gaps in the middle.
        """
        validation = ValidationResult()
        validate_page_number_sequence(validation, [5, 6, 7, 8, 9, 10])
        assert not validation.has_issues()

    def test_decreasing_sequence(self) -> None:
        """Test decreasing page numbers."""
        validation = ValidationResult()
        validate_page_number_sequence(validation, [1, 2, 5, 3, 4])  # Decreases at 3
        assert any(i.rule == "page_sequence" for i in validation.issues)

    def test_gap_in_middle(self) -> None:
        """Test gap in the middle of page numbers."""
        validation = ValidationResult()
        validate_page_number_sequence(validation, [1, 2, 5, 6])  # Gap: 2->5
        assert any(i.rule == "page_gaps" for i in validation.issues)
        # Should be a warning now
        gap_issue = next(i for i in validation.issues if i.rule == "page_gaps")
        assert gap_issue.severity == ValidationSeverity.WARNING

    def test_small_gap_not_allowed(self) -> None:
        """Test that even small gaps (>1) are flagged."""
        validation = ValidationResult()
        validate_page_number_sequence(validation, [1, 2, 4, 5])  # Gap: 2->4
        assert any(i.rule == "page_gaps" for i in validation.issues)


class TestValidateStepsHaveParts:
    """Tests for validate_steps_have_parts rule."""

    def test_all_steps_have_parts(self) -> None:
        """Test when all steps have parts."""
        validation = ValidationResult()
        validate_steps_have_parts(validation, [])
        assert not validation.has_issues()

    def test_some_steps_missing_parts(self) -> None:
        """Test some steps missing parts."""
        validation = ValidationResult()
        # (page, step_number) tuples
        validate_steps_have_parts(validation, [(1, 1), (3, 5), (5, 10)])
        assert validation.info_count == 1
        issue = validation.issues[0]
        assert issue.rule == "steps_without_parts"
        assert issue.pages == [1, 3, 5]
        assert issue.details is not None
        assert "step 1 (p.1)" in issue.details
        assert "step 5 (p.3)" in issue.details
        assert "step 10 (p.5)" in issue.details


def _make_page_data(page_num: int) -> PageData:
    """Create a minimal PageData for testing."""
    return PageData(
        page_number=page_num,
        bbox=BBox(0, 0, 100, 100),
        blocks=[],
    )


def _make_classification_result(
    page_data: PageData,
    page_number_val: int | None = None,
    step_numbers: list[int] | None = None,
    include_parts: bool = True,
) -> ClassificationResult:
    """Create a ClassificationResult with a Page for testing.

    Args:
        page_data: The PageData to associate
        page_number_val: The LEGO page number value (None for no page number)
        step_numbers: List of step numbers to include
        include_parts: Whether to include parts lists in steps
    """
    result = ClassificationResult(page_data=page_data)

    # Build the Page object
    page_num_elem = (
        PageNumber(bbox=BBox(0, 90, 10, 100), value=page_number_val)
        if page_number_val is not None
        else None
    )

    step_elems: list[Step] = []
    if step_numbers:
        for step_num in step_numbers:
            parts_list = None
            if include_parts:
                parts_list = PartsList(
                    bbox=BBox(0, 0, 20, 10),
                    parts=[
                        Part(
                            bbox=BBox(0, 0, 10, 10),
                            count=PartCount(bbox=BBox(0, 0, 5, 5), count=1),
                        )
                    ],
                )
            step_elems.append(
                Step(
                    bbox=BBox(0, 0, 80, 80),
                    step_number=StepNumber(bbox=BBox(0, 10, 10, 20), value=step_num),
                    parts_list=parts_list,
                )
            )

    page = Page(
        bbox=BBox(0, 0, 100, 100),
        page_number=page_num_elem,
        steps=step_elems,
    )

    # Add a candidate for the page
    candidate = Candidate(
        label="page",
        source_blocks=[],
        bbox=page.bbox,
        score=1.0,
        score_details=TestScore(),
        constructed=page,
    )
    result.add_candidate(candidate)

    return result


class TestValidateResults:
    """Tests for the main validate_results function."""

    def test_perfect_document(self) -> None:
        """Test document with no issues."""
        pages = [_make_page_data(i) for i in range(1, 4)]
        results = [
            _make_classification_result(pages[0], page_number_val=1, step_numbers=[1]),
            _make_classification_result(pages[1], page_number_val=2, step_numbers=[2]),
            _make_classification_result(pages[2], page_number_val=3, step_numbers=[3]),
        ]

        validation = validate_results(pages, results)
        # No errors or warnings expected
        assert validation.error_count == 0
        assert validation.warning_count == 0

    def test_missing_page_numbers(self) -> None:
        """Test detection of missing page numbers."""
        pages = [_make_page_data(i) for i in range(1, 4)]
        results = [
            _make_classification_result(
                pages[0], page_number_val=None, step_numbers=[1]
            ),
            _make_classification_result(pages[1], page_number_val=2, step_numbers=[2]),
            _make_classification_result(
                pages[2], page_number_val=None, step_numbers=[3]
            ),
        ]

        validation = validate_results(pages, results)
        assert any(i.rule == "missing_page_numbers" for i in validation.issues)

    def test_step_sequence_issues(self) -> None:
        """Test detection of step sequence issues."""
        pages = [_make_page_data(i) for i in range(1, 4)]
        results = [
            _make_classification_result(pages[0], page_number_val=1, step_numbers=[1]),
            _make_classification_result(
                pages[1], page_number_val=2, step_numbers=[3]
            ),  # Skipped step 2
            _make_classification_result(pages[2], page_number_val=3, step_numbers=[4]),
        ]

        validation = validate_results(pages, results)
        assert any(i.rule == "step_gaps" for i in validation.issues)


class TestPrintValidation:
    """Tests for print_validation function."""

    def test_print_no_issues(self, capsys: object) -> None:
        """Test printing when no issues."""
        validation = ValidationResult()
        print_validation(validation)
        # Check output contains success message
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "passed" in captured.out

    def test_print_with_issues(self, capsys: object) -> None:
        """Test printing with various issues."""
        validation = ValidationResult()
        validation.add(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                rule="test_error",
                message="Test error message",
                pages=[1, 2, 3],
            )
        )
        validation.add(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                rule="test_warning",
                message="Test warning message",
                details="Some details",
            )
        )

        print_validation(validation, use_color=False)
        captured = capsys.readouterr()  # type: ignore[union-attr]

        assert "test_error" in captured.out
        assert "Test error message" in captured.out
        assert "test_warning" in captured.out
        assert "Some details" in captured.out


# =============================================================================
# Domain Invariant Validation Rules Tests
# =============================================================================


def _make_page_with_steps(
    step_data: list[tuple[int, BBox, BBox | None]],  # (step_num, step_bbox, pl_bbox)
    page_number_val: int = 1,
    page_bbox: BBox | None = None,
) -> tuple[Page, PageData]:
    """Create a Page with steps for testing domain invariants.

    Args:
        step_data: List of (step_number, step_bbox, parts_list_bbox) tuples.
            If parts_list_bbox is None, no parts list is added.
        page_number_val: The page number value
        page_bbox: The page bounding box (default 0,0,100,100)

    Returns:
        Tuple of (Page, PageData)
    """
    if page_bbox is None:
        page_bbox = BBox(0, 0, 100, 100)

    page_data = PageData(
        page_number=1,
        bbox=page_bbox,
        blocks=[],
    )

    steps = []
    for step_num, step_bbox, pl_bbox in step_data:
        parts_list = None
        if pl_bbox is not None:
            # Create a parts list with one part
            part = Part(
                bbox=BBox(pl_bbox.x0, pl_bbox.y0, pl_bbox.x1, pl_bbox.y1 - 5),
                count=PartCount(
                    bbox=BBox(pl_bbox.x0, pl_bbox.y1 - 5, pl_bbox.x1, pl_bbox.y1),
                    count=1,
                ),
            )
            parts_list = PartsList(bbox=pl_bbox, parts=[part])

        step = Step(
            bbox=step_bbox,
            step_number=StepNumber(
                bbox=BBox(
                    step_bbox.x0, step_bbox.y0, step_bbox.x0 + 10, step_bbox.y0 + 10
                ),
                value=step_num,
            ),
            parts_list=parts_list,
        )
        steps.append(step)

    page = Page(
        bbox=page_bbox,
        page_number=PageNumber(bbox=BBox(90, 90, 100, 100), value=page_number_val),
        steps=steps,
    )

    return page, page_data


class TestValidatePartsListHasParts:
    """Tests for validate_parts_list_has_parts rule."""

    def test_no_empty_parts_lists(self) -> None:
        """Test page with all parts lists having parts."""
        page, page_data = _make_page_with_steps(
            [
                (1, BBox(0, 0, 50, 50), BBox(40, 0, 50, 20)),
            ]
        )
        validation = ValidationResult()
        validate_parts_list_has_parts(validation, page, page_data)
        assert not validation.has_issues()

    def test_empty_parts_list(self) -> None:
        """Test detection of empty parts list."""
        page, page_data = _make_page_with_steps(
            [
                (1, BBox(0, 0, 50, 50), BBox(40, 0, 50, 20)),
            ]
        )
        # Manually empty the parts list
        page.steps[0].parts_list.parts = []  # type: ignore[union-attr]

        validation = ValidationResult()
        validate_parts_list_has_parts(validation, page, page_data)
        assert validation.warning_count == 1
        assert validation.issues[0].rule == "empty_parts_list"


class TestValidatePartsListsNoOverlap:
    """Tests for validate_parts_lists_no_overlap rule."""

    def test_non_overlapping_parts_lists(self) -> None:
        """Test page with non-overlapping parts lists."""
        page, page_data = _make_page_with_steps(
            [
                (1, BBox(0, 0, 45, 50), BBox(35, 0, 45, 20)),
                (2, BBox(55, 0, 100, 50), BBox(90, 0, 100, 20)),
            ]
        )
        validation = ValidationResult()
        validate_parts_lists_no_overlap(validation, page, page_data)
        assert not validation.has_issues()

    def test_overlapping_parts_lists(self) -> None:
        """Test detection of overlapping parts lists."""
        page, page_data = _make_page_with_steps(
            [
                (1, BBox(0, 0, 60, 50), BBox(40, 0, 60, 20)),
                (2, BBox(40, 0, 100, 50), BBox(40, 0, 60, 20)),  # Same bbox!
            ]
        )
        validation = ValidationResult()
        validate_parts_lists_no_overlap(validation, page, page_data)
        assert validation.error_count == 1
        assert validation.issues[0].rule == "overlapping_parts_lists"


class TestValidateStepsNoSignificantOverlap:
    """Tests for validate_steps_no_significant_overlap rule."""

    def test_non_overlapping_steps(self) -> None:
        """Test page with non-overlapping steps."""
        page, page_data = _make_page_with_steps(
            [
                (1, BBox(0, 0, 45, 50), None),
                (2, BBox(55, 0, 100, 50), None),
            ]
        )
        validation = ValidationResult()
        validate_steps_no_significant_overlap(validation, page, page_data)
        assert not validation.has_issues()

    def test_significantly_overlapping_steps(self) -> None:
        """Test detection of significantly overlapping steps."""
        page, page_data = _make_page_with_steps(
            [
                (1, BBox(0, 0, 80, 50), None),
                (2, BBox(20, 0, 100, 50), None),  # 60% overlap
            ]
        )
        validation = ValidationResult()
        validate_steps_no_significant_overlap(
            validation, page, page_data, overlap_threshold=0.05
        )
        assert validation.warning_count == 1
        assert validation.issues[0].rule == "overlapping_steps"

    def test_minor_overlap_allowed(self) -> None:
        """Test that minor overlap below threshold is allowed."""
        page, page_data = _make_page_with_steps(
            [
                (1, BBox(0, 0, 51, 50), None),
                (2, BBox(50, 0, 100, 50), None),  # 1px overlap
            ]
        )
        validation = ValidationResult()
        validate_steps_no_significant_overlap(
            validation, page, page_data, overlap_threshold=0.05
        )
        assert not validation.has_issues()


class TestValidateElementsWithinPage:
    """Tests for validate_elements_within_page rule."""

    def test_elements_within_bounds(self) -> None:
        """Test page with all elements within bounds."""
        page, page_data = _make_page_with_steps(
            [
                (1, BBox(10, 10, 90, 90), BBox(70, 10, 90, 30)),
            ]
        )
        validation = ValidationResult()
        validate_elements_within_page(validation, page, page_data)
        assert not validation.has_issues()

    def test_element_outside_bounds(self) -> None:
        """Test detection of element outside page bounds."""
        page, page_data = _make_page_with_steps(
            [
                (1, BBox(10, 10, 110, 90), None),  # Extends past right edge
            ]
        )
        validation = ValidationResult()
        validate_elements_within_page(validation, page, page_data)
        assert validation.error_count >= 1
        assert any(i.rule == "element_outside_page" for i in validation.issues)
