"""Tests for the lego_page_builder module."""

from typing import Dict

from build_a_long.pdf_extract.classifier.lego_page_builder import build_page
from build_a_long.pdf_extract.classifier.types import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Page,
    PageNumber,
    Part,
    PartsList,
    Step,
)
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Image,
    Text,
    Element,
)


def make_candidates(labeled_elements: Dict[Element, str]) -> Dict[str, list[Candidate]]:
    """Helper to convert old _labeled_elements format to new candidates format."""
    candidates: Dict[str, list[Candidate]] = {}
    for element, label in labeled_elements.items():
        if label not in candidates:
            candidates[label] = []
        candidates[label].append(
            Candidate(
                bbox=element.bbox,
                label=label,
                score=1.0,
                score_details={},
                constructed=None,  # Tests don't need constructed elements
                source_element=element,
                is_winner=True,
            )
        )
    return candidates


class TestPageNumberExtraction:
    """Tests for extracting PageNumber elements."""

    def test_extract_single_page_number(self) -> None:
        """Test extracting a single page number."""
        page_bbox = BBox(0, 0, 100, 200)
        page_number_text = Text(bbox=BBox(5, 190, 15, 198), text="5")

        page_data = PageData(
            page_number=5,
            elements=[page_number_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates({page_number_text: "page_number"})
        )

        page = build_page(page_data, result)

        assert isinstance(page, Page)
        assert page.page_number is not None
        assert isinstance(page.page_number, PageNumber)
        assert page.page_number.value == 5
        assert page.page_number.bbox == page_number_text.bbox

    def test_extract_no_page_number(self) -> None:
        """Test when there's no page number."""
        page_bbox = BBox(0, 0, 100, 200)
        some_text = Text(bbox=BBox(5, 10, 15, 18), text="Hello")

        page_data = PageData(
            page_number=1,
            elements=[some_text],
            bbox=page_bbox,
        )

        result = ClassificationResult()

        page = build_page(page_data, result)

        assert page.page_number is None

    def test_extract_multiple_page_numbers_warns(self) -> None:
        """Test that multiple page numbers generates a warning."""
        page_bbox = BBox(0, 0, 100, 200)
        page_number_1 = Text(bbox=BBox(5, 190, 15, 198), text="5")
        page_number_2 = Text(bbox=BBox(85, 190, 95, 198), text="5")

        page_data = PageData(
            page_number=5,
            elements=[page_number_1, page_number_2],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates(
                {
                    page_number_1: "page_number",
                    page_number_2: "page_number",
                }
            )
        )

        page = build_page(page_data, result)

        assert page.page_number is not None
        assert len(page.warnings) > 0
        assert "2 page_number elements" in page.warnings[0]

    def test_invalid_page_number_text(self) -> None:
        """Test handling of non-numeric page number text."""
        page_bbox = BBox(0, 0, 100, 200)
        page_number_text = Text(bbox=BBox(5, 190, 15, 198), text="abc")

        page_data = PageData(
            page_number=1,
            elements=[page_number_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates({page_number_text: "page_number"})
        )

        page = build_page(page_data, result)

        assert page.page_number is None
        assert len(page.warnings) > 0
        assert "Could not parse page number" in page.warnings[0]


class TestStepExtraction:
    """Tests for extracting Step elements."""

    def test_extract_single_step(self) -> None:
        """Test extracting a single step."""
        page_bbox = BBox(0, 0, 100, 200)
        step_number_text = Text(bbox=BBox(10, 10, 20, 20), text="1")

        page_data = PageData(
            page_number=1,
            elements=[step_number_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates({step_number_text: "step_number"})
        )

        page = build_page(page_data, result)

        assert len(page.steps) == 1
        step = page.steps[0]
        assert isinstance(step, Step)
        assert step.step_number.value == 1
        assert step.step_number.bbox == step_number_text.bbox

    def test_extract_multiple_steps(self) -> None:
        """Test extracting multiple steps."""
        page_bbox = BBox(0, 0, 100, 200)
        step_1 = Text(bbox=BBox(10, 10, 20, 20), text="1")
        step_2 = Text(bbox=BBox(10, 50, 20, 60), text="2")

        page_data = PageData(
            page_number=1,
            elements=[step_1, step_2],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates(
                {
                    step_1: "step_number",
                    step_2: "step_number",
                }
            )
        )

        page = build_page(page_data, result)

        assert len(page.steps) == 2
        assert page.steps[0].step_number.value == 1
        assert page.steps[1].step_number.value == 2

    def test_invalid_step_number_text(self) -> None:
        """Test handling of non-numeric step number text."""
        page_bbox = BBox(0, 0, 100, 200)
        step_text = Text(bbox=BBox(10, 10, 20, 20), text="abc")

        page_data = PageData(
            page_number=1,
            elements=[step_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates({step_text: "step_number"})
        )

        page = build_page(page_data, result)

        assert len(page.steps) == 0
        assert len(page.warnings) > 0
        assert "Could not parse step number" in page.warnings[0]


class TestPartsListExtraction:
    """Tests for extracting PartsList elements."""

    def test_extract_empty_parts_list(self) -> None:
        """Test extracting a parts list with no parts."""
        page_bbox = BBox(0, 0, 100, 200)
        parts_list_drawing = Drawing(bbox=BBox(10, 10, 90, 50))

        page_data = PageData(
            page_number=1,
            elements=[parts_list_drawing],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates({parts_list_drawing: "parts_list"})
        )

        page = build_page(page_data, result)

        assert len(page.parts_lists) == 1
        parts_list = page.parts_lists[0]
        assert isinstance(parts_list, PartsList)
        assert len(parts_list.parts) == 0
        assert parts_list.bbox == parts_list_drawing.bbox

    def test_extract_parts_list_with_parts(self) -> None:
        """Test extracting a parts list with parts."""
        page_bbox = BBox(0, 0, 100, 200)
        parts_list_drawing = Drawing(bbox=BBox(10, 10, 90, 100))
        part_count_1 = Text(bbox=BBox(15, 15, 25, 25), text="2x")
        part_image_1 = Image(bbox=BBox(30, 15, 50, 35))
        part_count_2 = Text(bbox=BBox(15, 40, 25, 50), text="1x")
        part_image_2 = Image(bbox=BBox(30, 40, 50, 60))

        page_data = PageData(
            page_number=1,
            elements=[
                parts_list_drawing,
                part_count_1,
                part_image_1,
                part_count_2,
                part_image_2,
            ],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates(
                {
                    parts_list_drawing: "parts_list",
                    part_count_1: "part_count",
                    part_image_1: "part_image",
                    part_count_2: "part_count",
                    part_image_2: "part_image",
                }
            ),
            part_image_pairs=[
                (part_count_1, part_image_1),
                (part_count_2, part_image_2),
            ],
        )

        page = build_page(page_data, result)

        assert len(page.parts_lists) == 1
        parts_list = page.parts_lists[0]
        assert len(parts_list.parts) == 2

        # Check first part
        part_1 = parts_list.parts[0]
        assert isinstance(part_1, Part)
        assert part_1.count.count == 2

        # Check second part
        part_2 = parts_list.parts[1]
        assert isinstance(part_2, Part)
        assert part_2.count.count == 1

    def test_parts_outside_parts_list_not_included(self) -> None:
        """Test that parts outside the parts_list bbox are not included."""
        page_bbox = BBox(0, 0, 100, 200)
        parts_list_drawing = Drawing(bbox=BBox(10, 10, 50, 50))
        part_count_inside = Text(bbox=BBox(15, 15, 25, 25), text="2x")
        part_image_inside = Image(bbox=BBox(30, 15, 45, 35))
        part_count_outside = Text(bbox=BBox(60, 60, 70, 70), text="1x")
        part_image_outside = Image(bbox=BBox(75, 60, 90, 80))

        page_data = PageData(
            page_number=1,
            elements=[
                parts_list_drawing,
                part_count_inside,
                part_image_inside,
                part_count_outside,
                part_image_outside,
            ],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates(
                {
                    parts_list_drawing: "parts_list",
                    part_count_inside: "part_count",
                    part_image_inside: "part_image",
                    part_count_outside: "part_count",
                    part_image_outside: "part_image",
                }
            ),
            part_image_pairs=[
                (part_count_inside, part_image_inside),
                (part_count_outside, part_image_outside),
            ],
        )

        page = build_page(page_data, result)

        assert len(page.parts_lists) == 1
        parts_list = page.parts_lists[0]
        # Only the part inside the parts_list should be included
        assert len(parts_list.parts) == 1
        assert parts_list.parts[0].count.count == 2


class TestPartExtraction:
    """Tests for extracting Part elements."""

    def test_parse_part_count_with_x_suffix(self) -> None:
        """Test parsing part count text with 'x' suffix."""
        page_bbox = BBox(0, 0, 100, 200)
        parts_list = Drawing(bbox=BBox(10, 10, 90, 50))
        part_count = Text(bbox=BBox(15, 15, 25, 25), text="3x")
        part_image = Image(bbox=BBox(30, 15, 50, 35))

        page_data = PageData(
            page_number=1,
            elements=[parts_list, part_count, part_image],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates(
                {
                    parts_list: "parts_list",
                    part_count: "part_count",
                    part_image: "part_image",
                }
            ),
            part_image_pairs=[(part_count, part_image)],
        )

        page = build_page(page_data, result)

        assert len(page.parts_lists) == 1
        assert len(page.parts_lists[0].parts) == 1
        assert page.parts_lists[0].parts[0].count.count == 3

    def test_parse_part_count_without_x_suffix(self) -> None:
        """Test that part count text without 'x' suffix is rejected."""
        page_bbox = BBox(0, 0, 100, 200)
        parts_list = Drawing(bbox=BBox(10, 10, 90, 50))
        part_count = Text(bbox=BBox(15, 15, 25, 25), text="5")  # Missing 'x' suffix
        part_image = Image(bbox=BBox(30, 15, 50, 35))

        page_data = PageData(
            page_number=1,
            elements=[parts_list, part_count, part_image],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates(
                {
                    parts_list: "parts_list",
                    part_count: "part_count",
                    part_image: "part_image",
                }
            ),
            part_image_pairs=[(part_count, part_image)],
        )

        page = build_page(page_data, result)

        # Should fail to parse the part count without 'x' suffix
        assert len(page.parts_lists) == 1
        assert len(page.parts_lists[0].parts) == 0  # Part count parsing failed
        assert len(page.warnings) == 1
        assert "Could not parse part count" in page.warnings[0]

    def test_invalid_part_count_text(self) -> None:
        """Test handling of non-numeric part count text."""
        page_bbox = BBox(0, 0, 100, 200)
        parts_list = Drawing(bbox=BBox(10, 10, 90, 50))
        part_count = Text(bbox=BBox(15, 15, 25, 25), text="abc")
        part_image = Image(bbox=BBox(30, 15, 50, 35))

        page_data = PageData(
            page_number=1,
            elements=[parts_list, part_count, part_image],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates(
                {
                    parts_list: "parts_list",
                    part_count: "part_count",
                    part_image: "part_image",
                }
            ),
            part_image_pairs=[(part_count, part_image)],
        )

        page = build_page(page_data, result)

        assert len(page.parts_lists) == 1
        # Part should not be created due to invalid count
        assert len(page.parts_lists[0].parts) == 0
        assert len(page.warnings) > 0
        assert "Could not parse part count" in page.warnings[0]

    def test_part_bbox_combines_count_and_image(self) -> None:
        """Test that Part bbox is the union of count and image bboxes."""
        page_bbox = BBox(0, 0, 100, 200)
        parts_list = Drawing(bbox=BBox(10, 10, 90, 100))
        part_count = Text(bbox=BBox(15, 20, 25, 30), text="1x")
        part_image = Image(bbox=BBox(30, 15, 50, 35))

        page_data = PageData(
            page_number=1,
            elements=[parts_list, part_count, part_image],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates(
                {
                    parts_list: "parts_list",
                    part_count: "part_count",
                    part_image: "part_image",
                }
            ),
            part_image_pairs=[(part_count, part_image)],
        )

        page = build_page(page_data, result)

        part = page.parts_lists[0].parts[0]
        # BBox should be the union of part_count and part_image
        assert part.bbox.x0 == 15  # min of 15, 30
        assert part.bbox.y0 == 15  # min of 20, 15
        assert part.bbox.x1 == 50  # max of 25, 50
        assert part.bbox.y1 == 35  # max of 30, 35


class TestUnprocessedElements:
    """Tests for tracking unprocessed elements."""

    def test_unprocessed_elements_excluded_removed(self) -> None:
        """Test that removed elements are not in unprocessed."""
        page_bbox = BBox(0, 0, 100, 200)
        removed_text = Text(bbox=BBox(5, 5, 15, 15), text="removed")
        kept_text = Text(bbox=BBox(20, 20, 30, 30), text="kept")

        page_data = PageData(
            page_number=1,
            elements=[removed_text, kept_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates({kept_text: "some_label"}),
            _removal_reasons={id(removed_text): None},  # type: ignore
        )

        page = build_page(page_data, result)

        # removed_text should not be in unprocessed
        assert removed_text not in page.unprocessed_elements
        # kept_text should be in unprocessed (has label but not converted)
        assert kept_text in page.unprocessed_elements

    def test_unprocessed_elements_excluded_unlabeled(self) -> None:
        """Test that unlabeled elements are not in unprocessed."""
        page_bbox = BBox(0, 0, 100, 200)
        unlabeled_text = Text(bbox=BBox(5, 5, 15, 15), text="unlabeled")
        labeled_text = Text(bbox=BBox(20, 20, 30, 30), text="labeled")

        page_data = PageData(
            page_number=1,
            elements=[unlabeled_text, labeled_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates({labeled_text: "some_label"})
        )

        page = build_page(page_data, result)

        # unlabeled_text should not be in unprocessed
        assert unlabeled_text not in page.unprocessed_elements
        # labeled_text should be in unprocessed
        assert labeled_text in page.unprocessed_elements


class TestIntegration:
    """Integration tests combining multiple element types."""

    def test_complete_page_hierarchy(self) -> None:
        """Test building a complete page hierarchy."""
        page_bbox = BBox(0, 0, 200, 300)

        # Page number
        page_num = Text(bbox=BBox(5, 290, 15, 298), text="7")

        # Step 1
        step_1_num = Text(bbox=BBox(10, 10, 20, 20), text="1")

        # Step 2
        step_2_num = Text(bbox=BBox(10, 100, 20, 110), text="2")

        # Parts list
        parts_list = Drawing(bbox=BBox(120, 10, 190, 100))
        part_count_1 = Text(bbox=BBox(125, 15, 135, 25), text="2x")
        part_image_1 = Image(bbox=BBox(140, 15, 160, 35))

        page_data = PageData(
            page_number=7,
            elements=[
                page_num,
                step_1_num,
                step_2_num,
                parts_list,
                part_count_1,
                part_image_1,
            ],
            bbox=page_bbox,
        )

        result = ClassificationResult(
            _candidates=make_candidates(
                {
                    page_num: "page_number",
                    step_1_num: "step_number",
                    step_2_num: "step_number",
                    parts_list: "parts_list",
                    part_count_1: "part_count",
                    part_image_1: "part_image",
                }
            ),
            part_image_pairs=[(part_count_1, part_image_1)],
        )

        page = build_page(page_data, result)

        # Verify page number
        assert page.page_number is not None
        assert page.page_number.value == 7

        # Verify steps
        assert len(page.steps) == 2
        assert page.steps[0].step_number.value == 1
        assert page.steps[1].step_number.value == 2

        # Verify parts list
        assert len(page.parts_lists) == 1
        assert len(page.parts_lists[0].parts) == 1
        assert page.parts_lists[0].parts[0].count.count == 2
