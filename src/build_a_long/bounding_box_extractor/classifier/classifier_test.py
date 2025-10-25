"""Tests for the element classifier."""

from build_a_long.bounding_box_extractor.classifier.classifier import (
    _score_page_number_text,
    _classify_page_number,
    classify_elements,
)
from build_a_long.bounding_box_extractor.extractor import PageData
from build_a_long.bounding_box_extractor.extractor.bbox import BBox
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Root,
    Text,
    Drawing,
)


class TestScorePageNumberText:
    """Tests for the _score_page_number_text function."""

    def test_simple_numbers(self) -> None:
        """Test simple numeric page numbers."""
        assert _score_page_number_text("1") == 1.0
        assert _score_page_number_text("5") == 1.0
        assert _score_page_number_text("42") == 1.0
        assert _score_page_number_text("123") == 1.0

    def test_leading_zeros(self) -> None:
        """Test page numbers with leading zeros."""
        assert _score_page_number_text("01") == 0.95
        assert _score_page_number_text("001") == 0.95
        assert _score_page_number_text("005") == 0.95

    def test_whitespace_handling(self) -> None:
        """Test that whitespace is properly handled."""
        assert _score_page_number_text("  5  ") == 1.0
        assert _score_page_number_text("\t42\n") == 1.0

    def test_non_page_numbers(self) -> None:
        """Test that non-page-number text is rejected."""
        assert _score_page_number_text("hello") == 0.0
        assert _score_page_number_text("Step 3") == 0.0
        assert _score_page_number_text("1234") == 0.0  # Too many digits
        assert _score_page_number_text("12.5") == 0.0  # Decimal
        assert _score_page_number_text("") == 0.0


class TestClassifyPageNumber:
    """Tests for the _classify_page_number function."""

    def test_no_elements(self) -> None:
        """Test classification with no elements."""
        page_data = PageData(
            page_number=1,
            root=Root(bbox=BBox(0, 0, 100, 200)),
            elements=[],
        )
        # First calculate scores (would normally be done by classify_elements)
        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)
        _classify_page_number(page_data)
        # Should not raise any errors

    def test_single_page_number_bottom_left(self) -> None:
        """Test identifying a page number in the bottom-left corner."""
        page_bbox = BBox(0, 0, 100, 200)
        page_number_text = Text(
            bbox=BBox(5, 190, 15, 198),  # Bottom-left position
            text="1",
        )

        page_data = PageData(
            page_number=1,
            root=Root(bbox=page_bbox),
            elements=[page_number_text],
        )

        # Calculate scores first
        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)
        _classify_page_number(page_data)

        assert page_number_text.label == "page_number"
        assert "page_number" in page_number_text.label_scores
        assert page_number_text.label_scores["page_number"] > 0.5

    def test_single_page_number_bottom_right(self) -> None:
        """Test identifying a page number in the bottom-right corner."""
        page_bbox = BBox(0, 0, 100, 200)
        page_number_text = Text(
            bbox=BBox(90, 190, 98, 198),  # Bottom-right position
            text="5",
        )

        page_data = PageData(
            page_number=1,
            root=Root(bbox=page_bbox),
            elements=[page_number_text],
        )

        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)
        _classify_page_number(page_data)

        assert page_number_text.label == "page_number"
        assert "page_number" in page_number_text.label_scores

    def test_multiple_candidates_prefer_corners(self) -> None:
        """Test that corner elements score higher than center ones."""
        page_bbox = BBox(0, 0, 100, 200)

        # Element in center-bottom (less preferred)
        center_text = Text(
            bbox=BBox(45, 190, 55, 198),
            text="2",
        )

        # Element in corner (more preferred)
        corner_text = Text(
            bbox=BBox(5, 190, 15, 198),
            text="3",
        )

        page_data = PageData(
            page_number=1,
            root=Root(bbox=page_bbox),
            elements=[center_text, corner_text],
        )

        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)

        # Corner should have higher score
        assert (
            corner_text.label_scores["page_number"]
            > center_text.label_scores["page_number"]
        )

        _classify_page_number(page_data)
        assert corner_text.label == "page_number"
        assert center_text.label is None

    def test_prefer_numeric_match_to_page_index(self) -> None:
        """Prefer element whose numeric value equals PageData.page_number."""
        page_bbox = BBox(0, 0, 100, 200)
        # Two numbers, both near bottom, but only one matches the page number 7
        txt6 = Text(bbox=BBox(10, 190, 14, 196), text="6")
        txt7 = Text(bbox=BBox(90, 190, 94, 196), text="7")

        page_data = PageData(
            page_number=7, root=Root(bbox=page_bbox), elements=[txt6, txt7]
        )

        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)
        _classify_page_number(page_data)

        assert txt7.label == "page_number"
        assert txt6.label is None

    def test_remove_near_duplicate_bboxes(self) -> None:
        """After choosing page number, remove nearly identical shadow/duplicate elements."""
        page_bbox = BBox(0, 0, 100, 200)
        # Chosen page number
        pn = Text(bbox=BBox(10, 190, 14, 196), text="3")
        # Very similar drawing (e.g., stroke/shadow) almost same bbox
        from build_a_long.bounding_box_extractor.extractor.page_elements import Drawing

        dup = Drawing(bbox=BBox(10.2, 190.1, 14.1, 195.9))

        page_data = PageData(
            page_number=3, root=Root(bbox=page_bbox), elements=[pn, dup]
        )

        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)
        _classify_page_number(page_data)

        # Page number kept and labeled; duplicate removed from flat list
        assert pn.label == "page_number"
        assert pn in page_data.elements
        assert dup not in page_data.elements

    def test_not_in_bottom_region(self) -> None:
        """Test that elements outside bottom region score lower due to position."""
        page_bbox = BBox(0, 0, 100, 200)
        top_text = Text(
            bbox=BBox(5, 10, 15, 18),  # Top of page
            text="1",
        )

        page_data = PageData(
            page_number=1,
            root=Root(bbox=page_bbox),
            elements=[top_text],
        )

        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)

        # Should have score dominated by text (position score is 0.0)
        # Score = 0.7 * 1.0 (text) + 0.3 * 0.0 (position) = 0.7
        assert top_text.label_scores["page_number"] == 0.7

        _classify_page_number(page_data)
        # Still gets labeled since it's the only candidate with score > threshold
        # In real scenarios, there would be other elements with better positions
        assert top_text.label == "page_number"

    def test_non_numeric_text_scores_low(self) -> None:
        """Test that non-numeric text scores low."""
        page_bbox = BBox(0, 0, 100, 200)
        text_element = Text(
            bbox=BBox(5, 190, 50, 198),  # Bottom-left position
            text="Hello World",
        )

        page_data = PageData(
            page_number=1,
            root=Root(bbox=page_bbox),
            elements=[text_element],
        )

        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)

        # Should have low score due to text pattern (position is good but text is bad)
        assert text_element.label_scores["page_number"] < 0.5

        _classify_page_number(page_data)
        assert text_element.label is None


class TestClassifyElements:
    """Tests for the main classify_elements function."""

    def test_classify_multiple_pages(self) -> None:
        """Test classification across multiple pages."""
        pages = []
        for i in range(1, 4):
            page_bbox = BBox(0, 0, 100, 200)
            page_number_text = Text(
                bbox=BBox(5, 190, 15, 198),
                text=str(i),
            )

            page_data = PageData(
                page_number=i,
                root=Root(bbox=page_bbox),
                elements=[page_number_text],
            )
            pages.append(page_data)

        classify_elements(pages)

        # Verify all pages have their page numbers labeled and scored
        for page_data in pages:
            labeled_elements = [
                e
                for e in page_data.elements
                if isinstance(e, Text) and e.label == "page_number"
            ]
            assert len(labeled_elements) == 1
            # Check that scores were calculated
            assert "page_number" in labeled_elements[0].label_scores
            assert labeled_elements[0].label_scores["page_number"] > 0.5

    def test_empty_pages_list(self) -> None:
        """Test with an empty list of pages."""
        classify_elements([])
        # Should not raise any errors


class TestPartCountClassification:
    """Tests for detecting piece counts like '2x'."""

    def test_detect_multiple_piece_counts(self) -> None:
        page_bbox = BBox(0, 0, 100, 200)
        t1 = Text(bbox=BBox(10, 50, 20, 60), text="2x")
        t2 = Text(bbox=BBox(10, 50, 20, 60), text="2X")  # uppercase X
        t3 = Text(bbox=BBox(30, 50, 40, 60), text="2×")  # times symbol
        t4 = Text(bbox=BBox(50, 50, 70, 60), text="hello")

        page = PageData(
            page_number=1, root=Root(bbox=page_bbox), elements=[t1, t2, t3, t4]
        )
        classify_elements([page])

        assert t1.label == "part_count"
        assert t2.label == "part_count"
        assert t3.label == "part_count"
        assert t4.label is None


class TestStepNumberClassification:
    """Tests for step number detection with size heuristic."""

    def test_step_numbers_must_be_taller_than_page_number(self) -> None:
        page_bbox = BBox(0, 0, 200, 300)
        # Page number near bottom, small height (10)
        pn = Text(bbox=BBox(10, 285, 20, 295), text="5")

        # Candidate step numbers elsewhere
        big_step = Text(bbox=BBox(50, 100, 70, 120), text="12")  # height 20
        small_step = Text(bbox=BBox(80, 100, 88, 108), text="3")  # height 8 (too small)

        page = PageData(
            page_number=5,
            root=Root(bbox=page_bbox),
            elements=[pn, big_step, small_step],
        )

        classify_elements([page])

        assert pn.label == "page_number"
        assert big_step.label == "step_number"
        assert small_step.label is None


class TestPartsListClassification:
    """Tests for detecting a parts list drawing above a step containing part counts."""

    def test_parts_list_drawing_above_step(self) -> None:
        page_bbox = BBox(0, 0, 200, 300)

        # Page and step
        pn = Text(bbox=BBox(10, 285, 20, 295), text="6")
        step = Text(
            bbox=BBox(50, 180, 70, 210), text="10"
        )  # height 30 (taller than PN)

        # Two drawings above the step; only d1 contains part counts
        d1 = Drawing(bbox=BBox(30, 100, 170, 160))
        d2 = Drawing(bbox=BBox(20, 40, 180, 80))

        # Part counts inside d1
        pc1 = Text(bbox=BBox(40, 110, 55, 120), text="2x")
        pc2 = Text(bbox=BBox(100, 130, 115, 140), text="5×")

        # Some unrelated text
        other = Text(bbox=BBox(10, 10, 40, 20), text="hello")

        page = PageData(
            page_number=6,
            root=Root(bbox=page_bbox),
            elements=[pn, step, d1, d2, pc1, pc2, other],
        )

        classify_elements([page])

        # Part counts should be labeled, step labeled, and d1 chosen as parts list
        assert pc1.label == "part_count"
        assert pc2.label == "part_count"
        assert step.label == "step_number"
        assert d1.label == "parts_list"
        assert d2.label is None or d2.label != "parts_list"
