"""Tests for the element classifier."""

from build_a_long.pdf_extract.classifier import (
    Classifier,
    ClassifierConfig,
    StepNumberClassifier,
    classify_pages,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Text


class TestClassifyElements:
    """Tests for the main classify_elements function."""

    def test_classify_multiple_pages(self) -> None:
        """Test classification across multiple pages."""
        pages = []
        for i in range(1, 4):
            page_bbox = BBox(0, 0, 100, 200)
            page_number_text = Text(
                id=0,
                bbox=BBox(5, 190, 15, 198),
                text=str(i),
            )

            page_data = PageData(
                page_number=i,
                blocks=[page_number_text],
                bbox=page_bbox,
            )
            pages.append(page_data)

        batch_result = classify_pages(pages)

        # Verify histogram was built
        assert batch_result.histogram is not None

        # Verify all pages have their page numbers labeled
        assert len(batch_result.results) == 3
        for _i, (page_data, result) in enumerate(
            zip(pages, batch_result.results, strict=True)
        ):
            labeled_elements = [
                e
                for e in page_data.blocks
                if isinstance(e, Text) and result.get_label(e) == "page_number"
            ]
            assert len(labeled_elements) == 1
            # Check that scores were calculated
            assert result.has_label("page_number")
            page_number_scores = result.get_scores_for_label("page_number")
            assert labeled_elements[0] in page_number_scores
            score = page_number_scores[labeled_elements[0]].combined_score(
                ClassifierConfig()
            )
            assert score > 0.5

    def test_empty_pages_list(self) -> None:
        """Test with an empty list of pages."""
        batch_result = classify_pages([])
        assert len(batch_result.results) == 0
        assert batch_result.histogram is not None
        # Should not raise any errors


class TestPipelineEnforcement:
    """Tests to ensure classifier pipeline dependencies are enforced at init time."""

    def test_dependency_violation_raises(self) -> None:
        original_requires = StepNumberClassifier.requires
        try:
            # Inject an impossible requirement to trigger the enforcement failure.
            StepNumberClassifier.requires = {"page_number", "__missing_label__"}
            raised = False
            try:
                _ = Classifier(ClassifierConfig())
            except ValueError as e:  # expected
                raised = True
                assert "requires labels not yet produced" in str(e)
                assert "__missing_label__" in str(e)
            assert raised, "Expected ValueError due to unmet classifier requirements"
        finally:
            # Restore original declaration to avoid impacting other tests
            StepNumberClassifier.requires = original_requires


class TestBlockFiltering:
    """Tests for block filtering integration with classify_pages."""

    def test_classify_pages_filters_duplicate_blocks(self) -> None:
        """Test that classify_pages applies duplicate block filtering."""
        # Create a page with duplicate blocks
        page_bbox = BBox(0, 0, 100, 200)
        blocks: list[Text] = [
            # Page number with shadow
            Text(id=1, bbox=BBox(5, 190, 15, 198), text="1"),
            Text(id=2, bbox=BBox(5.5, 190.5, 15.5, 198.5), text="1"),
            # Another text element (no duplicate)
            Text(id=3, bbox=BBox(50, 100, 80, 110), text="Step 1"),
        ]

        page_data = PageData(
            page_number=1,
            blocks=list(blocks),  # Convert to list[Block]
            bbox=page_bbox,
        )

        batch_result = classify_pages([page_data])

        # After filtering, should have fewer blocks than original
        result = batch_result.results[0]
        labeled_blocks = result.get_labeled_blocks()
        # Should have filtered out one of the duplicate page numbers
        assert len(labeled_blocks) <= 2
