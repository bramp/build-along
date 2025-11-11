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
            candidate = next(
                c
                for c in result.get_candidates("page_number")
                if c.source_block == labeled_elements[0]
            )
            score = candidate.score
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
            StepNumberClassifier.requires = frozenset(
                {"page_number", "__missing_label__"}
            )
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
