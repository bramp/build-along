"""Tests for the element classifier."""

from build_a_long.pdf_extract.classifier.classifier import (
    classify_elements,
    Classifier,
)
from build_a_long.pdf_extract.classifier.step_number_classifier import (
    StepNumberClassifier,
)
from build_a_long.pdf_extract.classifier.types import ClassifierConfig
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_elements import Text


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
                elements=[page_number_text],
                bbox=page_bbox,
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
