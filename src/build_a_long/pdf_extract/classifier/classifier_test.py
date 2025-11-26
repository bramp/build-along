"""Tests for the element classifier."""

from build_a_long.pdf_extract.classifier import (
    Classifier,
    ClassifierConfig,
    StepNumberClassifier,
    classify_pages,
)
from build_a_long.pdf_extract.classifier.page_classifier import PageClassifier
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

        # Verify all pages have their page numbers identified
        assert len(batch_result.results) == 3
        for i, (_page_data, result) in enumerate(
            zip(pages, batch_result.results, strict=True), start=1
        ):
            # Check the final Page structure
            page = result.page
            assert page is not None

            # Verify page number was correctly identified
            assert page.page_number is not None
            assert page.page_number.value == i

    def test_empty_pages_list(self) -> None:
        """Test with an empty list of pages."""
        batch_result = classify_pages([])
        assert len(batch_result.results) == 0
        assert batch_result.histogram is not None
        # Should not raise any errors

    def test_duplicate_blocks_marked_as_removed(self) -> None:
        """Test that duplicate blocks are marked with removal reasons.

        This ensures duplicate filtering is tracked properly via removal_reasons
        rather than physically removing blocks from PageData.
        """
        # Create a page with duplicate blocks
        page_bbox = BBox(0, 0, 100, 200)
        # Two identical blocks (duplicates) - one should be marked as removed
        block1 = Text(id=0, bbox=BBox(10, 10, 50, 30), text="1")
        block2 = Text(id=1, bbox=BBox(10, 10, 50, 30), text="1")
        # A unique block
        block3 = Text(id=2, bbox=BBox(5, 190, 15, 198), text="10")

        original_page = PageData(
            page_number=10, blocks=[block1, block2, block3], bbox=page_bbox
        )

        batch_result = classify_pages([original_page])

        # Should have one result
        assert len(batch_result.results) == 1
        result = batch_result.results[0]

        # Original page should be unchanged (all blocks present)
        assert result.page_data is original_page
        assert len(result.page_data.blocks) == 3

        # One duplicate should be marked as removed
        removed_blocks = [b for b in result.page_data.blocks if result.is_removed(b)]
        assert len(removed_blocks) == 1
        removed_block = removed_blocks[0]

        # Check removal reason
        reason = result.get_removal_reason(removed_block)
        assert reason is not None
        assert reason.reason_type == "duplicate_bbox"
        assert reason.target_block in [block1, block2]

        # The kept block should not be removed
        kept_blocks = [b for b in result.page_data.blocks if not result.is_removed(b)]
        assert len(kept_blocks) == 2
        assert block3 in kept_blocks

        # Classification result should only reference non-removed blocks
        all_candidates = result.get_all_candidates()

        # Get all blocks referenced in candidates
        blocks_in_candidates = set()
        for candidates_list in all_candidates.values():
            for candidate in candidates_list:
                if candidate.source_blocks:
                    for block in candidate.source_blocks:
                        blocks_in_candidates.add(id(block))

        # All blocks in candidates should not be removed
        for block in result.page_data.blocks:
            if id(block) in blocks_in_candidates:
                assert not result.is_removed(block), (
                    "Candidate references a removed block"
                )

        # The removed duplicate should not appear in candidates
        assert id(removed_block) not in blocks_in_candidates, (
            "Removed duplicate block should not be referenced in candidates"
        )


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
                # The new dependency validation detects missing labels
                assert "dependencies cannot be satisfied" in str(
                    e
                ) or "Circular dependency" in str(e)
                assert "__missing_label__" in str(e)
            assert raised, "Expected ValueError due to unmet classifier requirements"
        finally:
            # Restore original declaration to avoid impacting other tests
            StepNumberClassifier.requires = original_requires

    def test_dependency_ordered_execution(self) -> None:
        """Test that classifiers are executed in dependency order."""
        classifier = Classifier(ClassifierConfig())

        # Get the batches
        batches = classifier._order_classifiers_by_dependencies()

        # Verify batches are ordered correctly
        assert len(batches) > 0, "Should have at least one batch"

        # Track what labels have been produced so far
        produced: set[str] = set()

        for batch_idx, batch in enumerate(batches):
            # All classifiers in this batch should have their dependencies met
            for cls in batch:
                requires = getattr(cls, "requires", frozenset())
                assert requires.issubset(produced), (
                    f"Batch {batch_idx}: {cls.__class__.__name__} requires "
                    f"{requires - produced} which haven't been produced yet"
                )

            # After this batch, add all outputs to produced
            for cls in batch:
                outputs = getattr(cls, "outputs", frozenset())
                produced |= outputs

    def test_first_batch_has_no_dependencies(self) -> None:
        """Test that classifiers in the first batch have no dependencies."""
        classifier = Classifier(ClassifierConfig())
        batches = classifier._order_classifiers_by_dependencies()

        assert len(batches) > 0, "Should have at least one batch"

        # First batch should contain only classifiers with no dependencies
        for cls in batches[0]:
            requires = getattr(cls, "requires", frozenset())
            assert len(requires) == 0, (
                f"{cls.__class__.__name__} in first batch should have no "
                f"dependencies, but requires {requires}"
            )

    def test_page_classifier_in_last_batch(self) -> None:
        """Test that PageClassifier is in the last batch (depends on most labels)."""
        classifier = Classifier(ClassifierConfig())
        batches = classifier._order_classifiers_by_dependencies()

        assert len(batches) > 0, "Should have at least one batch"

        # PageClassifier should be in the last batch
        last_batch = batches[-1]
        page_classifier_found = any(
            isinstance(cls, PageClassifier) for cls in last_batch
        )
        assert page_classifier_found, (
            "PageClassifier should be in the last batch since it depends on many labels"
        )
