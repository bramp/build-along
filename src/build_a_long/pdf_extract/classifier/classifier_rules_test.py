"""Rule-based tests over real fixtures for the PDF element classifier.

This suite validates high-level invariants that must hold after classification.

Rules covered:
- Every parts list must contain at least one part image inside it.
- No two parts lists overlap.
- Each part image is inside a parts list.
- Each element has at most one winner candidate.

Real fixture(s) live under this package's fixtures/ directory.
"""

import logging
from collections import defaultdict
from pathlib import Path

import pytest

from build_a_long.pdf_extract.classifier import ClassificationResult, classify_elements
from build_a_long.pdf_extract.extractor import ExtractionResult, PageData
from build_a_long.pdf_extract.extractor.page_blocks import Block
from build_a_long.pdf_extract.fixtures import RAW_FIXTURE_FILES

log = logging.getLogger(__name__)


def _load_pages_from_fixture(fixture_file: str) -> list[PageData]:
    """Load all pages from a fixture file.

    Args:
        fixture_file: Name of the fixture file (e.g., '6509377_page_010_raw.json')

    Returns:
        All pages from the extraction result

    Raises:
        ValueError: If the fixture contains no pages
    """
    fixture_path = Path(__file__).parent.parent / "fixtures" / fixture_file
    extraction: ExtractionResult = ExtractionResult.model_validate_json(
        fixture_path.read_text()
    )  # type: ignore[assignment]

    if not extraction.pages:
        raise ValueError(f"No pages found in {fixture_file}")

    return extraction.pages


# TODO A lot of the methods in ClassifiedPage overlap with ClassificationResult


class ClassifiedPage:
    """Wrapper around PageData providing convenient access to classified elements.

    This class provides helper methods to query elements by label type and
    supports hierarchical queries (e.g., finding children inside parent bboxes).
    Results are cached for efficiency.
    """

    def __init__(self, page: PageData, result: ClassificationResult):
        """Initialize with a classified PageData and its result.

        Args:
            page: PageData that has been run through classify_elements()
            result: The ClassificationResult for this page
        """
        self.page = page
        self.result = result
        self._cache: dict[str, list[Block]] = {}

    def elements_by_label(
        self, label: str, include_deleted: bool = False
    ) -> list[Block]:
        """Get all elements with the given label.

        Args:
            label: The label to filter by
            include_deleted: Whether to include deleted elements

        Returns:
            List of elements with matching label
        """
        cache_key = f"{label}:deleted={include_deleted}"
        if cache_key not in self._cache:
            if include_deleted:
                self._cache[cache_key] = [
                    e for e in self.page.blocks if self.result.get_label(e) == label
                ]
            else:
                self._cache[cache_key] = [
                    e
                    for e in self.page.blocks
                    if self.result.get_label(e) == label
                    and not self.result.is_removed(e)
                ]
        return self._cache[cache_key]

    def parts_lists(self) -> list[Block]:
        """Get all non-deleted parts_list elements."""
        return self.elements_by_label("parts_list")

    def part_images(self) -> list[Block]:
        """Get all non-deleted part_image elements."""
        return self.elements_by_label("part_image")

    def part_counts(self) -> list[Block]:
        """Get all non-deleted part_count elements."""
        return self.elements_by_label("part_count")

    def step_numbers(self) -> list[Block]:
        """Get all non-deleted step_number elements."""
        return self.elements_by_label("step_number")

    def children_of(self, parent: Block, label: str | None = None) -> list[Block]:
        """Return all non-deleted elements spatially contained within a parent element.

        Note: This uses bbox containment, not ElementTree hierarchy, because
        the hierarchy is based on "smallest containing bbox" which means there
        may be intermediate unlabeled elements between a parent and its
        logical children. For validation rules about spatial containment,
        bbox checking is more appropriate.

        Args:
            parent: The parent element to search within
            label: Optional label filter (e.g., "part_image")

        Returns:
            List of non-deleted Elements matching the label (if specified) that
            are fully contained within the parent's bbox
        """
        # Use spatial containment, not hierarchy
        result = []
        for elem in self.page.blocks:
            if id(elem) in self.result.removal_reasons:
                continue
            if label is not None and self.result.get_label(elem) != label:
                continue
            if elem.bbox.fully_inside(parent.bbox):
                result.append(elem)
        return result

    def print_summary(self, logger: logging.Logger | None = None) -> None:
        """Log a summary of labeled elements.

        Args:
            logger: Logger to use (defaults to module logger)
        """
        logger = logger or log
        label_counts = defaultdict(int)
        for e in self.page.blocks:
            label = (
                self.result.get_label(e) if self.result.get_label(e) else "<unknown>"
            )
            label_counts[label] += 1

        logger.info(f"Label counts: {dict(label_counts)}")


# TODO Replace this with just results.get_blocks_by_label()


def _parts_lists(page: PageData, result: ClassificationResult) -> list[Block]:
    return [
        e
        for e in page.blocks
        if result.get_label(e) == "parts_list" and not result.is_removed(e)
    ]


# TODO Replace this with just results.get_blocks_by_label()


def _part_images(page: PageData, result: ClassificationResult) -> list[Block]:
    return [
        e
        for e in page.blocks
        if result.get_label(e) == "part_image" and not result.is_removed(e)
    ]


# TODO Replace this with just results.get_blocks_by_label()


def _part_counts(page: PageData, result: ClassificationResult) -> list[Block]:
    return [
        e
        for e in page.blocks
        if result.get_label(e) == "part_count" and not result.is_removed(e)
    ]


def _print_label_counts(page: PageData, result: ClassificationResult) -> None:
    label_counts = defaultdict(int)
    for e in page.blocks:
        label = result.get_label(e) if result.get_label(e) else "<unknown>"
        label_counts[label] += 1

    # TODO The following logging shows "defaultdict(<class 'int'>,..." figure
    # out how to avoid that.
    log.info(f"Label counts: {label_counts}")


class TestClassifierRules:
    """End-to-end rules that must hold on real pages after classification."""

    @pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
    def test_no_labeled_element_is_deleted(self, fixture_file: str) -> None:
        """No element with a label should be marked as deleted.

        If an element has been classified with a label, it should not be deleted.
        This ensures that the classification and deletion logic don't conflict.
        """
        pages = _load_pages_from_fixture(fixture_file)

        for page_idx, page in enumerate(pages):
            # Run the full classification pipeline on the page
            result = classify_elements(page)

            # Find all elements that are both labeled and deleted
            labeled_and_deleted = []
            for elem in page.blocks:
                if result.get_label(elem) is not None and result.is_removed(elem):
                    labeled_and_deleted.append(elem)

            if labeled_and_deleted:
                log.error(
                    f"Found {len(labeled_and_deleted)} labeled elements "
                    f"that are deleted:"
                )
                for elem in labeled_and_deleted:
                    log.error(
                        f"  - {result.get_label(elem)} id:{elem.id} "
                        f"bbox:{elem.bbox} [DELETED]"
                    )

            assert len(labeled_and_deleted) == 0, (
                f"Found {len(labeled_and_deleted)} labeled elements that are "
                f"deleted in {fixture_file} page {page_idx}. "
                f"Labeled elements should not be deleted."
            )

    @pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
    def test_each_element_has_at_most_one_winner(self, fixture_file: str) -> None:
        """Each element should have at most one winner candidate across all labels.

        An element can have multiple candidates across different labels, but only
        one of them should be marked as a winner. This ensures classification
        decisions are unambiguous.
        """
        pages = _load_pages_from_fixture(fixture_file)

        for page_idx, page in enumerate(pages):
            # Run the full classification pipeline on the page
            result = classify_elements(page)

            # Track which blocks have won, and for which label
            block_to_winning_label: dict[int, str] = {}

            # Check all candidates across all labels
            all_candidates = result.get_all_candidates()
            for label, candidates in all_candidates.items():
                for candidate in candidates:
                    if not candidate.is_winner:
                        continue

                    # Skip synthetic candidates (no source block)
                    if candidate.source_block is None:
                        continue

                    block_id = candidate.source_block.id

                    # Check if this block already has a winner
                    if block_id in block_to_winning_label:
                        existing_label = block_to_winning_label[block_id]
                        pytest.fail(
                            f"Block {block_id} in {fixture_file} page {page_idx} "
                            f"has multiple winner candidates: '{existing_label}' "
                            f"and '{label}'. Each block should have at most one winner."
                        )

                    block_to_winning_label[block_id] = label


# TODO Add test to ensure nothing overlaps the page number or progress bar
# TODO Ensure bbox stay within the page boundaries
