"""Rule-based tests over real fixtures for the PDF element classifier.

This suite validates high-level invariants that must hold after classification.

Rules covered:
- No labeled element should be marked as deleted.
- Each element has at most one winner candidate.

Real fixture(s) live under this package's fixtures/ directory.
"""

import logging

import pytest

from build_a_long.pdf_extract.classifier import classify_elements
from build_a_long.pdf_extract.classifier.classification_result import Candidate
from build_a_long.pdf_extract.extractor import ExtractionResult, PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement
from build_a_long.pdf_extract.fixtures import FIXTURES_DIR, RAW_FIXTURE_FILES

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
    fixture_path = FIXTURES_DIR / fixture_file
    extraction: ExtractionResult = ExtractionResult.model_validate_json(
        fixture_path.read_text()
    )  # type: ignore[assignment]

    if not extraction.pages:
        raise ValueError(f"No pages found in {fixture_file}")

    return extraction.pages


class TestClassifierRules:
    """End-to-end rules that must hold on real pages after classification."""

    @pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
    def test_no_labeled_element_is_deleted(self, fixture_file: str) -> None:
        """No element with a label should be marked as deleted.

        If an element has been classified with a label, it should not be deleted.
        This ensures that the classification and deletion logic don't conflict.
        """
        pages: list[PageData] = _load_pages_from_fixture(fixture_file)

        for page_idx, page in enumerate(pages):
            # Run the full classification pipeline on the page
            result = classify_elements(page)

            # Find all elements that are both labeled and deleted
            # Build a map of source_block -> label for successfully constructed candidates
            block_to_label: dict[int, str] = {}
            for label, candidates in result.get_all_candidates().items():
                for candidate in candidates:
                    if (
                        candidate.constructed is not None
                        and candidate.source_block is not None
                    ):
                        block_to_label[id(candidate.source_block)] = label

            labeled_and_deleted = []
            for elem in page.blocks:
                if id(elem) in block_to_label and result.is_removed(elem):
                    labeled_and_deleted.append((elem, block_to_label[id(elem)]))

            if labeled_and_deleted:
                log.error(
                    f"Found {len(labeled_and_deleted)} labeled elements "
                    f"that are deleted:"
                )
                for elem, label in labeled_and_deleted:
                    log.error(f"  - {label} id:{elem.id} bbox:{elem.bbox} [DELETED]")

            assert len(labeled_and_deleted) == 0, (
                f"Found {len(labeled_and_deleted)} labeled elements that are "
                f"deleted in {fixture_file} page {page_idx}. "
                f"Labeled elements should not be deleted."
            )

    @pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
    def test_each_source_block_maps_to_one_element(self, fixture_file: str) -> None:
        """Each source block should map to at most one LegoPageElement.

        This validates that the classification pipeline doesn't create duplicate
        elements from the same source block. Each raw extraction block should
        produce at most one classified element in the final Page tree.
        """
        pages = _load_pages_from_fixture(fixture_file)

        for page_idx, page_data in enumerate(pages):
            # Run the full classification pipeline on the page
            result = classify_elements(page_data)
            page = result.page

            if page is None:
                continue

            # Get all candidates from the classification result
            all_candidates = result.get_all_candidates()

            # Build a mapping from constructed element ID to candidate
            element_id_to_candidate: dict[int, Candidate] = {}
            for _label, candidates in all_candidates.items():
                for candidate in candidates:
                    if candidate.constructed is not None:
                        elem_id = id(candidate.constructed)
                        assert elem_id not in element_id_to_candidate, (
                            f"Source block id:{id(candidate.source_block)} "
                            f"produced multiple elements of type "
                            f"{candidate.constructed.__class__.__name__} "
                            f"in {fixture_file} page {page_idx}"
                        )
                        element_id_to_candidate[elem_id] = candidate

            blocks_to_element: dict[int, LegoPageElement] = {}

            # Traverse all LegoPageElements in the Page tree
            for element in page.iter_elements():
                elem_id = id(element)

                # Skip synthetic/fallback elements that weren't created by candidates
                # (e.g., empty PartsLists created when Step has no parts_list)
                if elem_id not in element_id_to_candidate:
                    continue

                candidate = element_id_to_candidate[elem_id]

                if candidate.source_block:
                    if candidate.source_block.id in blocks_to_element:
                        existing_element = blocks_to_element[candidate.source_block.id]
                        assert candidate.source_block.id not in blocks_to_element, (
                            f"Source block id:{candidate.source_block.id} "
                            f"({candidate.source_block.tag}) mapped to multiple "
                            f"elements in {fixture_file} page {page_data.page_number}:\n"
                            f"  First:  {existing_element}\n"
                            f"  Second: {element}\n"
                            f"  Source: {candidate.source_block}"
                        )
                    blocks_to_element[candidate.source_block.id] = element
