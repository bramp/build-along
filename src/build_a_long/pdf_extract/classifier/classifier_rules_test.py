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
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    LegoPageElement,
    PartsList,
)
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
            # Build a map of source_block -> label for successfully constructed
            # candidates
            block_to_label: dict[int, str] = {}
            for label, candidates in result.get_all_candidates().items():
                for candidate in candidates:
                    if candidate.constructed is not None and candidate.source_blocks:
                        for block in candidate.source_blocks:
                            block_to_label[id(block)] = label

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
                            f"Source block id:"
                            f"{id(candidate.source_blocks[0]) if candidate.source_blocks else 'None'} "
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

                for source_block in candidate.source_blocks:
                    if source_block.id in blocks_to_element:
                        existing_element = blocks_to_element[source_block.id]
                        assert source_block.id not in blocks_to_element, (
                            f"Source block id:{source_block.id} "
                            f"({source_block.tag}) mapped to multiple "
                            f"elements in {fixture_file} page "
                            f"{page_data.page_number}:\n"
                            f"  First:  {existing_element}\n"
                            f"  Second: {element}\n"
                            f"  Source: {source_block}"
                        )
                    blocks_to_element[source_block.id] = element

    @pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
    def test_all_lego_elements_come_from_candidates(self, fixture_file: str) -> None:
        """All LegoPageElements in the final Page tree must come from candidates.

        This validates that classifiers don't create "orphan" elements directly
        without a corresponding candidate. Every LegoPageElement should be either:
        1. The constructed element of a candidate, or
        2. A synthetic/fallback element (e.g., empty PartsList when Step has no
           parts_list candidate)

        Ensures proper tracking of all elements through the classification pipeline.
        """
        pages = _load_pages_from_fixture(fixture_file)

        for page_idx, page_data in enumerate(pages):
            # Run the full classification pipeline on the page
            result = classify_elements(page_data)
            page = result.page

            if page is None:
                continue

            # Build a set of all constructed element IDs from candidates
            all_candidates = result.get_all_candidates()
            constructed_element_ids: set[int] = set()
            for _label, candidates in all_candidates.items():
                for candidate in candidates:
                    if candidate.constructed is not None:
                        constructed_element_ids.add(id(candidate.constructed))

            # Traverse all LegoPageElements in the Page tree
            orphan_elements: list[tuple[LegoPageElement, str]] = []
            for element in page.iter_elements():
                elem_id = id(element)
                elem_type = element.__class__.__name__

                # Skip the Page itself (it's the root container)
                if elem_type == "Page":
                    continue

                # Check if this element came from a candidate
                if elem_id not in constructed_element_ids:
                    # TODO Remove the following lines
                    # Known synthetic/fallback elements that are expected:
                    # - Empty PartsList when Step has no parts_list candidate
                    # - Diagram when Step couldn't find a matching diagram candidate
                    if isinstance(element, PartsList) and len(element.parts) == 0:
                        continue
                    if isinstance(element, Diagram):
                        # Fallback diagrams are allowed when StepClassifier
                        # can't find a matching diagram candidate
                        continue

                    orphan_elements.append((element, elem_type))

            if orphan_elements:
                log.error(
                    f"Found {len(orphan_elements)} orphan elements not from "
                    f"candidates in {fixture_file} page {page_idx}:"
                )
                for elem, elem_type in orphan_elements:
                    log.error(f"  - {elem_type} bbox:{elem.bbox}")

            assert len(orphan_elements) == 0, (
                f"Found {len(orphan_elements)} orphan LegoPageElements not from "
                f"candidates in {fixture_file} page {page_idx}. "
                f"All elements should come from candidates or be known fallbacks."
            )

    @pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
    def test_no_orphaned_constructed_candidates(self, fixture_file: str) -> None:
        """No candidate marked constructed without being in the final tree.

        This validates the transactional rollback semantics of build():
        - If a parent classifier's build() fails, all sub-candidates it built
          should be rolled back (constructed = None)
        - Only candidates that are actually used in the final Page tree should
          remain marked as constructed

        This catches bugs where:
        1. A classifier builds sub-candidates (e.g., step builds step_number)
        2. The classifier then fails (e.g., parts_list build fails)
        3. The step_number candidate remains orphaned with constructed set,
           but not actually used in the final tree
        """
        pages = _load_pages_from_fixture(fixture_file)

        for page_idx, page_data in enumerate(pages):
            # Run the full classification pipeline on the page
            result = classify_elements(page_data)
            page = result.page

            if page is None:
                continue

            # Build set of all element IDs actually used in the final Page tree
            used_element_ids: set[int] = set()
            for element in page.iter_elements():
                used_element_ids.add(id(element))

            # Check all candidates for orphaned constructed elements
            all_candidates = result.get_all_candidates()
            orphaned_candidates: list[tuple[str, Candidate]] = []

            for label, candidates in all_candidates.items():
                for candidate in candidates:
                    # If candidate is marked as constructed but not in the tree
                    if (
                        candidate.constructed is not None
                        and id(candidate.constructed) not in used_element_ids
                    ):
                        orphaned_candidates.append((label, candidate))

            if orphaned_candidates:
                log.error(
                    f"Found {len(orphaned_candidates)} orphaned constructed "
                    f"candidates in {fixture_file} page {page_idx}:"
                )
                for label, candidate in orphaned_candidates:
                    elem_type = candidate.constructed.__class__.__name__
                    log.error(
                        f"  - {label}: {elem_type} bbox:{candidate.bbox} "
                        f"score:{candidate.score:.3f} "
                        f"failure:{candidate.failure_reason}"
                    )

            assert len(orphaned_candidates) == 0, (
                f"Found {len(orphaned_candidates)} orphaned constructed candidates "
                f"in {fixture_file} page {page_idx}. "
                f"Candidates marked as constructed should either be in the final "
                f"Page tree or rolled back to constructed=None. "
                f"This indicates a transactional rollback failure."
            )
