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
from build_a_long.pdf_extract.extractor import ExtractionResult, PageData
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
