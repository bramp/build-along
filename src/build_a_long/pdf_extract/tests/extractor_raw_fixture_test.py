"""Tests to validate extractor output against _raw.json fixtures.

These tests verify that the extractor output matches the committed _raw.json
fixtures. This ensures changes to the extractor don't inadvertently break
extraction of real LEGO instruction PDFs.

The tests depend on PDF files in the data/ directory. These PDFs are:
- Downloaded via the downloader tool
- NOT committed to source control (for copyright reasons)
- Symlinked into the sandbox when running tests

If PDFs are missing, the tests are skipped with a helpful message.

If a test fails, the extractor output has changed. To update fixtures:
    pants run src/build_a_long/pdf_extract/classifier/tools/regenerate_fixtures.py
"""

import bz2
import difflib

import pytest

from build_a_long.pdf_extract.extractor import ExtractionResult
from build_a_long.pdf_extract.fixtures import FIXTURES_DIR
from build_a_long.pdf_extract.tests.fixture_utils import (
    FixtureDefinition,
    extract_pages_from_pdf,
    load_fixture_definitions,
)

# Load fixture definitions once at module level
FIXTURE_DEFINITIONS = load_fixture_definitions()


def _compare_json(
    expected_json: str, actual_json: str, fixture_name: str
) -> str | None:
    """Compare two JSON strings and return diff if different.

    Args:
        expected_json: Expected JSON content
        actual_json: Actual JSON content
        fixture_name: Name for diff output

    Returns:
        Diff string if different, None if identical
    """
    if expected_json == actual_json:
        return None

    diff_lines = list(
        difflib.unified_diff(
            expected_json.splitlines(keepends=True),
            actual_json.splitlines(keepends=True),
            fromfile=f"{fixture_name} (expected)",
            tofile=f"{fixture_name} (actual)",
            lineterm="",
        )
    )

    # Limit diff to first 100 lines
    if len(diff_lines) > 100:
        diff_lines = diff_lines[:100] + ["\n... (diff truncated) ...\n"]

    return "".join(diff_lines)


class TestExtractorRawFixtures:
    """Tests comparing extractor output against _raw.json fixtures.

    Tests are organized by PDF file (from index.json5) to efficiently
    extract all needed pages in a single pass per PDF.
    """

    @pytest.mark.parametrize(
        "fixture_def",
        FIXTURE_DEFINITIONS,
        ids=lambda f: f.element_id,
    )
    def test_extraction_matches_fixtures(self, fixture_def: FixtureDefinition) -> None:
        """Test that extractor output matches all fixtures for a PDF.

        This test:
        1. Opens the PDF once
        2. Extracts all pages defined in the fixture
        3. Compares each against the corresponding fixture file

        If the PDF is not found, the test is skipped.
        """
        # TODO: Re-enable skip if performance regresses
        # # Skip full-document fixtures for now (too slow)
        # if not fixture_def.is_per_page:
        #     pytest.skip("Skipping full-document fixture (too slow)")

        pdf_path = fixture_def.pdf_path

        if not pdf_path.exists():
            pytest.skip(
                f"PDF not found: {pdf_path}. "
                f"Download PDFs to data/ directory to run this test."
            )

        # Extract all needed pages in one pass (also resolves page ranges)
        extraction = extract_pages_from_pdf(pdf_path, fixture_def.pages)
        page_numbers = extraction.page_numbers

        # Collect all failures to report together
        failures: list[str] = []

        if fixture_def.is_per_page:
            # Per-page fixtures: compare each page separately
            for page_num in page_numbers:
                fixture_file = fixture_def.get_fixture_filename(page_num)
                fixture_path = FIXTURES_DIR / fixture_file

                if not fixture_path.exists():
                    failures.append(f"Fixture file not found: {fixture_file}")
                    continue

                page_data = extraction.pages.get(page_num)
                if page_data is None:
                    failures.append(f"Page {page_num} not extracted")
                    continue

                expected = ExtractionResult.model_validate_json(
                    fixture_path.read_text()
                )
                actual = ExtractionResult(pages=[page_data])

                diff = _compare_json(expected.to_json(), actual.to_json(), fixture_file)
                if diff:
                    failures.append(f"Mismatch in {fixture_file}:\n{diff}")
        else:
            # Whole-document fixture: compare all pages together
            fixture_file = fixture_def.get_fixture_filename()
            fixture_path = FIXTURES_DIR / fixture_file

            if not fixture_path.exists():
                failures.append(f"Fixture file not found: {fixture_file}")
            else:
                # Load fixture (may be compressed)
                if fixture_def.compress:
                    json_bytes = bz2.decompress(fixture_path.read_bytes())
                    expected = ExtractionResult.model_validate_json(
                        json_bytes.decode("utf-8")
                    )
                else:
                    expected = ExtractionResult.model_validate_json(
                        fixture_path.read_text()
                    )

                # Build actual result in page order
                pages = [extraction.pages[pn] for pn in sorted(extraction.pages.keys())]
                actual = ExtractionResult(pages=pages)

                diff = _compare_json(expected.to_json(), actual.to_json(), fixture_file)
                if diff:
                    failures.append(f"Mismatch in {fixture_file}:\n{diff}")

        if failures:
            pytest.fail(
                "Extraction output doesn't match fixture(s).\n"
                "To update fixtures, run:\n"
                "    pants run src/build_a_long/pdf_extract/classifier/tools/regenerate_fixtures.py\n\n"
                + "\n\n".join(failures)
            )
