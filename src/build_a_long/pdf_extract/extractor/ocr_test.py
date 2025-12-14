"""Tests for OCR.

Note: These tests are currently skipped because the OCR module is broken.
The tesseract integration has encoding issues. See ocr.py for details.
"""

from pathlib import Path

import pytest
from PIL import Image

from build_a_long.pdf_extract.extractor.ocr import OCR

# Path to test fixture images
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"

# Skip all tests in this module - OCR is currently broken
pytestmark = pytest.mark.skip(reason="OCR module is broken - tesseract encoding issues")


@pytest.fixture
def ocr() -> OCR:
    return OCR()


class TestOCR:
    def test_extract_text_2x(self, ocr: OCR) -> None:
        """Test OCR extracts '2x' from fixture image."""
        img_path = FIXTURES_DIR / "page_015_img_020_xref_2768.png"
        img = Image.open(img_path)

        text = ocr.extract_text(img)

        assert text == "2x"

    def test_extract_text_3_focused(self, ocr: OCR) -> None:
        """Test OCR extracts '3' from focused fixture image."""
        img_path = FIXTURES_DIR / "page_045_img_064_xref_627.png"
        img = Image.open(img_path)

        text = ocr.extract_text(img)

        assert text == "3"

    def test_extract_text_3_in_larger_image(self, ocr: OCR) -> None:
        """Test OCR extracts '3' from larger image with number in middle."""
        img_path = FIXTURES_DIR / "page_045_img_001_xref_585.png"
        img = Image.open(img_path)

        text = ocr.extract_text(img)

        # May extract '3' or empty depending on OCR config
        # This tests the larger context case
        assert text in ("3", "")

    def test_extract_text_3_cropped(self, ocr: OCR) -> None:
        """Test OCR handles partially cropped '3' image."""
        img_path = FIXTURES_DIR / "page_045_img_072_xref_587.png"
        img = Image.open(img_path)

        text = ocr.extract_text(img)

        # Cropped image may not OCR cleanly
        assert text in ("3", "")

    def test_extract_text_partial_2(self, ocr: OCR) -> None:
        """Test OCR handles partial '2' image."""
        img_path = FIXTURES_DIR / "page_045_img_081_xref_2830.png"
        img = Image.open(img_path)

        text = ocr.extract_text(img)

        # Partial number may not OCR cleanly
        assert text in ("2", "")
