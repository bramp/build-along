"""
OCR utilities for extracting text from images.

Provides an abstraction layer over OCR libraries (currently pytesseract) to allow
easy switching or mocking in the future.

.. warning::
    This module is currently BROKEN and not in use. The tesseract integration
    has encoding issues that cause failures. Left here for potential future fixing.
    See ocr_test.py for the skipped tests.
"""

import logging
import shutil

import pytesseract
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def get_tesseract_path() -> str | None:
    """Find the tesseract executable path."""
    return shutil.which("tesseract")


class OCR:
    """OCR engine for extracting text from image data."""

    def __init__(self) -> None:
        """Initialize OCR engine."""
        # Set the tesseract command explicitly to avoid PATH issues in sandboxes
        tesseract_path = get_tesseract_path()
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def extract_text(self, image: PILImage.Image) -> str | None:
        """Extract text from image using OCR.

        Args:
            image: PIL Image.

        Returns:
            Extracted text string if successful and non-empty, None otherwise.
        """
        if not get_tesseract_path():
            logger.warning("Tesseract not available")
            return None

        try:
            # Use --psm 7 (Treat the image as a single text line)
            # Use --oem 1 (Neural nets LSTM engine only)
            # Whitelist characters for parts counts (numbers and x/X)
            config = r"--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789xX"
            text = pytesseract.image_to_string(image, config=config)

            # Clean up result
            text = text.strip()
            return text if text else None

        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None
