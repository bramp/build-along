#!/usr/bin/env python3
"""Extract images from specific PDF pages for OCR test fixtures.

This script extracts images from specified PDF pages at high resolution
for use as OCR test fixtures.

Usage:
    pants run src/build_a_long/pdf_extract/classifier/tools/generate_ocr_fixtures.py
"""

import argparse
import logging
import sys
from pathlib import Path

import pymupdf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Default output directory for OCR fixtures
DEFAULT_OUTPUT_DIR = Path("src/build_a_long/pdf_extract/fixtures/images")

# Scale factor for rendering (4x gives good OCR quality)
DEFAULT_SCALE = 4.0


def extract_images_from_page(
    doc: pymupdf.Document,
    page_num: int,
    output_dir: Path,
    scale: float = DEFAULT_SCALE,
    min_size: int = 10,
) -> list[Path]:
    """Extract all images from a PDF page.

    Args:
        doc: The PDF document
        page_num: Page number (1-indexed)
        output_dir: Directory to save extracted images
        scale: Scale factor for rendering
        min_size: Minimum width/height to include an image

    Returns:
        List of paths to saved images
    """
    page = doc[page_num - 1]  # 0-indexed

    # Get all images on the page
    image_infos = page.get_image_info(xrefs=True)

    # Render the full page at high resolution
    mat = pymupdf.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
    page_img = pix.pil_image()

    log.info(f"Page {page_num}: Found {len(image_infos)} images")

    saved_paths: list[Path] = []

    for i, img_info in enumerate(image_infos):
        bbox = img_info.get("bbox", (0, 0, 0, 0))
        xref = img_info.get("xref", 0)
        width = img_info.get("width", 0)
        height = img_info.get("height", 0)

        # Skip tiny images
        if width < min_size or height < min_size:
            continue

        # Crop the image from the rendered page
        crop_box = (
            int(bbox[0] * scale),
            int(bbox[1] * scale),
            int(bbox[2] * scale),
            int(bbox[3] * scale),
        )

        try:
            cropped = page_img.crop(crop_box)

            # Save the image
            filename = f"page_{page_num:03d}_img_{i:03d}_xref_{xref}.png"
            filepath = output_dir / filename
            cropped.save(filepath)

            log.info(f"  {filename}: bbox={bbox}, size={width}x{height}")
            saved_paths.append(filepath)
        except Exception as e:
            log.warning(f"  Failed to crop image {i}: {e}")

    return saved_paths


def main() -> None:
    """Extract images from PDF pages for OCR fixtures."""
    parser = argparse.ArgumentParser(
        description="Extract images from PDF pages for OCR test fixtures."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "pages",
        type=int,
        nargs="+",
        help="Page numbers to extract (1-indexed)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help=f"Scale factor for rendering (default: {DEFAULT_SCALE})",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=10,
        help="Minimum image dimension to include (default: 10)",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        log.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(args.pdf_path)

    total_images = 0
    for page_num in args.pages:
        if page_num < 1 or page_num > len(doc):
            log.warning(f"Skipping invalid page number: {page_num}")
            continue

        saved = extract_images_from_page(
            doc,
            page_num,
            args.output,
            scale=args.scale,
            min_size=args.min_size,
        )
        total_images += len(saved)

    doc.close()
    log.info(f"Extracted {total_images} images to {args.output}")


if __name__ == "__main__":
    main()
