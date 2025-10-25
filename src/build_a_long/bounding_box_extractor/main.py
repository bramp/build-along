import argparse
import json
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any, Dict, List

import pymupdf

from build_a_long.bounding_box_extractor.extractor import (
    extract_bounding_boxes,
    PageData,
)
from build_a_long.bounding_box_extractor.classifier import classify_elements
from build_a_long.bounding_box_extractor.drawing import draw_and_save_bboxes
from build_a_long.bounding_box_extractor.parser import parse_page_range

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _element_to_json(ele: Any) -> Dict[str, Any]:
    """Convert a PageElement to a JSON-friendly dict using asdict()."""
    data = asdict(ele)
    data["__type__"] = ele.__class__.__name__
    return data


def _node_to_json(element: Any) -> Dict[str, Any]:
    """Convert a PageElement with children to JSON recursively."""
    return {
        "element": _element_to_json(element),
        "children": [_node_to_json(c) for c in element.children],
    }


def serialize_extracted_data(pages: List[PageData]) -> Dict[str, Any]:
    """Convert extracted data with dataclass elements to JSON-serializable format.

    Args:
        pages: List of PageData containing all pages

    Returns:
        JSON-serializable dictionary with type metadata
    """
    json_data: Dict[str, Any] = {"pages": []}
    for page_data in pages:
        json_page: Dict[str, Any] = {"page_number": page_data.page_number}
        json_page["elements"] = [_element_to_json(e) for e in page_data.elements]
        json_page["hierarchy"] = [_node_to_json(n) for n in page_data.root.children]
        json_data["pages"].append(json_page)
    return json_data


def save_json(pages: List[PageData], output_dir: Path, pdf_path: Path) -> None:
    """Save extracted data as JSON file.

    Args:
        pages: List of PageData to serialize
        output_dir: Directory where JSON should be saved
        pdf_path: Original PDF path (used for naming the JSON file)
    """
    json_data = serialize_extracted_data(pages)
    output_json_path = output_dir / (pdf_path.stem + ".json")
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    logger.info("Saved JSON to %s", output_json_path)


def render_annotated_images(
    doc: pymupdf.Document, pages: List[PageData], output_dir: Path
) -> None:
    """Render PDF pages with annotated bounding boxes as PNG images.

    Args:
        doc: The open PyMuPDF Document
        pages: List of PageData containing hierarchy information
        output_dir: Directory where PNG images should be saved
    """
    for page_data in pages:
        page_num = page_data.page_number  # 1-indexed
        page = doc[page_num - 1]  # 0-indexed
        output_path = output_dir / f"page_{page_num:03d}.png"
        draw_and_save_bboxes(
            page,
            page_data.root.children,
            output_path,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Extract bounding boxes from a PDF file and export images/JSON for debugging."
    )
    parser.add_argument("pdf_path", help="The path to the PDF file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save images and JSON files. Defaults to same directory as PDF.",
    )
    parser.add_argument(
        "--pages",
        type=str,
        help='Page range to process (1-indexed), e.g., "5", "5-10", "10-" (from 10 to end), or "-5" (from 1 to 5). Defaults to all pages.',
    )
    parser.add_argument(
        "--include-types",
        type=str,
        help="Comma-separated list of element types to include. Defaults to all types.",
        default="text,image,drawing",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for the bounding box extractor CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_arguments()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        return 2

    # Default output directory to same directory as PDF
    output_dir = args.output_dir if args.output_dir else pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse page range
    start_page = None
    end_page = None
    if args.pages:
        try:
            start_page, end_page = parse_page_range(args.pages)
        except ValueError as e:
            logger.error("Invalid page range: %s", e)
            return 2

    include_types = args.include_types.split(",")

    logging.info("Processing PDF: %s", pdf_path)

    # Extract bounding box data from PDF (open document once and reuse)
    with pymupdf.open(str(pdf_path)) as doc:
        pages: List[PageData] = extract_bounding_boxes(
            doc, start_page, end_page, include_types=include_types
        )

        # Classify elements to add labels (e.g., page numbers)
        classify_elements(pages)

        # Save results as JSON and render annotated images
        save_json(pages, output_dir, pdf_path)
        render_annotated_images(doc, pages, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
