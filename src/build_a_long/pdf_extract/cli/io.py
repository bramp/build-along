"""Input/Output operations for PDF extraction."""

import json
import logging
from pathlib import Path
from typing import Any

import pymupdf

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.drawing import draw_and_save_bboxes
from build_a_long.pdf_extract.extractor import ExtractionResult, PageData

logger = logging.getLogger(__name__)


def _prune_element_metadata(page: dict[str, Any]) -> dict[str, Any]:
    """Prune noisy/empty fields from PageData dict.

    Applies to each element in page["elements"] only:
    - Drop "deleted" when falsy (e.g., False)
    - Drop "label" when None

    Args:
        page: Page dictionary from PageData.to_dict()

    Returns:
        The same page dict with pruned fields (modified in-place)
    """
    elements = page.get("elements", [])
    if isinstance(elements, list):
        for ele in elements:
            if isinstance(ele, dict):
                if not bool(ele.get("deleted")):
                    ele.pop("deleted", None)
                if ele.get("label") is None:
                    ele.pop("label", None)
    return page


# TODO I don't think this works, as it doesn't actually output the classified results
def save_classified_json(
    pages: list[PageData],
    results: list[ClassificationResult],
    output_dir: Path,
    pdf_path: Path,
) -> None:
    """Save extracted data as JSON file.

    Args:
        pages: List of PageData to serialize
        results: List of ClassificationResult (one per page) - not currently serialized
        output_dir: Directory where JSON should be saved
        pdf_path: Original PDF path (used for naming the JSON file)
    """
    json_data = ExtractionResult(pages=pages).to_dict()
    output_json_path = output_dir / (pdf_path.stem + ".json")
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    logger.info("Saved JSON to %s", output_json_path)


def save_raw_json(pages: list[PageData], output_dir: Path, pdf_path: Path) -> None:
    """Save extracted raw data as JSON files, one per page.

    Args:
        pages: List of PageData to serialize
        output_dir: Directory where JSON should be saved
        pdf_path: Original PDF path (used for naming the JSON file)
    """
    for page_data in pages:
        json_page = _prune_element_metadata(page_data.to_dict())

        output_json_path = output_dir / (
            f"{pdf_path.stem}_page_{page_data.page_number:03d}_raw.json"
        )
        with open(output_json_path, "w") as f:
            json.dump(json_page, f, indent=4)
        logger.info(
            "Saved raw JSON for page %d to %s",
            page_data.page_number,
            output_json_path,
        )


def render_annotated_images(
    doc: pymupdf.Document,
    pages: list[PageData],
    results: list[ClassificationResult],
    output_dir: Path,
    *,
    draw_deleted: bool = False,
) -> None:
    """Render PDF pages with annotated bounding boxes as PNG images.

    Args:
        doc: The open PyMuPDF Document
        pages: List of PageData containing extracted elements
        results: List of ClassificationResult with labels and elements
        output_dir: Directory where PNG images should be saved
        draw_deleted: If True, also render elements marked as deleted.
    """
    for page_data, result in zip(pages, results, strict=True):
        page_num = page_data.page_number  # 1-indexed
        page = doc[page_num - 1]  # 0-indexed
        output_path = output_dir / f"page_{page_num:03d}.png"
        draw_and_save_bboxes(page, result, output_path, draw_deleted=draw_deleted)
