"""Input/Output operations for PDF extraction."""

import bz2
import gzip
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


class _RoundingEncoder(json.JSONEncoder):
    """JSON encoder that rounds floats to 4 decimal places."""

    DEFAULT_DECIMALS = 2

    def iterencode(self, o: Any, _one_shot: bool = False):
        """Encode object in chunks, rounding floats during iteration."""

        # Pre-process the entire object to round floats before encoding
        def round_floats(obj: Any) -> Any:
            if isinstance(obj, float):
                return round(obj, self.DEFAULT_DECIMALS)
            elif isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [round_floats(item) for item in obj]
            return obj

        return super().iterencode(round_floats(o), _one_shot)


def open_compressed(path: Path, mode: str = "rt", **kwargs):
    """Open a file, automatically detecting compression.

    Supports uncompressed files, gzip .gz, and bz2 .bz2 files.
    Works like the built-in open() but handles compressed files transparently.

    Args:
        path: Path to file (compressed or uncompressed)
        mode: File mode (e.g., 'rt', 'rb', 'wt', 'wb')
        **kwargs: Additional arguments passed to the opener (e.g., encoding)

    Returns:
        File handle (text or binary mode depending on mode parameter)

    Examples:
        >>> with open_compressed(Path("data.json.bz2")) as f:
        ...     data = json.load(f)
        >>> with open_compressed(Path("data.txt.gz"), encoding="utf-8") as f:
        ...     text = f.read()
    """
    if path.suffix == ".bz2":
        return bz2.open(path, mode, **kwargs)
    elif path.suffix == ".gz":
        return gzip.open(path, mode, **kwargs)
    else:
        return open(path, mode, **kwargs)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON from file, automatically detecting compression.

    Supports uncompressed .json, gzip .json.gz, and bz2 .json.bz2 files.

    Args:
        path: Path to JSON file (compressed or uncompressed)

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If the file contains invalid JSON
    """
    try:
        with open_compressed(path, "rt", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[return-value]
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from {path}: {e.msg} at line {e.lineno}, "
            f"column {e.colno}"
        ) from e


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
    json_data = ExtractionResult(pages=pages).model_dump()
    output_json_path = output_dir / (pdf_path.stem + ".json")
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    logger.info("Saved JSON to %s", output_json_path)


def save_raw_json(
    pages: list[PageData],
    output_dir: Path,
    pdf_path: Path,
    *,
    compress: bool = False,
    per_page: bool = False,
) -> None:
    """Save extracted raw data as JSON file(s).

    Floats are automatically rounded to 4 decimal places to reduce file size.
    JSON is indented with tabs for better compression and readability.

    Args:
        pages: List of PageData to serialize
        output_dir: Directory where JSON should be saved
        pdf_path: Original PDF path (used for naming the JSON file)
        compress: If True, use bz2 compression (default: False)
        per_page: If True, save one JSON file per page; if False, save all
            pages in a single file
    """

    suffix = ".json.bz2" if compress else ".json"
    opener = bz2.open if compress else open  # type: ignore[assignment]
    compression_note = "compressed " if compress else ""

    # Helper to save a single JSON file
    def _save_json_file(
        data: dict[str, Any], output_path: Path, page_desc: str
    ) -> None:
        with opener(output_path, "wt", encoding="utf-8") as f:  # type: ignore[operator]
            # Use custom encoder to round floats
            json.dump(data, f, indent="\t", cls=_RoundingEncoder)

        logger.info(
            "Saved %sraw JSON for %s to %s",
            compression_note,
            page_desc,
            output_path,
        )

    if per_page:
        # Save individual JSON files for each page
        for page_data in pages:
            json_data = ExtractionResult(pages=[page_data]).model_dump()
            page_num = page_data.page_number
            output_path = (
                output_dir / f"{pdf_path.stem}_page_{page_num:03d}_raw{suffix}"
            )
            _save_json_file(json_data, output_path, f"page {page_num}")
    else:
        # Save all pages in a single JSON file
        json_data = ExtractionResult(pages=pages).model_dump()
        output_path = output_dir / f"{pdf_path.stem}_raw{suffix}"
        _save_json_file(json_data, output_path, f"{len(pages)} pages")


def render_annotated_images(
    doc: pymupdf.Document,
    pages: list[PageData],
    results: list[ClassificationResult],
    output_dir: Path,
    *,
    draw_blocks: bool = False,
    draw_elements: bool = False,
    draw_deleted: bool = False,
    debug_candidates_label: str | None = None,
) -> None:
    """Render PDF pages with annotated bounding boxes as PNG images.

    Args:
        doc: The open PyMuPDF Document
        pages: List of PageData containing extracted elements
        results: List of ClassificationResult with labels and elements
        output_dir: Directory where PNG images should be saved
        draw_blocks: If True, render classified PDF blocks.
        draw_elements: If True, render classified LEGO page elements.
        draw_deleted: If True, also render elements marked as deleted.
        debug_candidates_label: If provided, only render candidates with this label.
    """
    for page_data, result in zip(pages, results, strict=True):
        page_num = page_data.page_number  # 1-indexed
        page = doc[page_num - 1]  # 0-indexed
        output_path = output_dir / f"page_{page_num:03d}.png"
        draw_and_save_bboxes(
            page,
            result,
            output_path,
            draw_blocks=draw_blocks,
            draw_elements=draw_elements,
            draw_deleted=draw_deleted,
            debug_candidates_label=debug_candidates_label,
        )
