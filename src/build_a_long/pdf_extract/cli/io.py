"""Input/Output operations for PDF extraction."""

import bz2
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pymupdf

from build_a_long.pdf_extract.classifier import ClassificationResult
from build_a_long.pdf_extract.cli.output_models import DebugOutput
from build_a_long.pdf_extract.drawing import draw_and_save_bboxes
from build_a_long.pdf_extract.extractor import ExtractionResult, PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import Manual

logger = logging.getLogger(__name__)


class _RoundingEncoder(json.JSONEncoder):
    """JSON encoder that rounds floats to 2 decimal places."""

    DEFAULT_DECIMALS = 2

    def iterencode(self, o: Any, _one_shot: bool = False):
        """Encode object in chunks, rounding floats during iteration."""

        # Pre-process the entire object to round floats before encoding
        def round_floats(obj: Any) -> Any:
            if isinstance(obj, float):
                return round(obj, self.DEFAULT_DECIMALS)
            elif isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return tuple(round_floats(item) for item in obj)
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


def save_debug_json(
    results: list[ClassificationResult],
    output_dir: Path,
    pdf_path: Path,
) -> None:
    """Save debug data with raw blocks and all candidates as JSON.

    This includes all the classification details: blocks, candidates with scores,
    removal reasons, etc. Useful for debugging and understanding the classification
    process.

    Args:
        results: List of ClassificationResult with candidates and classifications
        output_dir: Directory where JSON should be saved
        pdf_path: Original PDF path (used for naming the JSON file)
    """
    # Build debug structure using Pydantic models
    debug_output = DebugOutput(pages=results)

    output_json_path = output_dir / (pdf_path.stem + "_debug.json")
    with open(output_json_path, "w") as f:
        f.write(
            debug_output.model_dump_json(by_alias=True, indent=2, exclude_none=True)
        )
    logger.info("Saved debug JSON to %s", output_json_path)


def save_pages_json(
    manual: Manual,
    output_dir: Path,
    pdf_path: Path,
) -> None:
    """Save final classified Page elements as JSON.

    This outputs the structured, hierarchical Page elements with their catalog,
    steps, parts lists, etc. This is the "final result" of classification.

    Args:
        manual: Manual containing all classified pages
        output_dir: Directory where JSON should be saved
        pdf_path: Original PDF path (used for naming the JSON file)
    """
    output_json_path = output_dir / (pdf_path.stem + ".json")
    with open(output_json_path, "w") as f:
        f.write(manual.to_json(indent=2))
    logger.info("Saved pages JSON to %s", output_json_path)


def save_raw_json(
    pages: list[PageData],
    output_dir: Path,
    pdf_path: Path,
    *,
    compress: bool = False,
    per_page: bool = False,
) -> None:
    """Save extracted raw data as JSON file(s).

    Floats are automatically rounded to 2 decimal places to reduce file size.
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
            json_data = ExtractionResult(pages=[page_data]).model_dump(
                by_alias=True, exclude_none=True
            )
            page_num = page_data.page_number
            output_path = (
                output_dir / f"{pdf_path.stem}_page_{page_num:03d}_raw{suffix}"
            )
            _save_json_file(json_data, output_path, f"page {page_num}")
    else:
        # Save all pages in a single JSON file
        json_data = ExtractionResult(pages=pages).model_dump(
            by_alias=True, exclude_none=True
        )
        output_path = output_dir / f"{pdf_path.stem}_raw{suffix}"
        _save_json_file(json_data, output_path, f"{len(pages)} pages")


def render_annotated_images(
    doc: pymupdf.Document,
    results: list[ClassificationResult],
    output_dir: Path,
    pdf_path: Path,
    *,
    draw_blocks: bool = False,
    draw_elements: bool = False,
    draw_deleted: bool = False,
    draw_drawings: bool = False,
    debug_candidates_label: str | None = None,
) -> None:
    """Render PDF pages with annotated bounding boxes as PNG images.

    Args:
        doc: The open PyMuPDF Document
        results: List of ClassificationResult with labels and elements
        output_dir: Directory where PNG images should be saved
        pdf_path: Path to the original PDF file (used for naming output files)
        draw_blocks: If True, render classified PDF blocks.
        draw_elements: If True, render classified LEGO page elements.
        draw_deleted: If True, also render elements marked as deleted.
        draw_drawings: If True, render the actual drawing paths.
        debug_candidates_label: If provided, only render candidates with this label.
    """
    for result in results:
        page_num = result.page_data.page_number  # 1-indexed
        page = doc[page_num - 1]  # 0-indexed
        output_path = output_dir / f"{pdf_path.stem}_page_{page_num:03d}.png"
        draw_and_save_bboxes(
            page,
            result,
            output_path,
            draw_blocks=draw_blocks,
            draw_elements=draw_elements,
            draw_deleted=draw_deleted,
            draw_drawings=draw_drawings,
            debug_candidates_label=debug_candidates_label,
        )
