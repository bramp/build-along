"""Main CLI entry point for PDF extraction tool."""

import hashlib
import logging
import os
import time
from pathlib import Path

import pymupdf

from build_a_long.pdf_extract.classifier import (
    FontSizeHints,
    classify_elements,
    classify_pages,
)
from build_a_long.pdf_extract.cli import (
    ProcessingConfig,
    parse_arguments,
    print_classification_debug,
    print_font_hints,
    print_histogram,
    render_annotated_images,
    save_debug_json,
    save_manual_json,
    save_raw_json,
)
from build_a_long.pdf_extract.cli.reporting import (
    build_and_print_page_hierarchy,
    print_summary,
)
from build_a_long.pdf_extract.extractor import (
    ExtractionResult,
    extract_bounding_boxes,
)
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import Manual
from build_a_long.pdf_extract.extractor.page_blocks import Image
from build_a_long.pdf_extract.parser import parse_page_ranges
from build_a_long.pdf_extract.parser.page_ranges import PageRanges
from build_a_long.pdf_extract.validation import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    print_validation,
)

logger = logging.getLogger(__name__)


def _setup_logging(log_level: str) -> None:
    """Configure logging based on level.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _validate_pdf_path(pdf_path: Path) -> bool:
    """Validate that PDF or JSON file exists.

    Args:
        pdf_path: Path to PDF or JSON file

    Returns:
        True if file exists, False otherwise
    """
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        return False
    return True


def _load_json_pages(json_path: Path) -> list:
    """Load pages from a JSON fixture file.

    Args:
        json_path: Path to JSON file (raw extraction result)

    Returns:
        List of PageData objects
    """
    logger.info("Loading pages from JSON: %s", json_path)
    extraction = ExtractionResult.model_validate_json(json_path.read_text())

    if not extraction.pages:
        logger.error("No pages found in JSON file: %s", json_path)
        return []

    logger.info("Loaded %d page(s) from JSON", len(extraction.pages))
    return extraction.pages


def _parse_page_selection(pages_arg: str | None, doc_length: int) -> PageRanges | None:
    """Parse page ranges from arguments.

    Args:
        pages_arg: Page range string from command line (e.g., "5-10,15")
        doc_length: Total number of pages in the document

    Returns:
        PageRanges object or None if parsing failed
    """
    if not pages_arg:
        return PageRanges.all()

    try:
        return parse_page_ranges(pages_arg)
    except ValueError as e:
        logger.error("Invalid --pages: %s", e)
        return None


def _is_full_page_image(page: PageData) -> bool:
    """Check if a page is dominated by a single large image.

    Args:
        page: The page to check.

    Returns:
        True if the page is dominated by a single image, False otherwise.
    """
    page_area = page.bbox.area
    if page_area <= 0:
        return False

    # Find image blocks
    images = [b for b in page.blocks if isinstance(b, Image)]
    if not images:
        return False

    # Check if any image covers > 90% of the page
    for img in images:
        if img.bbox.area / page_area > 0.90:
            return True

    return False


def _process_pdf(config: ProcessingConfig, pdf_path: Path, output_dir: Path) -> int:
    """Process a single PDF file with the given configuration.

    Args:
        config: Processing configuration
        pdf_path: Path to the PDF file to process
        output_dir: Output directory for this PDF

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Start timing from this point
    start_time = time.monotonic()

    # Calculate source metadata (size and hash) in a single file open
    with open(pdf_path, "rb") as f:
        source_size = os.fstat(f.fileno()).st_size
        source_hash = hashlib.file_digest(f, "sha256").hexdigest()

    # Extract and classify
    with pymupdf.open(str(pdf_path)) as doc:
        page_ranges = _parse_page_selection(config.page_ranges, len(doc))
        if page_ranges is None:
            return 2

        # Log which PDF and pages we're processing in a single line
        print(f"Processing: {pdf_path} (pages: {page_ranges})")

        page_numbers = list(page_ranges.page_numbers(len(doc)))

        # Extract all pages for font hint generation (hints need global context)
        # Include metadata if debug-extra-json is set OR if we need to draw paths
        include_metadata = config.debug_extra_json or config.draw_drawings

        # Warn if extra metadata will be captured due to draw_drawings
        if (
            config.draw_drawings
            and not config.debug_extra_json
            and config.save_debug_json
        ):
            logger.warning(
                "Drawing paths require extra metadata. Raw JSON output will "
                "include additional metadata (colors, fonts, dimensions, etc.). "
                "Use --debug-extra-json to explicitly enable this."
            )

        all_pages = extract_bounding_boxes(
            doc,
            list(range(1, len(doc) + 1)),
            include_types=config.include_types,
            include_metadata=include_metadata,
        )

        # Check if the PDF is likely a scan / full-page images
        # If more than 50% of pages are full-page images, skip it
        full_page_image_count = sum(1 for p in all_pages if _is_full_page_image(p))
        if len(all_pages) > 0 and (full_page_image_count / len(all_pages)) > 0.5:
            reason = (
                f"PDF appears to be composed of full-page images "
                f"({full_page_image_count}/{len(all_pages)} pages)"
            )

            manual = Manual(
                source_pdf=pdf_path.name,
                source_size=source_size,
                source_hash=source_hash,
                unsupported_reason=reason,
            )
            output_path = save_manual_json(manual, output_dir, pdf_path)

            # Print validation error for skipped PDF
            if config.save_summary:
                validation = ValidationResult()
                validation.add(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        rule="full_page_images",
                        message=reason,
                        details="Skipping classification for this PDF",
                    )
                )
                print()
                print_validation(validation)

            elapsed = time.monotonic() - start_time
            print(f"Saved: {output_path} (took {elapsed:.1f}s)")

            return 0

        # Filter to requested pages for actual processing
        if page_numbers:
            pages = [p for p in all_pages if p.page_number in page_numbers]
        else:
            pages = all_pages

        # Save raw JSON if requested
        if config.save_debug_json:
            # Save per-page files if specific pages were selected
            per_page = bool(page_ranges.ranges)  # True if specific ranges, False if all
            save_raw_json(
                pages,
                output_dir,
                pdf_path,
                compress=config.compress_json,
                per_page=per_page,
            )

        # Print font hints if requested (before classification)
        if config.print_font_hints:
            font_hints = FontSizeHints.from_pages(all_pages)
            print_font_hints(font_hints)

        # Classify elements (use all_pages for hints, but only classify selected pages)
        batch_result = classify_pages(pages, pages_for_hints=all_pages)

        # Extract page_data from results for compatibility
        classified_pages = [result.page_data for result in batch_result.results]

        # Save debug classification JSON if requested
        if config.save_debug_json:
            save_debug_json(
                batch_result.results,
                output_dir,
                pdf_path,
            )

        _print_debug_output(
            config,
            classified_pages,
            batch_result.results,
            batch_result.histogram,
        )

        # Summary output
        if config.save_summary:
            print_summary(
                classified_pages,
                batch_result.results,
                detailed=config.summary_detailed,
            )

        # Save results
        manual = batch_result.manual
        manual.source_pdf = pdf_path.name
        manual.source_size = source_size
        manual.source_hash = source_hash

        output_path = save_manual_json(manual, output_dir, pdf_path)
        elapsed = time.monotonic() - start_time
        print(f"Saved: {output_path} (took {elapsed:.1f}s)")

        if config.draw_blocks or config.draw_elements or config.draw_drawings:
            render_annotated_images(
                doc,
                batch_result.results,
                output_dir,
                pdf_path,
                draw_blocks=config.draw_blocks,
                draw_elements=config.draw_elements,
                draw_deleted=config.draw_deleted,
                draw_drawings=config.draw_drawings,
                debug_candidates_label=config.debug_candidates_label,
            )

    return 0


def _process_json(config: ProcessingConfig, json_path: Path) -> int:
    """Process a JSON file (raw extraction result) with classification only.

    Args:
        config: Processing configuration
        json_path: Path to the JSON file to process

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logging.info("Processing JSON: %s", json_path)

    # Load pages from JSON
    pages = _load_json_pages(json_path)
    if not pages:
        return 2

    # Classify each page individually to get results
    results = [classify_elements(page) for page in pages]

    # Print font hints if requested
    if config.print_font_hints:
        font_hints = FontSizeHints.from_pages(pages)
        print_font_hints(font_hints)

    # For JSON files, we don't have a batch histogram, so pass None
    _print_debug_output(config, pages, results, histogram=None)

    # Summary output
    if config.save_summary:
        print_summary(pages, results, detailed=config.summary_detailed)

    return 0


def _print_debug_output(
    config: ProcessingConfig, pages: list, results: list, histogram
) -> None:
    """Print debug output based on configuration.

    Args:
        config: Processing configuration
        pages: List of PageData
        results: List of ClassificationResult
        histogram: TextHistogram or None
    """
    # Debug outputs
    if config.print_histogram and histogram:
        print_histogram(histogram)

    if config.debug_classification:
        for page, result in zip(pages, results, strict=True):
            print_classification_debug(page, result)
        build_and_print_page_hierarchy(pages, results)

    if config.debug_candidates:
        for page, result in zip(pages, results, strict=True):
            print_classification_debug(
                page,
                result,
                show_hierarchy=False,
                label=config.debug_candidates_label,
            )


def main() -> int:
    """Main entry point for the bounding box extractor CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_arguments()
    _setup_logging(args.log_level)

    # Create configuration from arguments
    config = ProcessingConfig.from_args(args)

    # Validate inputs
    for pdf_path in config.pdf_paths:
        if not _validate_pdf_path(pdf_path):
            return 2

    # Process each file (PDF or JSON)
    for file_path in config.pdf_paths:
        # Check if it's a JSON file (raw extraction result) or PDF
        if file_path.suffix.lower() == ".json":
            # Process JSON file directly (no output dir needed, no extraction)
            exit_code = _process_json(config, file_path)
        else:
            # Process PDF with extraction
            # Determine output directory for this PDF
            if config.output_dir is not None:
                output_dir = config.output_dir
            else:
                # Default to same directory as the PDF
                output_dir = file_path.parent

            # Ensure output directory exists, unless it's /dev/null
            if output_dir != Path("/dev/null"):
                output_dir.mkdir(parents=True, exist_ok=True)

            # Process this PDF
            exit_code = _process_pdf(config, file_path, output_dir)

        if exit_code != 0:
            return exit_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
