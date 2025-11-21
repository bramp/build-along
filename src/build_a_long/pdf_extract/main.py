"""Main CLI entry point for PDF extraction tool."""

import logging
from pathlib import Path

import pymupdf

from build_a_long.pdf_extract.classifier import classify_elements, classify_pages
from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.cli import (
    ProcessingConfig,
    parse_arguments,
    print_classification_debug,
    print_font_hints,
    print_histogram,
    print_label_counts,
    render_annotated_images,
    save_classified_json,
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
from build_a_long.pdf_extract.parser import parse_page_ranges
from build_a_long.pdf_extract.parser.page_ranges import PageRanges

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


def _process_pdf(config: ProcessingConfig, pdf_path: Path, output_dir: Path) -> int:
    """Process a single PDF file with the given configuration.

    Args:
        config: Processing configuration
        pdf_path: Path to the PDF file to process
        output_dir: Output directory for this PDF

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logging.info("Processing PDF: %s", pdf_path)

    # Extract and classify
    with pymupdf.open(str(pdf_path)) as doc:
        page_ranges = _parse_page_selection(config.page_ranges, len(doc))
        if page_ranges is None:
            return 2

        logger.info("Selected pages: %s", page_ranges)
        page_numbers = list(page_ranges.page_numbers(len(doc)))

        pages = extract_bounding_boxes(
            doc, page_numbers, include_types=config.include_types
        )

        # Save raw JSON if requested
        if config.save_raw_json:
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
            font_hints = FontSizeHints.from_pages(pages)
            print_font_hints(font_hints)

        # Classify elements
        batch_result = classify_pages(pages)

        # Extract page_data from results for compatibility
        classified_pages = [result.page_data for result in batch_result.results]

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
        save_classified_json(
            classified_pages, batch_result.results, output_dir, pdf_path
        )

        if config.draw_blocks or config.draw_elements:
            render_annotated_images(
                doc,
                pages,
                batch_result.results,
                output_dir,
                draw_blocks=config.draw_blocks,
                draw_elements=config.draw_elements,
                draw_deleted=config.draw_deleted,
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
            print_label_counts(page, result)
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

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process this PDF
            exit_code = _process_pdf(config, file_path, output_dir)

        if exit_code != 0:
            return exit_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
