"""Main CLI entry point for PDF extraction tool."""

import logging
from pathlib import Path

import pymupdf

from build_a_long.pdf_extract.classifier import classify_pages
from build_a_long.pdf_extract.cli import (
    ProcessingConfig,
    parse_arguments,
    print_classification_debug,
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
from build_a_long.pdf_extract.extractor import extract_bounding_boxes
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
    """Validate that PDF file exists.

    Args:
        pdf_path: Path to PDF file

    Returns:
        True if file exists, False otherwise
    """
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        return False
    return True


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


def _process_pdf(config: ProcessingConfig) -> int:
    """Process the PDF file with the given configuration.

    Args:
        config: Processing configuration

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logging.info("Processing PDF: %s", config.pdf_path)

    # Extract and classify
    with pymupdf.open(str(config.pdf_path)) as doc:
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
            save_raw_json(pages, config.output_dir, config.pdf_path)

        # Classify elements
        batch_result = classify_pages(pages)

        # Debug outputs
        if config.print_histogram:
            print_histogram(batch_result.histogram)

        if config.debug_classification:
            for page, result in zip(pages, batch_result.results, strict=True):
                print_label_counts(page, result)
                print_classification_debug(page, result)
            build_and_print_page_hierarchy(pages, batch_result.results)

        # Summary output
        if config.save_summary:
            print_summary(pages, batch_result.results, detailed=config.summary_detailed)

        # Save results
        save_classified_json(
            pages, batch_result.results, config.output_dir, config.pdf_path
        )

        if config.draw_images:
            render_annotated_images(
                doc,
                pages,
                batch_result.results,
                config.output_dir,
                draw_deleted=config.draw_deleted,
            )

    return 0


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
    if not _validate_pdf_path(config.pdf_path):
        return 2

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Process the PDF
    return _process_pdf(config)


if __name__ == "__main__":
    raise SystemExit(main())
