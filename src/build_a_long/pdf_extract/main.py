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
)
from build_a_long.pdf_extract.cli.reporting import (
    build_and_print_page_hierarchy,
    print_summary,
)
from build_a_long.pdf_extract.extractor.extractor import (
    ExtractionResult,
    PageData,
    extract_page_data,
)
from build_a_long.pdf_extract.extractor.page_blocks import Image
from build_a_long.pdf_extract.parser import parse_page_ranges
from build_a_long.pdf_extract.parser.page_ranges import PageRanges
from build_a_long.pdf_extract.validation.printer import print_validation
from build_a_long.pdf_extract.validation.runner import validate_results

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
    """Check if a page is dominated by a single large image AND has no other significant content.

    This heuristic aims to identify scanned PDFs where the entire page content
    is essentially one large image, while avoiding false positives for PDFs
    with background images but also selectable text or other elements.

    Args:
        page: The page to check.

    Returns:
        True if the page is dominated by a single image with no other significant content,
        False otherwise.
    """
    page_area = page.bbox.area
    if page_area <= 0:
        return False

    # Find image blocks that cover more than 95% of the page
    # TODO make the 95% threshold configurable if needed
    large_images = []
    for block in page.blocks:
        if isinstance(block, Image) and block.bbox.area / page_area > 0.95:
            large_images.append(block)

    if len(large_images) > (len(page.blocks) / 2):
        logger.debug(
            "Page %d has multiple large images (%d) covering >95%% of page, considered full-page image.",
            page.page_number,
            len(large_images),
        )
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

        # Extract text-only for all pages for font hint generation
        # (hints need global context across the entire document)
        logger.debug(
            "Extracting text blocks from all pages for font hint generation..."
        )
        full_document_text_pages = extract_page_data(
            doc,
            list(range(1, len(doc) + 1)),  # All pages
            include_types={"text"},  # Only text blocks
            include_metadata=False,  # No extra metadata needed for hints
        )
        logger.debug("Finished extracting text blocks for hints.")

        # Extract full data for requested pages
        # Always include metadata - classifiers need it
        # (e.g., ArrowClassifier needs Drawing.items to detect arrowheads)
        logger.debug(f"Extracting all blocks from requested pages: {page_numbers}...")
        requested_pages_with_all_blocks = extract_page_data(
            doc,
            page_numbers,  # Only requested pages
            include_types=config.include_types,  # All types as per config
            include_metadata=True,
        )
        logger.debug("Finished extracting all blocks from requested pages.")

        # At this point, `pages` refers to the original `all_pages` which is no longer
        # the case. We need to set `pages` to `requested_pages_with_all_blocks`
        pages = requested_pages_with_all_blocks

        # Modify full-page image check (applied to
        # requested_pages_with_all_blocks)
        # TODO Let's move this into the validation checks
        full_page_image_count = sum(1 for p in pages if _is_full_page_image(p))
        if len(pages) > 0 and (full_page_image_count / len(pages)) > 0.5:
            reason = (
                f"Warning: More than 50% of processed pages appear to be composed of full-page images "
                f"({full_page_image_count}/{len(pages)} pages). "
                f"Processing will continue, but results for these pages may be poor."
            )
            logger.warning(reason)

        # Print font hints if requested (before classification)
        if config.print_font_hints:
            # TODO Why do we generate hints here, AND inside classify_pages
            font_hints = FontSizeHints.from_pages(
                full_document_text_pages
            )  # Use text-only pages for hints
            print_font_hints(font_hints)

        # Classify elements (use full_document_text_pages for hints, but only classify selected pages)
        batch_result = classify_pages(pages, pages_for_hints=full_document_text_pages)

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

        # Run validation checks
        validation = validate_results(classified_pages, batch_result.results)
        print_validation(validation)

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
