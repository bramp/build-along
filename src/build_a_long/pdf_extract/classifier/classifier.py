"""
Rule-based classifier for labeling page elements.

Pipeline order and dependencies
--------------------------------
The classification pipeline operates in two main phases:

1. **Bottom-up Scoring**: All classifiers run independently to identify potential
   candidates (e.g. page numbers, part counts, step numbers) and score them based
   on heuristics. No construction of final elements happens here.

2. **Top-down Construction**: The root `PageClassifier` is invoked to construct
   the final `Page` object. It recursively requests the construction of its
   dependencies (e.g. "Give me the best PageNumber"), which in turn construct
   their own dependencies. This ensures a consistent and validated object tree.

"""

from __future__ import annotations

import logging

from build_a_long.pdf_extract.classifier.bags import (
    BagNumberClassifier,
    NewBagClassifier,
)
from build_a_long.pdf_extract.classifier.batch_classification_result import (
    BatchClassificationResult,
)
from build_a_long.pdf_extract.classifier.block_filter import (
    filter_background_blocks,
    filter_duplicate_blocks,
    filter_overlapping_text_blocks,
)
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.pages import (
    PageHintCollection,
)
from build_a_long.pdf_extract.classifier.pages.page_classifier import PageClassifier
from build_a_long.pdf_extract.classifier.pages.page_number_classifier import (
    PageNumberClassifier,
)
from build_a_long.pdf_extract.classifier.pages.progress_bar_classifier import (
    ProgressBarClassifier,
)
from build_a_long.pdf_extract.classifier.parts import (
    PartCountClassifier,
    PartNumberClassifier,
    PartsClassifier,
    PartsImageClassifier,
    PartsListClassifier,
    PieceLengthClassifier,
    ShineClassifier,
)
from build_a_long.pdf_extract.classifier.removal_reason import RemovalReason
from build_a_long.pdf_extract.classifier.steps import (
    ArrowClassifier,
    DiagramClassifier,
    RotationSymbolClassifier,
    StepClassifier,
    StepNumberClassifier,
)
from build_a_long.pdf_extract.classifier.text import FontSizeHints, TextHistogram
from build_a_long.pdf_extract.classifier.topological_sort import topological_sort
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PageNumber,
    PartCount,
    PartsList,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks

logger = logging.getLogger(__name__)

# Pages with more blocks than this threshold will be skipped during classification.
# This avoids O(n²) algorithms (like duplicate detection) that become prohibitively
# slow on pages with thousands of vector drawings. Such pages are typically info
# pages where each character is a separate vector graphic.
# TODO: Add spatial indexing to handle high-block pages efficiently.
MAX_BLOCKS_PER_PAGE = 1000


def classify_elements(page: PageData) -> ClassificationResult:
    """Classify and label elements on a single page using rule-based heuristics.

    Args:
        page: A single PageData object to classify.

    Returns:
        A ClassificationResult object containing the classification results.
    """
    config = ClassifierConfig()
    classifier = Classifier(config)

    return classifier.classify(page)


def classify_pages(
    pages: list[PageData], pages_for_hints: list[PageData] | None = None
) -> BatchClassificationResult:
    """Classify and label elements across multiple pages using rule-based heuristics.

    This function performs a three-phase process:
    1. Filtering phase: Mark duplicate/similar blocks as removed on each page
    2. Analysis phase: Build font size hints from text properties (excluding
       removed blocks)
    3. Classification phase: Use hints to guide element classification

    Args:
        pages: A list of PageData objects to classify.
        pages_for_hints: Optional list of pages to use for generating font/page hints.
            If None, uses `pages`. This allows generating hints from all pages
            while only classifying a subset (e.g., when using --pages filter).

    Returns:
        BatchClassificationResult containing per-page results and global histogram
    """
    # Use all pages for hint generation if provided, otherwise use selected pages
    hint_pages = pages_for_hints if pages_for_hints is not None else pages

    # Phase 1: Filter duplicate blocks on each page and track removals
    # Skip pages with too many blocks to avoid O(n²) performance issues
    removed_blocks_per_page: list[dict[Blocks, RemovalReason]] = []
    skipped_pages: set[int] = set()  # Track page numbers that are skipped

    for page_data in pages:
        # Skip pages with too many blocks - these are likely info/inventory pages
        # with vectorized text that cause O(n²) algorithms to be very slow
        if len(page_data.blocks) > MAX_BLOCKS_PER_PAGE:
            logger.debug(
                f"Page {page_data.page_number}: skipping classification "
                f"({len(page_data.blocks)} blocks exceeds threshold of "
                f"{MAX_BLOCKS_PER_PAGE})"
            )
            skipped_pages.add(page_data.page_number)
            removed_blocks_per_page.append({})
            continue

        kept_blocks = page_data.blocks

        # Filter background blocks (full page blocks like background images)
        kept_blocks, background_removed = filter_background_blocks(
            kept_blocks, page_data.bbox.width, page_data.bbox.height
        )

        # Filter overlapping text blocks (e.g., "4" and "43" at same origin)
        kept_blocks, text_removed = filter_overlapping_text_blocks(kept_blocks)

        # Filter duplicate image/drawing blocks based on IOU
        kept_blocks, bbox_removed = filter_duplicate_blocks(kept_blocks)

        # Combine all removal mappings into a single dict for this page
        combined_removed_mapping = {
            **text_removed,
            **bbox_removed,
            **background_removed,
        }

        logger.debug(
            f"Page {page_data.page_number}: "
            f"filtered {len(text_removed)} overlapping text, "
            f"{len(bbox_removed)} duplicate bbox blocks, "
            f"{len(background_removed)} background blocks"
        )

        removed_blocks_per_page.append(combined_removed_mapping)

    # Phase 2: Extract font size hints from hint pages (excluding removed blocks)
    # Build pages with non-removed blocks for hint extraction and histogram

    # Filter duplicates from hint pages (may be different from pages to classify)
    hint_pages_without_duplicates = []
    for page_data in hint_pages:
        # Skip high-block pages for hints too (same threshold)
        if len(page_data.blocks) > MAX_BLOCKS_PER_PAGE:
            continue

        # TODO We are re-filtering duplicates here; optimize by changing the API
        # to accept one list of PageData, and seperate by page_numbers.
        kept_blocks = page_data.blocks
        kept_blocks, _ = filter_background_blocks(
            kept_blocks, page_data.bbox.width, page_data.bbox.height
        )
        kept_blocks, _ = filter_overlapping_text_blocks(kept_blocks)
        kept_blocks, _ = filter_duplicate_blocks(kept_blocks)

        hint_pages_without_duplicates.append(
            PageData(
                page_number=page_data.page_number,
                bbox=page_data.bbox,
                blocks=kept_blocks,
            )
        )

    # Build pages without duplicates for classification
    pages_without_duplicates = []
    for page_data, removed_mapping in zip(pages, removed_blocks_per_page, strict=True):
        # We need to filter blocks that were removed by ANY filter
        non_removed_blocks = [
            block for block in page_data.blocks if block not in removed_mapping
        ]
        pages_without_duplicates.append(
            PageData(
                page_number=page_data.page_number,
                bbox=page_data.bbox,
                blocks=non_removed_blocks,
            )
        )

    # Generate hints from hint pages, histogram from pages to classify
    font_size_hints = FontSizeHints.from_pages(hint_pages_without_duplicates)
    page_hints = PageHintCollection.from_pages(hint_pages_without_duplicates)
    histogram = TextHistogram.from_pages(pages_without_duplicates)

    # Phase 3: Classify using the hints (on pages without duplicates)
    config = ClassifierConfig(font_size_hints=font_size_hints, page_hints=page_hints)
    classifier = Classifier(config)

    results = []
    for page_data, page_without_duplicates, removed_mapping in zip(
        pages, pages_without_duplicates, removed_blocks_per_page, strict=True
    ):
        # Handle skipped pages
        if page_data.page_number in skipped_pages:
            result = ClassificationResult(
                page_data=page_data,
                skipped_reason=(
                    f"Page has {len(page_data.blocks)} blocks, which exceeds "
                    f"the threshold of {MAX_BLOCKS_PER_PAGE}. This is likely an "
                    f"info/inventory page with vectorized text."
                ),
            )
            results.append(result)
            continue

        # Classify using only non-removed blocks
        result = classifier.classify(page_without_duplicates)

        # Update result to use original page_data (with all blocks)
        result.page_data = page_data

        # Mark removed blocks
        for removed_block, removal_reason in removed_mapping.items():
            result.mark_removed(removed_block, removal_reason)

        results.append(result)

    return BatchClassificationResult(results=results, histogram=histogram)


type Classifiers = (
    PageNumberClassifier
    | ProgressBarClassifier
    | BagNumberClassifier
    | PartCountClassifier
    | PartNumberClassifier
    | StepNumberClassifier
    | PieceLengthClassifier
    | PartsClassifier
    | PartsListClassifier
    | PartsImageClassifier
    | ShineClassifier
    | NewBagClassifier
    | DiagramClassifier
    | ArrowClassifier
    | StepClassifier
    | PageClassifier
)


class Classifier:
    """
    Performs a single run of classification based on rules, configuration, and hints.
    This class should be stateless.
    """

    def __init__(self, config: ClassifierConfig):
        self.config = config
        # Sort classifiers topologically based on their dependencies
        self.classifiers = topological_sort(
            [
                PageNumberClassifier(config),
                ProgressBarClassifier(config),
                BagNumberClassifier(config),
                PartCountClassifier(config),
                PartNumberClassifier(config),
                StepNumberClassifier(config),
                PieceLengthClassifier(config),
                PartsClassifier(config),
                PartsListClassifier(config),
                DiagramClassifier(config),
                RotationSymbolClassifier(config),
                ArrowClassifier(config),
                PartsImageClassifier(config),
                ShineClassifier(config),
                NewBagClassifier(config),
                StepClassifier(config),
                PageClassifier(config),
            ]
        )

    def classify(self, page_data: PageData) -> ClassificationResult:
        """
        Runs the classification logic and returns a result.
        It does NOT modify page_data directly.

        The classification process runs in three phases:
        1. Score all classifiers (bottom-up) - auto-registers classifiers
        2. Construct final elements (top-down starting from Page)
        """
        result = ClassificationResult(page_data=page_data)

        logger.debug(f"Starting classification for page {page_data.page_number}")

        # 1. Score all classifiers (Bottom-Up)
        # Note: score() automatically registers each classifier for its output labels
        for classifier in self.classifiers:
            classifier.score(result)

        # 2. Construct (Top-Down)
        # Find the PageClassifier to start the construction process
        page_classifier = next(
            c for c in self.classifiers if isinstance(c, PageClassifier)
        )
        page_classifier.build_all(result)

        # TODO Do we actualy ever add warnings?
        warnings = self._log_post_classification_warnings(page_data, result)
        for warning in warnings:
            result.add_warning(warning)

        return result

    def _log_post_classification_warnings(
        self, page_data: PageData, result: ClassificationResult
    ) -> list[str]:
        warnings = []

        # Check if there's a page number
        page_numbers = result.get_winners_by_score("page_number", PageNumber)
        if not page_numbers:
            warnings.append(f"Page {page_data.page_number}: missing page number")

        # Get elements by label
        parts_lists = result.get_winners_by_score("parts_list", PartsList)
        part_counts = result.get_winners_by_score("part_count", PartCount)

        for pl in parts_lists:
            inside_counts = [t for t in part_counts if t.bbox.fully_inside(pl.bbox)]
            if not inside_counts:
                warnings.append(
                    f"Page {page_data.page_number}: parts list at {pl.bbox} "
                    f"contains no part counts"
                )

        steps = result.get_winners_by_score("step_number", StepNumber)
        ABOVE_EPS = 2.0
        for step in steps:
            sb = step.bbox
            above = [pl for pl in parts_lists if pl.bbox.y1 <= sb.y0 + ABOVE_EPS]
            if not above:
                warnings.append(
                    f"Page {page_data.page_number}: step number '{step.value}' "
                    f"at {sb} has no parts list above it"
                )
        return warnings
