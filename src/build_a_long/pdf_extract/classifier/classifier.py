"""
Rule-based classifier for labeling page elements.

Pipeline order and dependencies
--------------------------------
The classification pipeline operates in two main phases:

1. **Bottom-up Scoring**: All classifiers run independently to identify potential
   candidates (e.g. page numbers, part counts, step numbers) and score them based
   on heuristics. No construction of final elements happens here.

2. **Conflict Resolution**: Global conflict resolution logic runs to identify
   cases where a single element is claimed by multiple candidates (e.g. a number
   could be a step number or a piece length). Candidates are prioritized and
   filtered.

3. **Top-down Construction**: The root `PageClassifier` is invoked to construct
   the final `Page` object. It recursively requests the construction of its
   dependencies (e.g. "Give me the best PageNumber"), which in turn construct
   their own dependencies. This ensures a consistent and validated object tree.

"""

from __future__ import annotations

import logging

from build_a_long.pdf_extract.classifier.bag_number_classifier import (
    BagNumberClassifier,
)
from build_a_long.pdf_extract.classifier.block_filter import filter_duplicate_blocks
from build_a_long.pdf_extract.classifier.classification_result import (
    BatchClassificationResult,
    ClassificationResult,
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.classifier.diagram_classifier import (
    DiagramClassifier,
)
from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.new_bag_classifier import (
    NewBagClassifier,
)
from build_a_long.pdf_extract.classifier.page_classifier import PageClassifier
from build_a_long.pdf_extract.classifier.page_hints import PageHints
from build_a_long.pdf_extract.classifier.page_number_classifier import (
    PageNumberClassifier,
)
from build_a_long.pdf_extract.classifier.part_count_classifier import (
    PartCountClassifier,
)
from build_a_long.pdf_extract.classifier.part_number_classifier import (
    PartNumberClassifier,
)
from build_a_long.pdf_extract.classifier.parts_classifier import (
    PartsClassifier,
)
from build_a_long.pdf_extract.classifier.parts_image_classifier import (
    PartsImageClassifier,
)
from build_a_long.pdf_extract.classifier.parts_list_classifier import (
    PartsListClassifier,
)
from build_a_long.pdf_extract.classifier.piece_length_classifier import (
    PieceLengthClassifier,
)
from build_a_long.pdf_extract.classifier.progress_bar_classifier import (
    ProgressBarClassifier,
)
from build_a_long.pdf_extract.classifier.step_classifier import (
    StepClassifier,
)
from build_a_long.pdf_extract.classifier.step_number_classifier import (
    StepNumberClassifier,
)
from build_a_long.pdf_extract.classifier.text_histogram import TextHistogram
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PageNumber,
    PartCount,
    PartsList,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks

logger = logging.getLogger(__name__)


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
    duplicate_removals: list[dict[Blocks, Blocks]] = []
    for page_data in pages:
        # Get blocks to keep and mapping of removed blocks
        kept_blocks, removed_mapping = filter_duplicate_blocks(page_data.blocks)

        logger.debug(
            f"Page {page_data.page_number}: "
            f"filtered {len(removed_mapping)} duplicate blocks"
        )

        duplicate_removals.append(removed_mapping)

    # Phase 2: Extract font size hints from hint pages (excluding removed blocks)
    # Build pages with non-removed blocks for hint extraction and histogram

    # Filter duplicates from hint pages (may be different from pages to classify)
    hint_pages_without_duplicates = []
    for page_data in hint_pages:
        # TODO We are re-filtering duplicates here; optimize by changing the API
        # to accept one list of PageData, and seperate by page_numbers.
        kept_blocks, _ = filter_duplicate_blocks(page_data.blocks)
        hint_pages_without_duplicates.append(
            PageData(
                page_number=page_data.page_number,
                bbox=page_data.bbox,
                blocks=kept_blocks,
            )
        )

    # Build pages without duplicates for classification
    pages_without_duplicates = []
    for page_data, removed_mapping in zip(pages, duplicate_removals, strict=True):
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
    page_hints = PageHints.from_pages(hint_pages_without_duplicates)
    histogram = TextHistogram.from_pages(pages_without_duplicates)

    # Phase 3: Classify using the hints (on pages without duplicates)
    config = ClassifierConfig(font_size_hints=font_size_hints, page_hints=page_hints)
    classifier = Classifier(config)

    results = []
    for page_data, page_without_duplicates, removed_mapping in zip(
        pages, pages_without_duplicates, duplicate_removals, strict=True
    ):
        # Classify using only non-removed blocks
        result = classifier.classify(page_without_duplicates)

        # Update result to use original page_data (with all blocks)
        result.page_data = page_data

        # Mark duplicate blocks as removed
        for removed_block, kept_block in removed_mapping.items():
            result.mark_removed(
                removed_block,
                RemovalReason(reason_type="duplicate_bbox", target_block=kept_block),
            )

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
    | NewBagClassifier
    | DiagramClassifier
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
        self.classifiers: list[LabelClassifier] = [
            PageNumberClassifier(config),
            ProgressBarClassifier(config),
            BagNumberClassifier(config),
            PartCountClassifier(config),
            PartNumberClassifier(config),
            StepNumberClassifier(config),
            PieceLengthClassifier(config),
            PartsClassifier(config),
            PartsListClassifier(config),
            PartsImageClassifier(config),
            NewBagClassifier(config),
            DiagramClassifier(config),
            StepClassifier(config),
            PageClassifier(config),
        ]

    def classify(self, page_data: PageData) -> ClassificationResult:
        """
        Runs the classification logic and returns a result.
        It does NOT modify page_data directly.

        The classification process runs in three phases:
        1. Score all classifiers (bottom-up) - auto-registers classifiers
        2. Resolve conflicts (global)
        3. Construct final elements (top-down starting from Page)
        """
        result = ClassificationResult(page_data=page_data)

        logger.debug(f"Starting classification for page {page_data.page_number}")

        # 1. Score all classifiers (Bottom-Up)
        # Note: score() automatically registers each classifier for its output labels
        for classifier in self.classifiers:
            classifier.score(result)

        # 2. Resolve conflicts
        # resolve_label_conflicts(result)

        # 3. Construct (Top-Down)
        # Find the PageClassifier to start the construction process
        page_classifier = next(
            c for c in self.classifiers if isinstance(c, PageClassifier)
        )
        page_classifier.build_all(result)

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
