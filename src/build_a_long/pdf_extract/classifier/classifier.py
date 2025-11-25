"""
Rule-based classifier for labeling page elements.

Pipeline order and dependencies
--------------------------------
Classifiers run in dependency order, automatically determined at runtime.
Each classifier declares its `requires` (input labels) and `outputs` (produced labels).

The Classifier batches classifiers into groups where:
- All classifiers in a batch have their dependencies satisfied by previous batches
- Classifiers within a batch can theoretically run in parallel (same dependencies)

Example dependency chain:
1) PageNumberClassifier  → outputs: "page_number" (no dependencies)
2) ProgressBarClassifier → outputs: "progress_bar" (requires: page_number)
3) PartCountClassifier   → outputs: "part_count" (no dependencies)
4) StepNumberClassifier  → outputs: "step_number" (requires: page_number)
5) PartsClassifier       → outputs: "part"
                            (requires: part_count, part_number?, piece_length?)
6) PartsListClassifier   → outputs: "parts_list" (requires: part)
7) DiagramClassifier     → outputs: "diagram"
                            (requires: parts_list, progress_bar)
8) StepClassifier        → outputs: "step"
                            (requires: step_number, parts_list)
9) PageClassifier        → outputs: "page"
                            (requires: page_number, progress_bar, new_bag,
                             step, parts_list)

The actual execution order is determined by the dependency graph and may differ
from the order classifiers are registered in __init__.
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
from build_a_long.pdf_extract.classifier.conflict_resolution import (
    resolve_label_conflicts,
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

        # Order classifiers by dependencies and cache the batches
        # This will raise ValueError if there are circular or missing dependencies
        self.batches = self._order_classifiers_by_dependencies()

    def _order_classifiers_by_dependencies(
        self,
    ) -> list[list[LabelClassifier]]:
        """Order classifiers into batches based on their dependencies.

        Returns a list of batches, where each batch contains classifiers that can
        run in parallel (all their dependencies are satisfied by previous batches).

        Returns:
            List of batches, where each batch is a list of classifiers.

        Raises:
            ValueError: If circular dependencies are detected.
        """
        batches: list[list[LabelClassifier]] = []
        remaining = list(self.classifiers)
        produced: set[str] = set()
        max_iterations = len(self.classifiers) + 1  # Safety limit

        for _ in range(max_iterations):
            if not remaining:
                break

            # Find all classifiers whose dependencies are satisfied
            current_batch: list[LabelClassifier] = []
            for classifier in remaining:
                if classifier.requires.issubset(produced):
                    current_batch.append(classifier)

            # If no classifiers can run, we have circular dependencies or missing deps
            if not current_batch:
                unsatisfied = []
                for classifier in remaining:
                    missing = classifier.requires - produced
                    if missing:
                        cls_name = classifier.__class__.__name__
                        unsatisfied.append(f"{cls_name} (needs: {sorted(missing)})")

                raise ValueError(
                    f"Circular dependency or missing dependencies detected. "
                    f"Cannot satisfy: {'; '.join(unsatisfied)}"
                )

            # Add this batch and update state
            batches.append(current_batch)
            for classifier in current_batch:
                produced |= classifier.outputs

            # Remove processed classifiers
            remaining = [c for c in remaining if c not in current_batch]

        if remaining:
            # This shouldn't happen if max_iterations is sufficient
            raise ValueError(
                f"Failed to schedule all classifiers after {max_iterations} iterations"
            )

        return batches

    def classify(self, page_data: PageData) -> ClassificationResult:
        """
        Runs the classification logic and returns a result.
        It does NOT modify page_data directly.

        The classification process runs in batches based on dependencies:
        - For each batch: score → resolve conflicts → construct
        - This ensures candidates are available for dependent batches
        """
        result = ClassificationResult(page_data=page_data)

        logger.debug(
            f"Running classifiers in {len(self.batches)} batches "
            f"for page {page_data.page_number}"
        )

        # Process each batch: score, resolve conflicts, construct
        for batch_idx, batch in enumerate(self.batches):
            logger.debug(
                f"  Batch {batch_idx + 1}: "
                f"{', '.join(c.__class__.__name__ for c in batch)}"
            )

            # Phase 1: Score all classifiers in this batch
            for classifier in batch:
                classifier.score(result)

            # Phase 2: Resolve conflicts for this batch
            resolve_label_conflicts(result)

            # Phase 3: Construct all classifiers in this batch
            for classifier in batch:
                classifier.construct(result)

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
