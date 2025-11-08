"""
Rule-based classifier for labeling page elements.

Pipeline order and dependencies
--------------------------------
Classifiers run in a fixed, enforced order because later stages depend on
labels produced by earlier stages:

1) PageNumberClassifier → outputs: "page_number"
2) PartCountClassifier  → outputs: "part_count"
3) StepNumberClassifier → outputs: "step_number" (uses page_number size)
4) PartsClassifier      → outputs: "part" (requires part_count, pairs with images)
5) PartsListClassifier  → outputs: "parts_list" (requires step_number and part)
6) PartsImageClassifier → outputs: "part_image" (requires parts_list, part_count)
7) StepClassifier       → outputs: "step" (requires step_number and parts_list)

If the order is changed such that a classifier runs before its requirements
are available, a ValueError will be raised at initialization time.
"""

from __future__ import annotations

import logging

from build_a_long.pdf_extract.classifier.block_filter import filter_duplicate_blocks
from build_a_long.pdf_extract.classifier.classification_result import (
    BatchClassificationResult,
    ClassificationResult,
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.classifier.page_number_classifier import (
    PageNumberClassifier,
)
from build_a_long.pdf_extract.classifier.part_count_classifier import (
    PartCountClassifier,
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
from build_a_long.pdf_extract.classifier.step_classifier import (
    StepClassifier,
)
from build_a_long.pdf_extract.classifier.step_number_classifier import (
    StepNumberClassifier,
)
from build_a_long.pdf_extract.classifier.text_histogram import TextHistogram
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Text

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
    pages: list[PageData], max_iterations: int = 1
) -> BatchClassificationResult:
    """Classify and label elements across multiple pages using rule-based heuristics.

    This function performs a three-phase process:
    1. Filtering phase: Remove duplicate/similar blocks on each page
    2. Analysis phase: Build font size hints from text properties across all pages
    3. Classification phase: Use hints to guide element classification

    Args:
        pages: A list of PageData objects to classify.

    Returns:
        BatchClassificationResult containing per-page results and global histogram
    """
    # Phase 1: Filter duplicate blocks on each page
    filtered_pages = []
    for page_data in pages:
        filtered_blocks = filter_duplicate_blocks(page_data.blocks)

        logger.debug(
            f"Page {page_data.page_number}: "
            f"filtered {len(page_data.blocks) - len(filtered_blocks)} "
            f"duplicate blocks"
        )

        # Create a new PageData with filtered blocks
        filtered_page = PageData(
            page_number=page_data.page_number,
            bbox=page_data.bbox,
            blocks=filtered_blocks,
        )
        filtered_pages.append(filtered_page)

    # Phase 2: Extract font size hints from all pages
    font_size_hints = FontSizeHints.from_pages(filtered_pages)

    # Build histogram for result (keeping for compatibility)
    histogram = TextHistogram.from_pages(filtered_pages)

    # Phase 3: Classify using the hints
    config = ClassifierConfig(font_size_hints=font_size_hints)
    classifier = Classifier(config)

    results = []
    for page_data in filtered_pages:
        result = classifier.classify(page_data)
        results.append(result)

    return BatchClassificationResult(results=results, histogram=histogram)


type Classifiers = (
    PageNumberClassifier
    | PartCountClassifier
    | StepNumberClassifier
    | PartsClassifier
    | PartsListClassifier
    | PartsImageClassifier
    | StepClassifier
)


class Classifier:
    """
    Performs a single run of classification based on rules, configuration, and hints.
    This class should be stateless.
    """

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.classifiers: list[Classifiers] = [
            PageNumberClassifier(config, self),
            PartCountClassifier(config, self),
            StepNumberClassifier(config, self),
            PartsClassifier(config, self),
            PartsListClassifier(config, self),
            PartsImageClassifier(config, self),
            StepClassifier(config, self),
        ]

        # TODO Create a directed graph, and run it in order.
        produced: set[str] = set()
        for c in self.classifiers:
            cls = c.__class__
            need = getattr(c, "requires", set())
            if not need.issubset(produced):
                missing = ", ".join(sorted(need - produced))
                raise ValueError(
                    f"Classifier order invalid: {cls.__name__} requires "
                    f"labels not yet produced: {missing}"
                )
            produced |= getattr(c, "outputs", set())

    def classify(self, page_data: PageData) -> ClassificationResult:
        """
        Runs the classification logic and returns a result.
        It does NOT modify page_data directly.
        """
        result = ClassificationResult(page_data=page_data)

        for classifier in self.classifiers:
            classifier.evaluate(page_data, result)
            classifier.classify(page_data, result)

        warnings = self._log_post_classification_warnings(page_data, result)
        for warning in warnings:
            result.add_warning(warning)

        return result

    def _remove_child_bboxes(
        self,
        page_data: PageData,
        target,
        result: ClassificationResult,
        keep_ids: set[int] | None = None,
    ) -> None:
        if keep_ids is None:
            keep_ids = set()

        target_bbox = target.bbox

        for ele in page_data.blocks:
            if ele is target or id(ele) in keep_ids:
                continue
            b = ele.bbox
            if b.fully_inside(target_bbox):
                result.mark_removed(
                    ele, RemovalReason(reason_type="child_bbox", target_block=target)
                )

    def _remove_similar_bboxes(
        self,
        page_data: PageData,
        target,
        result: ClassificationResult,
        keep_ids: set[int] | None = None,
    ) -> None:
        if keep_ids is None:
            keep_ids = set()

        target_area = target.bbox.area
        tx, ty = target.bbox.center

        IOU_THRESHOLD = 0.8
        CENTER_EPS = 1.5
        AREA_TOL = 0.12

        for ele in page_data.blocks:
            if ele is target or id(ele) in keep_ids:
                continue

            b = ele.bbox
            iou = target.bbox.iou(b)
            if iou >= IOU_THRESHOLD:
                result.mark_removed(
                    ele,
                    RemovalReason(reason_type="similar_bbox", target_block=target),
                )
                continue

            cx, cy = b.center
            if abs(cx - tx) <= CENTER_EPS and abs(cy - ty) <= CENTER_EPS:
                area = b.area
                if (
                    target_area > 0
                    and abs(area - target_area) / target_area <= AREA_TOL
                ):
                    result.mark_removed(
                        ele,
                        RemovalReason(reason_type="similar_bbox", target_block=target),
                    )

    def _log_post_classification_warnings(
        self, page_data: PageData, result: ClassificationResult
    ) -> list[str]:
        warnings = []

        labeled_blocks = result.get_labeled_blocks()

        # Check if there's a page number
        has_page_number = any(
            label == "page_number" for label in labeled_blocks.values()
        )
        if not has_page_number:
            warnings.append(f"Page {page_data.page_number}: missing page number")

        # Get elements by label
        parts_lists = [
            e for e, label in labeled_blocks.items() if label == "parts_list"
        ]
        part_counts = [
            e for e, label in labeled_blocks.items() if label == "part_count"
        ]

        for pl in parts_lists:
            inside_counts = [t for t in part_counts if t.bbox.fully_inside(pl.bbox)]
            if not inside_counts:
                warnings.append(
                    f"Page {page_data.page_number}: parts list at {pl.bbox} "
                    f"contains no part counts"
                )

        steps: list[Text] = [
            e
            for e, label in labeled_blocks.items()
            if label == "step_number" and isinstance(e, Text)
        ]
        ABOVE_EPS = 2.0
        for step in steps:
            sb = step.bbox
            above = [pl for pl in parts_lists if pl.bbox.y1 <= sb.y0 + ABOVE_EPS]
            if not above:
                warnings.append(
                    f"Page {page_data.page_number}: step number '{step.text}' "
                    f"at {sb} has no parts list above it"
                )
        return warnings
