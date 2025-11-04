"""
Rule-based classifier for labeling page elements.

Pipeline order and dependencies
--------------------------------
Classifiers run in a fixed, enforced order because later stages depend on
labels produced by earlier stages:

1) PageNumberClassifier → outputs: "page_number"
2) PartCountClassifier  → outputs: "part_count"
3) StepNumberClassifier → outputs: "step_number" (uses page_number size as context)
4) PartsListClassifier  → outputs: "parts_list" (requires step_number and part_count)
5) PartsImageClassifier → outputs: "part_image" (requires parts_list and part_count)

If the order is changed such that a classifier runs before its requirements
are available, a ValueError will be raised at initialization time.
"""

import logging
from typing import List, Optional, Set

from build_a_long.pdf_extract.classifier.page_number_classifier import (
    PageNumberClassifier,
)
from build_a_long.pdf_extract.classifier.part_count_classifier import (
    PartCountClassifier,
)
from build_a_long.pdf_extract.classifier.parts_list_classifier import (
    PartsListClassifier,
)
from build_a_long.pdf_extract.classifier.parts_image_classifier import (
    PartsImageClassifier,
)
from build_a_long.pdf_extract.classifier.step_number_classifier import (
    StepNumberClassifier,
)
from build_a_long.pdf_extract.classifier.types import (
    ClassifierConfig,
    ClassificationHints,
    ClassificationResult,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import Text

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
    orchestrator = ClassificationOrchestrator(classifier)

    return orchestrator.process_page(page)


def classify_pages(pages: List[PageData]) -> List[ClassificationResult]:
    """Classify and label elements across multiple pages using rule-based heuristics.

    Args:
        pages: A list of PageData objects to classify.

    Returns:
        List of ClassificationResult objects, one per page.
    """
    config = ClassifierConfig()
    classifier = Classifier(config)
    orchestrator = ClassificationOrchestrator(classifier)

    results = []
    for page_data in pages:
        result = orchestrator.process_page(page_data)
        results.append(result)

    return results


class Classifier:
    """
    Performs a single run of classification based on rules, configuration, and hints.
    This class should be stateless.
    """

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.classifiers = [
            PageNumberClassifier(config, self),
            PartCountClassifier(config, self),
            StepNumberClassifier(config, self),
            PartsListClassifier(config, self),
            PartsImageClassifier(config, self),
        ]

        produced: Set[str] = set()
        for c in self.classifiers:
            cls = c.__class__
            need = getattr(c, "requires", set())
            if not need.issubset(produced):
                missing = ", ".join(sorted(need - produced))
                raise ValueError(
                    f"Classifier order invalid: {cls.__name__} requires labels not yet produced: {missing}"
                )
            produced |= getattr(c, "outputs", set())

    def classify(
        self, page_data: PageData, hints: Optional[ClassificationHints] = None
    ) -> ClassificationResult:
        """
        Runs the classification logic and returns a result.
        It does NOT modify page_data directly.
        """
        result = ClassificationResult()

        for classifier in self.classifiers:
            classifier.evaluate(page_data, result)
            classifier.classify(page_data, result, hints)

        warnings = self._log_post_classification_warnings(page_data, result)
        for warning in warnings:
            result.add_warning(warning)

        # Extract persisted relations from PartsImageClassifier
        part_image_pairs = []
        for classifier in self.classifiers:
            if isinstance(classifier, PartsImageClassifier):
                part_image_pairs = classifier.get_part_image_pairs()
                break

        result.part_image_pairs = part_image_pairs

        return result

    def _remove_child_bboxes(
        self,
        page_data: PageData,
        target,
        result: ClassificationResult,
        keep_ids: Optional[Set[int]] = None,
    ) -> None:
        if keep_ids is None:
            keep_ids = set()

        target_bbox = target.bbox

        for ele in page_data.elements:
            if ele is target or id(ele) in keep_ids:
                continue
            b = ele.bbox
            if b.fully_inside(target_bbox):
                result.mark_removed(
                    ele, RemovalReason(reason_type="child_bbox", target_element=target)
                )

    def _remove_similar_bboxes(
        self,
        page_data: PageData,
        target,
        result: ClassificationResult,
        keep_ids: Optional[Set[int]] = None,
    ) -> None:
        if keep_ids is None:
            keep_ids = set()

        target_area = target.bbox.area
        tx, ty = target.bbox.center

        IOU_THRESHOLD = 0.8
        CENTER_EPS = 1.5
        AREA_TOL = 0.12

        for ele in page_data.elements:
            if ele is target or id(ele) in keep_ids:
                continue

            b = ele.bbox
            iou = target.bbox.iou(b)
            if iou >= IOU_THRESHOLD:
                result.mark_removed(
                    ele,
                    RemovalReason(reason_type="similar_bbox", target_element=target),
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
                        RemovalReason(
                            reason_type="similar_bbox", target_element=target
                        ),
                    )

    def _log_post_classification_warnings(
        self, page_data: PageData, result: ClassificationResult
    ) -> List[str]:
        warnings = []

        labeled_elements = result.get_labeled_elements()

        # Check if there's a page number
        has_page_number = any(
            label == "page_number" for label in labeled_elements.values()
        )
        if not has_page_number:
            warnings.append(f"Page {page_data.page_number}: missing page number")

        # Get elements by label
        parts_lists = [
            e for e, label in labeled_elements.items() if label == "parts_list"
        ]
        part_counts = [
            e for e, label in labeled_elements.items() if label == "part_count"
        ]

        for pl in parts_lists:
            inside_counts = [t for t in part_counts if t.bbox.fully_inside(pl.bbox)]
            if not inside_counts:
                warnings.append(
                    f"Page {page_data.page_number}: parts list at {pl.bbox} contains no part counts"
                )

        steps: list[Text] = [
            e
            for e, label in labeled_elements.items()
            if label == "step_number" and isinstance(e, Text)
        ]
        ABOVE_EPS = 2.0
        for step in steps:
            sb = step.bbox
            above = [pl for pl in parts_lists if pl.bbox.y1 <= sb.y0 + ABOVE_EPS]
            if not above:
                warnings.append(
                    f"Page {page_data.page_number}: step number '{step.text}' at {sb} has no parts list above it"
                )
        return warnings


class ClassificationOrchestrator:
    """
    Manages the backtracking classification process.
    This class is stateful.
    """

    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.history: List[ClassificationResult] = []

    def process_page(self, page_data: PageData) -> ClassificationResult:
        """
        Orchestrates the classification of a single page, with backtracking.

        Returns:
            The final ClassificationResult containing labels and removal info.
        """
        hints = ClassificationHints()

        max_iterations = 1  # TODO raise this in future
        for i in range(max_iterations):
            result = self.classifier.classify(page_data, hints)
            self.history.append(result)

            inconsistencies = result.get_warnings()
            if not inconsistencies:
                return result

            hints = self._generate_new_hints(result, inconsistencies)

        final_result = self.history[-1]
        return final_result

    def _generate_new_hints(
        self, result: ClassificationResult, inconsistencies: List[str]
    ) -> ClassificationHints:
        """
        Creates new hints to guide the next classification run.
        """
        # TODO Here, we should look at the font sizes, and use that to help
        # bias the next run towards more consistent results.
        # TODO Figure out how to pass the hints between pages (of the book).
        return ClassificationHints()
