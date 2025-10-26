"""
Parts list classifier.
"""

from typing import TYPE_CHECKING, Any, Dict, Set

from build_a_long.bounding_box_extractor.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.bounding_box_extractor.classifier.part_count_classifier import (
    PartCountClassifier,
)
from build_a_long.bounding_box_extractor.classifier.types import ClassifierConfig
from build_a_long.bounding_box_extractor.extractor import PageData
from build_a_long.bounding_box_extractor.extractor.page_elements import Drawing, Text

if TYPE_CHECKING:
    from build_a_long.bounding_box_extractor.classifier.classifier import Classifier


class PartsListClassifier(LabelClassifier):
    """Classifier for parts lists."""

    def __init__(self, config: ClassifierConfig, classifier: "Classifier"):
        super().__init__(config, classifier)

    def calculate_scores(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
    ) -> None:
        pass

    def classify(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
        to_remove: Set[int],
    ) -> None:
        steps = labeled_elements.get("step_number", [])
        if not steps:
            return

        texts: list[Text] = [e for e in page_data.elements if isinstance(e, Text)]
        if not texts:
            return

        drawings: list[Drawing] = [
            e for e in page_data.elements if isinstance(e, Drawing)
        ]
        if not drawings:
            return

        ABOVE_EPS = 2.0
        used_drawings: set[int] = set()
        if "parts_list" not in labeled_elements:
            labeled_elements["parts_list"] = []

        for step in steps:
            sb = step.bbox
            candidates: list[tuple[Drawing, int, float, float]] = []
            for d in drawings:
                if id(d) in used_drawings:
                    continue
                db = d.bbox
                if db.y1 > sb.y0 + ABOVE_EPS:
                    continue

                contained = [
                    t
                    for t in texts
                    if t.bbox.fully_inside(db)
                    and (
                        t in labeled_elements.get("part_count", [])
                        or PartCountClassifier._score_part_count_text(t.text) >= 0.9
                    )
                ]
                if not contained:
                    continue
                count = len(contained)
                proximity = max(0.0, sb.y0 - db.y1)
                area = db.area()
                candidates.append((d, count, proximity, area))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[2])
            chosen, _, _, _ = candidates[0]
            labeled_elements["parts_list"].append(chosen)
            used_drawings.add(id(chosen))

            keep_ids: Set[int] = set()
            chosen_bbox = chosen.bbox
            for ele in page_data.elements:
                if ele.bbox.fully_inside(chosen_bbox) and (
                    any(
                        ele in v
                        for v in labeled_elements.values()
                        if isinstance(v, list)
                    )
                    or isinstance(ele, Drawing)
                ):
                    keep_ids.add(id(ele))

            self.classifier._remove_child_bboxes(page_data, chosen, to_remove, keep_ids)
            self.classifier._remove_similar_bboxes(
                page_data, chosen, to_remove, keep_ids
            )
