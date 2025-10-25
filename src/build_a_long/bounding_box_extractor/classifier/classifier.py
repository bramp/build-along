"""
Rule-based classifier for labeling page elements.

This module provides heuristic-based classification to identify and label
specific types of elements on a PDF page, such as page numbers, step numbers,
parts lists, etc.

Each classifier computes probability scores (0.0 to 1.0) for how likely an
element matches a given label. The element with the highest score above a
threshold is assigned that label.
"""

import logging
import math
import re
from typing import List, Optional, Set

from build_a_long.bounding_box_extractor.extractor import PageData
from build_a_long.bounding_box_extractor.extractor.page_elements import Drawing, Text

logger = logging.getLogger(__name__)

# Minimum score threshold for assigning a label
MIN_CONFIDENCE_THRESHOLD = 0.5


def _score_page_number_text(text: str) -> float:
    """Check if text content is likely to be a page number.

    Page numbers are typically:
    - Small integers (1-3 digits)
    - Sometimes with leading zeros (e.g., "001", "01")

    Args:
        text: The text content to check

    Returns:
        Score from 0.0 to 1.0, where 1.0 is most likely a page number
    """
    text = text.strip()

    # With leading zeros - high confidence
    if re.match(r"^0+\d{1,3}$", text):
        return 0.95

    # Simple numeric check (1-3 digits) - highest confidence
    if re.match(r"^\d{1,3}$", text):
        return 1.0

    # Not a page number
    return 0.0


def _score_part_count_text(text: str) -> float:
    """Score how likely the text represents a piece count like '2x'.

    Rules:
    - umber followed by an 'x' or '×' optionally with space (e.g., '2x', '10 x')
    """
    t = text.strip()
    # number followed by x or times symbol
    if re.fullmatch(r"\d{1,3}\s*[x×]", t, flags=re.IGNORECASE):
        return 1.0
    return 0.0


def _score_step_number_text(text: str) -> float:
    """Score how likely the text represents a step number.

    Rules:
    - 1 to 4 digits
    - No leading zero
    """
    t = text.strip()
    if re.fullmatch(r"[1-9]\d{0,3}", t):
        return 1.0
    return 0.0


def _score_page_number_position(element: Text, page_bbox, page_height: float) -> float:
    """Score how likely element position indicates a page number.

    Page numbers are typically found in:
    - Lower left corner of the page
    - Lower right corner of the page
    - Sometimes centered at the bottom

    Args:
        element: The text element to score
        page_bbox: The bounding box of the entire page
        page_height: Height of the page

    Returns:
        Score from 0.0 to 1.0 based on position
    """
    # Check if element is in bottom region (lower 10% of page)
    bottom_threshold = page_bbox.y1 - (page_height * 0.1)
    element_center_y = (element.bbox.y0 + element.bbox.y1) / 2

    if element_center_y < bottom_threshold:
        return 0.0  # Not in bottom region

    # Calculate position score based on distance from corners
    element_center_x = (element.bbox.x0 + element.bbox.x1) / 2

    # Distance from bottom-left corner
    dist_bottom_left = math.sqrt(
        (element_center_x - page_bbox.x0) ** 2 + (element_center_y - page_bbox.y1) ** 2
    )

    # Distance from bottom-right corner
    dist_bottom_right = math.sqrt(
        (element_center_x - page_bbox.x1) ** 2 + (element_center_y - page_bbox.y1) ** 2
    )

    # Use the minimum distance to either corner
    min_dist = min(dist_bottom_left, dist_bottom_right)

    # Normalize distance to a score (closer = higher score)
    # Using exponential decay: score = e^(-dist/scale)
    # where scale controls how quickly score decreases with distance
    scale = 50.0  # Tune this based on typical page dimensions
    position_score = math.exp(-min_dist / scale)

    return position_score


def _calculate_page_number_scores(page_data: PageData) -> None:
    """Calculate page number probability scores for all text elements.

    This function computes scores for each text element based on:
    - Text content matching page number patterns
    - Position on the page (bottom corners preferred)

    The final score is a weighted combination of these factors.

    Args:
        page_data: The page data containing elements to score
    """
    if not page_data.elements:
        return

    page_bbox = page_data.bbox
    assert page_bbox is not None
    page_height = page_bbox.y1 - page_bbox.y0

    # Calculate scores for each text element
    for element in page_data.elements:
        if not isinstance(element, Text):
            continue

        # Score text content
        text_score = _score_page_number_text(element.text)

        # Score position
        position_score = _score_page_number_position(element, page_bbox, page_height)

        # Combine scores (weighted average)
        # Text pattern is more important (70%) than position (30%)
        final_score = 0.7 * text_score + 0.3 * position_score

        # Store the score
        element.label_scores["page_number"] = final_score

        logger.debug(
            "Page %d element %r: text_score=%.2f, position_score=%.2f, final=%.2f",
            page_data.page_number,
            element.text,
            text_score,
            position_score,
            final_score,
        )


def _calculate_part_count_scores(page_data: PageData) -> None:
    """Calculate piece count scores for text elements.

    Currently uses only text pattern signals; position not considered.
    """
    if not page_data.elements:
        return

    for element in page_data.elements:
        if not isinstance(element, Text):
            continue
        score = _score_part_count_text(element.text)
        if score > 0.0:
            element.label_scores["part_count"] = score


def _calculate_step_number_scores(page_data: PageData) -> None:
    """Calculate step number scores for text elements.

    Incorporates a size heuristic relative to the detected page number:
    step numbers should be graphically taller than the page number.
    """
    if not page_data.elements:
        return

    # Find page number height, if available
    page_num_height: Optional[float] = None
    for e in page_data.elements:
        if isinstance(e, Text) and e.label == "page_number":
            page_num_height = max(0.0, e.bbox.y1 - e.bbox.y0)
            break

    for element in page_data.elements:
        if not isinstance(element, Text):
            continue
        text_score = _score_step_number_text(element.text)
        if text_score == 0.0:
            continue

        size_score = 0.0
        if page_num_height and page_num_height > 0.0:
            h = max(0.0, element.bbox.y1 - element.bbox.y0)
            # Gate: must be at least 10% taller than the page number
            if h <= page_num_height * 1.1:
                final = 0.0
                element.label_scores["step_number"] = final
                continue
            # Map ratio to [0,1] for scoring bonus: 10% -> 0, 60%+ -> 1
            ratio_over = (h / page_num_height) - 1.0
            size_score = max(0.0, min(1.0, ratio_over / 0.5))

        # Combine scores: text dominates
        final = 0.8 * text_score + 0.2 * size_score
        element.label_scores["step_number"] = final


def _classify_page_number(page_data: PageData) -> None:
    """Identify and label the page number element based on scores.

    After scores are calculated, this function selects the element with the
    highest page_number score above the threshold and assigns it the
    "page_number" label.

    Args:
        page_data: The page data containing elements to classify
    """
    if not page_data.elements:
        return

    # Find candidates with scores and potential numeric values
    candidates: list[tuple[Text, float, Optional[int]]] = []
    for element in page_data.elements:
        if not isinstance(element, Text):
            continue
        score = element.label_scores.get("page_number", 0.0)
        if score < MIN_CONFIDENCE_THRESHOLD:
            continue
        value = _extract_page_number_value(element.text)
        candidates.append((element, score, value))

    # Prefer candidates whose numeric value matches the PageData.page_number
    matching = [c for c in candidates if c[2] == page_data.page_number]
    chosen: Optional[tuple[Text, float, Optional[int]]] = None
    if matching:
        chosen = max(matching, key=lambda c: c[1])
    elif candidates:
        chosen = max(candidates, key=lambda c: c[1])

    if chosen is None:
        logger.debug(
            "No page number candidates found on page %d with score >= %.2f",
            page_data.page_number,
            MIN_CONFIDENCE_THRESHOLD,
        )
        return

    best_candidate, best_score, _ = chosen
    best_candidate.label = "page_number"
    logger.info(
        "Labeled element as page_number on page %d: %r (score=%.2f)",
        page_data.page_number,
        best_candidate.text,
        best_score,
    )

    # Remove near-duplicate visual elements around the chosen page number
    to_remove: Set[int] = set()
    _remove_child_bboxes(page_data, best_candidate, to_remove)
    _remove_similar_bboxes(page_data, best_candidate, to_remove)
    _prune_elements(page_data, to_remove)


def _classify_part_counts(page_data: PageData) -> None:
    """Label all text elements that look like piece counts as 'part_count'.

    Unlike page numbers, there can be many per page, so we label all
    elements with score above the threshold individually.
    """
    # First collect candidates to avoid mutating the list while iterating
    candidates: list[Text] = []
    for element in page_data.elements:
        if not isinstance(element, Text):
            continue
        score = element.label_scores.get("part_count", 0.0)
        if score >= MIN_CONFIDENCE_THRESHOLD:
            candidates.append(element)

    # Apply labels
    for ele in candidates:
        ele.label = ele.label or "part_count"

    # Remove shadows/duplicates and fully-contained elements around each
    for ele in candidates:
        if ele in page_data.elements:
            to_remove: Set[int] = set()
            _remove_child_bboxes(page_data, ele, to_remove)
            _remove_similar_bboxes(page_data, ele, to_remove)
            _prune_elements(page_data, to_remove)


def _classify_step_numbers(page_data: PageData) -> None:
    """Label text elements that look like step numbers as 'step_number'.

    Multiple step numbers can exist on a single page.
    """
    candidates: list[Text] = []
    for element in page_data.elements:
        if not isinstance(element, Text):
            continue
        score = element.label_scores.get("step_number", 0.0)
        if score >= MIN_CONFIDENCE_THRESHOLD:
            candidates.append(element)

    for ele in candidates:
        ele.label = ele.label or "step_number"

    for ele in candidates:
        if ele in page_data.elements:
            to_remove: Set[int] = set()
            _remove_child_bboxes(page_data, ele, to_remove)
            _remove_similar_bboxes(page_data, ele, to_remove)
            _prune_elements(page_data, to_remove)


def _classify_parts_list(page_data: PageData) -> None:
    """Detect and label parts list regions.

    Heuristic:
    - Candidate is a Drawing whose bbox is above a step_number's bbox.
    - It must contain at least one element labeled 'part_count' (or strong pattern match).
    - For each step_number, choose exactly one candidate: the closest drawing to the step.
    """
    # Gather labeled step numbers and part counts
    steps: list[Text] = [
        e
        for e in page_data.elements
        if isinstance(e, Text) and e.label == "step_number"
    ]
    if not steps:
        return
    texts: list[Text] = [e for e in page_data.elements if isinstance(e, Text)]
    if not texts:
        return

    drawings: list[Drawing] = [e for e in page_data.elements if isinstance(e, Drawing)]
    if not drawings:
        return

    # Small vertical slack to consider a drawing 'above' the step number
    ABOVE_EPS = 2.0

    used_drawings: set[int] = set()
    for step in steps:
        sb = step.bbox
        # Build candidates per step
        candidates: list[tuple[Drawing, int, float, float]] = []
        for d in drawings:
            if id(d) in used_drawings:
                continue
            db = d.bbox
            # Above the step number
            if db.y1 > sb.y0 + ABOVE_EPS:
                continue
            # Count contained part counts (prefer labeled; fall back to pattern)
            contained = [
                t
                for t in texts
                if t.bbox.fully_inside(db)
                and (t.label == "part_count" or _score_part_count_text(t.text) >= 0.9)
            ]
            if not contained:
                continue
            count = len(contained)
            # distance from drawing bottom to step top (smaller is closer)
            proximity = max(0.0, sb.y0 - db.y1)
            area = db.area()
            candidates.append((d, count, proximity, area))

        if not candidates:
            # Fallback: allow drawings anywhere that contain counts; pick closest by proximity
            fb: list[tuple[Drawing, int, float, float]] = []
            for d in drawings:
                if id(d) in used_drawings:
                    continue
                db = d.bbox
                contained = [
                    t
                    for t in texts
                    if t.bbox.fully_inside(db)
                    and (
                        t.label == "part_count" or _score_part_count_text(t.text) >= 0.9
                    )
                ]
                if not contained:
                    continue
                count = len(contained)
                # proximity: if drawing is below the step, use large penalty
                if db.y1 <= sb.y0:
                    proximity = sb.y0 - db.y1
                else:
                    proximity = (db.y1 - sb.y0) + 1000.0
                area = db.area()
                fb.append((d, count, proximity, area))

            if not fb:
                continue
            # Choose the closest by proximity only
            fb.sort(key=lambda x: x[2])
            candidates = [fb[0]]

        # Choose candidate: minimize proximity only (closest to the step)
        candidates.sort(key=lambda x: x[2])
        chosen, _, _, _ = candidates[0]
        chosen.label = chosen.label or "parts_list"
        used_drawings.add(id(chosen))

        # Remove near-duplicate shadows around the parts list, but keep labeled elements inside
        # (part counts, other drawings, etc. that are semantically part of the list)
        keep_ids: Set[int] = set()
        chosen_bbox = chosen.bbox
        for ele in page_data.elements:
            # Keep elements inside the parts list that have labels or are Drawings
            if ele.bbox.fully_inside(chosen_bbox) and (
                ele.label is not None or isinstance(ele, Drawing)
            ):
                keep_ids.add(id(ele))

        to_remove: Set[int] = set()
        _remove_child_bboxes(page_data, chosen, to_remove, keep_ids)
        _remove_similar_bboxes(page_data, chosen, to_remove, keep_ids)
        _prune_elements(page_data, to_remove)


def classify_elements(pages: List[PageData]) -> None:
    """Classify and label elements across all pages using rule-based heuristics.

    This function applies various classification rules to identify specific
    element types such as page numbers, step numbers, parts lists, etc.

    The classification is done in two phases:
    1. Calculate probability scores for each potential label
    2. Assign labels based on highest scores above threshold

    The scores are stored in each element's label_scores dictionary,
    and the final label is set in the label field.

    Args:
        pages: List of PageData objects to classify
    """
    for page_data in pages:
        logger.debug("Classifying elements on page %d", page_data.page_number)

        # Phase 1: Calculate scores
        _calculate_page_number_scores(page_data)
        _calculate_part_count_scores(page_data)

        # Phase 2: Assign labels based on scores
        _classify_page_number(page_data)
        _classify_part_counts(page_data)

        # Now that page_number is labeled, compute step scores and classify
        _calculate_step_number_scores(page_data)

        _classify_step_numbers(page_data)
        _classify_parts_list(page_data)

        _log_post_classification_warnings(page_data)


def _extract_page_number_value(text: str) -> Optional[int]:
    """Extract a numeric page value from text if present.

    Supports forms like "7", "007", "Page 7", "p.7".
    Returns None if no page-like number is found.
    """
    t = text.strip()
    m = re.match(r"^0*(\d{1,3})$", t)
    if m:
        return int(m.group(1))
    m = re.match(r"^(?:page|p\.?)\s*0*(\d{1,3})$", t, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _remove_child_bboxes(
    page_data: PageData,
    target,
    to_remove_ids: Set[int],
    keep_ids: Optional[Set[int]] = None,
) -> None:
    """Remove elements that are fully contained within the target bbox.

    This removes child elements like glyph strokes or nested drawings that
    are completely inside the target element's bounding box.

    Args:
        page_data: The page data containing elements
        target: The target element whose children should be removed
        to_remove_ids: Set of element IDs to remove (modified in place)
        keep_ids: Optional set of element IDs to keep (won't be removed)
    """
    if keep_ids is None:
        keep_ids = set()

    target_bbox = target.bbox

    for ele in page_data.elements:
        if ele is target or id(ele) in keep_ids:
            continue
        # TODO This could be just target.children.
        b = ele.bbox
        if b.fully_inside(target_bbox):
            to_remove_ids.add(id(ele))


def _remove_similar_bboxes(
    page_data: PageData,
    target,
    to_remove_ids: Set[int],
    keep_ids: Optional[Set[int]] = None,
) -> None:
    """Remove elements with very similar bounding boxes to the target.

    Intended to drop duplicate drawing strokes or shadow copies co-located
    with the selected element (e.g., drop shadows on page numbers or parts lists).

    Args:
        page_data: The page data containing elements
        target: The target element whose duplicates should be removed
        to_remove_ids: Set of element IDs to remove (modified in place)
        keep_ids: Optional set of element IDs to keep (won't be removed)
    """
    if keep_ids is None:
        keep_ids = set()

    target_bbox = target.bbox
    target_area = target_bbox.area()
    tx, ty = target_bbox.center()

    IOU_THRESHOLD = 0.8
    CENTER_EPS = 1.5
    AREA_TOL = 0.12  # +-12%

    for ele in page_data.elements:
        if ele is target or id(ele) in keep_ids:
            continue

        b = ele.bbox
        iou = target_bbox.iou(b)
        if iou >= IOU_THRESHOLD:
            to_remove_ids.add(id(ele))
            continue

        # Fallback: very close centers and near-equal area
        cx, cy = b.center()
        if abs(cx - tx) <= CENTER_EPS and abs(cy - ty) <= CENTER_EPS:
            area = b.area()
            if target_area > 0 and abs(area - target_area) / target_area <= AREA_TOL:
                to_remove_ids.add(id(ele))


def _prune_elements(page_data: PageData, to_remove_ids: Set[int]) -> None:
    """Remove elements with IDs in to_remove_ids from the page.

    Removes from the flat element list and rebuilds the hierarchy.

    Args:
        page_data: The page data to prune
        to_remove_ids: Set of element IDs to remove
    """
    if not to_remove_ids:
        return

    # TODO print useful information (beyond just the id).
    logger.debug(
        "Removing %d duplicate/child elements on page %d",
        len(to_remove_ids),
        page_data.page_number,
    )

    # Prune from flat element list
    page_data.elements = [e for e in page_data.elements if id(e) not in to_remove_ids]

    # Rebuild hierarchy from remaining elements
    # Import here to avoid circular dependency
    from build_a_long.bounding_box_extractor.extractor.hierarchy import (
        build_hierarchy_from_elements,
    )

    page_data.hierarchy = build_hierarchy_from_elements(page_data.elements)


def _log_post_classification_warnings(page_data: PageData) -> None:
    """Emit warnings for common inconsistencies after classification.

    Rules:
    - Page has no page number
    - Parts list contains no part counts (interpreted as no labeled 'part_count' texts inside)
    - Step number has no corresponding parts list above it
    """
    # 1) Page-number presence
    has_page_number = any(
        isinstance(e, Text) and e.label == "page_number" for e in page_data.elements
    )
    if not has_page_number:
        logger.warning("Page %d: missing page number", page_data.page_number)

    # Collect elements we need
    parts_lists: list[Drawing] = [
        e
        for e in page_data.elements
        if isinstance(e, Drawing) and e.label == "parts_list"
    ]
    texts: list[Text] = [e for e in page_data.elements if isinstance(e, Text)]

    # 2) Parts list with zero part counts inside
    for pl in parts_lists:
        inside_counts = [
            t for t in texts if t.label == "part_count" and t.bbox.fully_inside(pl.bbox)
        ]
        if not inside_counts:
            logger.warning(
                "Page %d: parts list at %s contains no part counts",
                page_data.page_number,
                pl.bbox,
            )

    # 3) Step numbers without a parts list above them
    # TODO We should tie together the step number to it's specific parts list.
    steps: list[Text] = [e for e in texts if e.label == "step_number"]
    ABOVE_EPS = 2.0
    for step in steps:
        sb = step.bbox
        # Find any labeled parts list above this step
        above = [pl for pl in parts_lists if pl.bbox.y1 <= sb.y0 + ABOVE_EPS]
        if not above:
            logger.warning(
                "Page %d: step number '%s' at %s has no parts list above it",
                page_data.page_number,
                step.text,
                sb,
            )
