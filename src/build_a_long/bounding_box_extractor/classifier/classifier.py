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
from typing import List

from build_a_long.bounding_box_extractor.extractor import PageData
from build_a_long.bounding_box_extractor.extractor.page_elements import Text

logger = logging.getLogger(__name__)

# Minimum score threshold for assigning a label
MIN_CONFIDENCE_THRESHOLD = 0.5


def _score_page_number_text(text: str) -> float:
    """Check if text content is likely to be a page number.

    Page numbers are typically:
    - Small integers (1-3 digits)
    - Sometimes with leading zeros (e.g., "001", "01")
    - May have formatting like "Page 5", "p.5", or just "5"

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

    # Common page number formats - medium-high confidence
    if re.match(r"^(page|p\.?)\s*\d{1,3}$", text, re.IGNORECASE):
        return 0.85

    # Not a page number
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

    # Get page dimensions from root bbox
    page_bbox = page_data.root.bbox
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

    # Find the element with the highest page_number score
    best_candidate: Text | None = None
    best_score = MIN_CONFIDENCE_THRESHOLD

    for element in page_data.elements:
        if not isinstance(element, Text):
            continue

        score = element.label_scores.get("page_number", 0.0)
        if score > best_score:
            best_score = score
            best_candidate = element

    # Label the best candidate
    if best_candidate:
        best_candidate.label = "page_number"
        logger.info(
            "Labeled element as page_number on page %d: %r (score=%.2f)",
            page_data.page_number,
            best_candidate.text,
            best_score,
        )
    else:
        logger.debug(
            "No page number candidates found on page %d with score >= %.2f",
            page_data.page_number,
            MIN_CONFIDENCE_THRESHOLD,
        )


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
        # Future scorers can be added here:
        # _calculate_step_number_scores(page_data)
        # _calculate_parts_list_scores(page_data)

        # Phase 2: Assign labels based on scores
        _classify_page_number(page_data)
        # Future classifiers can be added here:
        # _classify_step_numbers(page_data)
        # _classify_parts_list(page_data)
