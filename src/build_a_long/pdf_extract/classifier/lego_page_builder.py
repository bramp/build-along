"""
Build a Page (LegoPageElement) from ClassificationResult.

This module provides the build_page function that extracts the Page element
constructed by PageClassifier from the classification results.
"""

from __future__ import annotations

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import Page


# TODO Do I need this? It's a simple wrapper, where I could just use
# results.page instead.
def build_page(
    result: ClassificationResult,
) -> Page:
    """Build a Page (LegoPageElement) from classification results.

    This function extracts the Page element that was constructed by PageClassifier
    from the classification results.

    Args:
        result: Classification result containing the Page candidate from PageClassifier

    Returns:
        Page element containing the structured hierarchy and any warnings

    Raises:
        ValueError: If no Page was found in the classification results

    Example:
        >>> pages = extract_bounding_boxes(doc, None)
        >>> results = classify_pages(pages)
        >>> for result in results.results:
        ...     page = build_page(result)
        ...     if page.page_number:
        ...         print(f"Page {page.page_number.value}")
        ...     for step in page.steps:
        ...         print(f"  Step {step.step_number.value}")
    """
    # Get the Page candidate from classification results
    page_candidates = result.get_candidates("page")

    if not page_candidates:
        raise ValueError("No Page found in classification results")

    # There should be exactly one winning Page candidate
    winners = [c for c in page_candidates if c.is_winner]
    if not winners:
        raise ValueError("No winning Page candidate found in classification results")

    if len(winners) > 1:
        raise ValueError(f"Multiple winning Page candidates found: {len(winners)}")

    page = winners[0].constructed
    if not isinstance(page, Page):
        raise ValueError(f"Expected Page element, got {type(page)}")

    return page
