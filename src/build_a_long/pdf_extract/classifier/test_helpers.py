"""Common test helpers for classifier tests."""

from build_a_long.pdf_extract.classifier.classification_result import Candidate
from build_a_long.pdf_extract.classifier.parts_image_classifier import _PartImageScore
from build_a_long.pdf_extract.extractor.page_elements import Element


def make_candidates(
    labeled_elements: dict[Element, str],
    part_image_pairs: list[tuple[Element, Element]] | None = None,
) -> dict[str, list[Candidate]]:
    """Helper to create candidates from labeled elements.

    Args:
        labeled_elements: Dictionary mapping elements to their labels
        part_image_pairs: Optional list of (part_count, part_image) tuples.
            If provided, part_image candidates will include the pairing
            information in their score_details.

    Returns:
        Dictionary of candidates by label
    """
    if part_image_pairs is None:
        part_image_pairs = []

    candidates: dict[str, list[Candidate]] = {}

    # Create a mapping from part_image to part_count for lookup
    image_to_count: dict[Element, Element] = {}
    for part_count, part_image in part_image_pairs:
        image_to_count[part_image] = part_count

    for element, label in labeled_elements.items():
        if label not in candidates:
            candidates[label] = []

        # For part_image candidates, include the pairing in score_details
        if label == "part_image" and element in image_to_count:
            part_count = image_to_count[element]
            score_details = _PartImageScore(
                distance=0.0,
                part_count=part_count,  # type: ignore
                image=element,  # type: ignore
            )
        else:
            score_details = {}

        candidates[label].append(
            Candidate(
                bbox=element.bbox,
                label=label,
                score=1.0,
                score_details=score_details,
                constructed=None,
                source_element=element,
                is_winner=True,
            )
        )
    return candidates
