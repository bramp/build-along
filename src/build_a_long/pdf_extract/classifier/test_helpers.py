"""Common test helpers for classifier tests."""

from build_a_long.pdf_extract.classifier.classification_result import Candidate
from build_a_long.pdf_extract.classifier.parts.parts_image_classifier import (
    _PartImageScore,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks


def make_candidates(
    labeled_blocks: dict[Blocks, str],
    part_image_pairs: list[tuple[Blocks, Blocks]] | None = None,
) -> dict[str, list[Candidate]]:
    """Helper to create candidates from labeled blocks.

    Args:
        labeled_blocks: Dictionary mapping blocks to their labels
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
    image_to_count: dict[Blocks, Blocks] = {}
    for part_count, part_image in part_image_pairs:
        image_to_count[part_image] = part_count

    for block, label in labeled_blocks.items():
        if label not in candidates:
            candidates[label] = []

        # For part_image candidates, include the pairing in score_details
        if label == "part_image" and block in image_to_count:
            part_count = image_to_count[block]
            score_details = _PartImageScore(
                distance=0.0,
                part_count=part_count,  # type: ignore
                image=block,  # type: ignore
            )
        else:
            score_details = {}

        candidates[label].append(
            Candidate(
                bbox=block.bbox,
                label=label,
                score=1.0,
                score_details=score_details,
                source_blocks=[block],
            )
        )
    return candidates
