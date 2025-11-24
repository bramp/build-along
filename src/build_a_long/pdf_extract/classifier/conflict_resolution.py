"""
Conflict resolution for blocks with multiple label candidates.

When a single block (e.g., a Text element) is claimed by multiple classifiers
with different labels, we need to determine which label should "win". This module
implements centralized conflict resolution logic using a priority system.

Example conflicts:
- A "4" in a circle could be piece_length OR step_number
- A number in the corner could be page_number OR step_number
- Text could be part_count OR step_number

Resolution uses label priority (some labels are inherently more specific than others).
"""

from __future__ import annotations

import logging

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
)

logger = logging.getLogger(__name__)


# Label priority order (lower number = higher priority)
# More specific labels have higher priority
LABEL_PRIORITY: dict[str, int] = {
    "page_number": 1,  # Very specific, bottom corners only
    "bag_number": 2,  # Specific section markers
    "piece_length": 3,  # Specific, appears in circles
    "part_number": 4,  # Catalog-specific
    "part_count": 5,  # Part list specific
    "progress_bar": 6,  # Visual indicator
    "step_number": 7,  # Common but less specific
    "new_bag": 8,  # Derived element
    "part": 9,  # Composite element
    "parts_list": 10,  # Container element
    "part_image": 11,  # Association metadata
    "diagram": 12,  # Large regions
    "step": 13,  # Composite element
    "page": 14,  # Top-level composite
}


def resolve_label_conflicts(result: ClassificationResult) -> None:
    """Resolve conflicts where single blocks have multiple label candidates.

    For each block that appears in multiple candidates with different labels,
    applies domain-specific rules to determine which label should win. Losing
    candidates are marked with failure_reason.

    This function modifies the ClassificationResult in-place by setting
    failure_reason on losing candidates.

    Args:
        result: The classification result containing candidates to resolve
    """
    # Build a mapping from block to list of (label, candidate) tuples
    block_to_candidates: dict[int, list[tuple[str, Candidate]]] = {}

    # Collect all candidates organized by their source blocks
    all_candidates = result.get_all_candidates()
    for label, candidates in all_candidates.items():
        for candidate in candidates:
            # Only consider successfully constructed candidates
            if candidate.constructed is None:
                continue

            # Track each source block
            for block in candidate.source_blocks:
                block_id = id(block)
                if block_id not in block_to_candidates:
                    block_to_candidates[block_id] = []
                block_to_candidates[block_id].append((label, candidate))

    # Find blocks with conflicts (multiple different labels)
    conflicts_resolved = 0
    for _block_id, label_candidate_pairs in block_to_candidates.items():
        # Get unique labels for this block
        labels = {label for label, _ in label_candidate_pairs}

        if len(labels) <= 1:
            # No conflict - all candidates have the same label
            continue

        # We have a conflict! Resolve it
        winner_candidate, reason = _resolve_conflict(labels, label_candidate_pairs)
        winner_label = next(
            label for label, cand in label_candidate_pairs if cand is winner_candidate
        )

        # Mark losing candidates
        for _label, candidate in label_candidate_pairs:
            if candidate is not winner_candidate and candidate.failure_reason is None:
                candidate.failure_reason = (
                    f"Conflict resolution: {reason}. Winner: {winner_label}"
                )
                # Keep constructed element for debugging, but mark as failed
                conflicts_resolved += 1

        if logger.isEnabledFor(logging.DEBUG):
            labels_str = ", ".join(sorted(labels))
            logger.debug(
                f"Resolved conflict for block: "
                f"labels=[{labels_str}], winner={winner_label}"
            )

    if conflicts_resolved > 0:
        logger.info(f"Resolved {conflicts_resolved} label conflicts")


def _resolve_conflict(
    labels: set[str], label_candidate_pairs: list[tuple[str, Candidate]]
) -> tuple[Candidate, str]:
    """Resolve a conflict between multiple labels for the same block.

    Currently uses LABEL_PRIORITY to select the winner. Logs when conflicts occur
    so we can build up domain-specific rules over time.

    Args:
        labels: Set of conflicting labels
        label_candidate_pairs: List of (label, candidate) tuples for this block

    Returns:
        Tuple of (winning_candidate, reason)
    """

    # TODO Add more complex resolution logic here in the future

    # Use priority system to select winner
    # Find the label with highest priority (lowest priority number)
    winner_label = min(labels, key=lambda label: LABEL_PRIORITY.get(label, 999))
    winner_priority = LABEL_PRIORITY.get(winner_label, 999)

    # Find the candidate with the winning label
    winner_candidate = next(
        candidate for label, candidate in label_candidate_pairs if label == winner_label
    )

    # Build reason
    reason = f"'{winner_label}' has priority {winner_priority}"

    # Log the conflict for future analysis
    labels_str = ", ".join(sorted(labels))
    logger.warning(
        f"Conflict resolved using priority: winner={winner_label} "
        f"(priority={winner_priority}), all_labels=[{labels_str}]"
    )

    return winner_candidate, reason
