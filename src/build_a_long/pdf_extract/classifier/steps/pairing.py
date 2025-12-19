"""Shared pairing logic for step numbers and diagrams.

This module provides common utilities for pairing step numbers with diagrams,
using Hungarian algorithm matching with configurable scoring. The logic is
shared between:
- StepClassifier (main step number to diagram assignment)
- SubStepClassifier (substep number to diagram pairing)

The pairing follows LEGO instruction conventions:
- Step numbers are typically positioned in the top-left of their associated diagram
- Pairings should not cross divider lines
- Distance-based scoring with position preferences
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

from build_a_long.pdf_extract.extractor.bbox import BBox

log = logging.getLogger(__name__)


# Default maximum distance for pairing (in page units)
DEFAULT_MAX_PAIRING_DISTANCE = 500.0


class PairingConfig(BaseModel, frozen=True):
    """Configuration for step number to diagram pairing.

    Attributes:
        max_distance: Maximum distance between step number and diagram center
            for a valid pairing. Pairs beyond this distance get infinite cost.
        position_weight: Weight for position score (0.0-1.0). Higher values
            prefer diagrams positioned correctly relative to step numbers.
        distance_weight: Weight for distance score (0.0-1.0). Higher values
            prefer closer diagrams.
        check_dividers: Whether to reject pairings that cross dividers.
        top_left_tolerance: Tolerance (in page units) for "top-left" position.
            Step number center should be within this distance of diagram's
            top-left corner to be considered well-positioned.
    """

    max_distance: float = DEFAULT_MAX_PAIRING_DISTANCE
    position_weight: float = 0.5
    distance_weight: float = 0.5
    check_dividers: bool = True
    top_left_tolerance: float = 100.0


def calculate_position_score(
    step_bbox: BBox,
    diagram_bbox: BBox,
    tolerance: float = 100.0,
) -> float:
    """Score based on step number being in the top-left of the diagram.

    In LEGO instructions, step numbers are typically positioned:
    - To the left of the diagram (or left-aligned with it)
    - Above or at the same level as the diagram (or top-aligned with it)

    The score is continuous based on distance from the ideal top-left position,
    with tolerance for alignment cases.

    Args:
        step_bbox: Bounding box of the step number
        diagram_bbox: Bounding box of the diagram
        tolerance: Distance tolerance for position checks. Scores decay
            smoothly within this distance from the ideal position.

    Returns:
        Score between 0.0 and 1.0, based on how well positioned the step
        number is relative to the diagram's top-left.
    """
    step_center_x, step_center_y = step_bbox.center
    diag_x0, diag_y0 = diagram_bbox.x0, diagram_bbox.y0  # Top-left corner
    diag_center_x, diag_center_y = diagram_bbox.center

    # Calculate how far "wrong" the position is for each axis
    # Negative values mean we're in the correct direction
    # (left of center / above center)
    # Positive values mean we're in the wrong direction

    # X axis: ideal is to the left of diagram center
    # x_offset > 0 means step is to the right of diagram center (bad)
    x_offset = step_center_x - diag_center_x

    # Y axis: ideal is above diagram center
    # y_offset > 0 means step is below diagram center (bad)
    y_offset = step_center_y - diag_center_y

    # Calculate x score: 1.0 if left of center, decay if right of center
    if x_offset <= 0:
        # To the left of center - perfect on x axis
        x_score = 1.0
    elif x_offset <= tolerance:
        # Within tolerance to the right - linear decay
        x_score = 1.0 - (x_offset / tolerance) * 0.5  # Decay to 0.5 at tolerance
    else:
        # Beyond tolerance to the right - further decay
        excess = x_offset - tolerance
        x_score = max(0.0, 0.5 - (excess / tolerance) * 0.5)  # Decay to 0.0

    # Calculate y score: 1.0 if above center, decay if below center
    if y_offset <= 0:
        # Above center - perfect on y axis
        y_score = 1.0
    elif y_offset <= tolerance:
        # Within tolerance below - linear decay
        y_score = 1.0 - (y_offset / tolerance) * 0.5  # Decay to 0.5 at tolerance
    else:
        # Beyond tolerance below - further decay
        excess = y_offset - tolerance
        y_score = max(0.0, 0.5 - (excess / tolerance) * 0.5)  # Decay to 0.0

    # Bonus for being near the top-left corner of the diagram
    dist_to_top_left = (
        (step_center_x - diag_x0) ** 2 + (step_center_y - diag_y0) ** 2
    ) ** 0.5

    if dist_to_top_left <= tolerance:
        # Near top-left corner - bonus that decays with distance
        corner_bonus = 0.2 * (1.0 - dist_to_top_left / tolerance)
    else:
        corner_bonus = 0.0

    # Combined score: geometric mean of x and y scores, plus corner bonus
    # Geometric mean ensures both axes need to be reasonable
    base_score = (x_score * y_score) ** 0.5
    final_score = min(1.0, base_score + corner_bonus)

    return final_score


def calculate_distance_score(
    step_bbox: BBox,
    diagram_bbox: BBox,
    max_distance: float = DEFAULT_MAX_PAIRING_DISTANCE,
) -> float:
    """Score based on distance between step number and diagram.

    Uses distance from step number center to the nearest point on the diagram
    bounding box (not center-to-center) for more accurate proximity scoring.

    Args:
        step_bbox: Bounding box of the step number
        diagram_bbox: Bounding box of the diagram
        max_distance: Maximum distance for a non-zero score

    Returns:
        Score between 0.0 and 1.0:
        - 1.0: Step number is at or inside the diagram bbox
        - Linear decay to 0.0 at max_distance
    """
    step_center_x, step_center_y = step_bbox.center

    # Calculate distance to nearest point on diagram bbox
    # Clamp step center to the diagram bbox to find nearest point
    nearest_x = max(diagram_bbox.x0, min(step_center_x, diagram_bbox.x1))
    nearest_y = max(diagram_bbox.y0, min(step_center_y, diagram_bbox.y1))

    # Distance from step center to nearest point on diagram
    distance = (
        (step_center_x - nearest_x) ** 2 + (step_center_y - nearest_y) ** 2
    ) ** 0.5

    if distance > max_distance:
        return 0.0

    # Linear decay from 1.0 at distance=0 to 0.0 at max_distance
    return 1.0 - (distance / max_distance)


def has_divider_between(
    bbox1: BBox,
    bbox2: BBox,
    divider_bboxes: Sequence[BBox] = (),
) -> bool:
    """Check if there is a divider line between two bboxes.

    A divider is considered "between" if it crosses the line segment connecting
    the centers of the two bboxes. Dividers that are contained within
    either bbox are ignored (they are internal dividers, not separating).

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        divider_bboxes: Sequence of divider bounding boxes to check

    Returns:
        True if a divider is between the two bboxes
    """
    if not divider_bboxes:
        return False

    center1 = bbox1.center
    center2 = bbox2.center

    for div_bbox in divider_bboxes:
        # Skip dividers that are fully contained within either bbox
        # (these are internal dividers within a container, not separating)
        if bbox1.contains(div_bbox) or bbox2.contains(div_bbox):
            continue

        # Check if the line segment between centers intersects this divider
        if div_bbox.line_intersects(center1, center2):
            return True

    return False


def calculate_pairing_cost(
    step_bbox: BBox,
    diagram_bbox: BBox,
    config: PairingConfig,
    divider_bboxes: Sequence[BBox] = (),
) -> float:
    """Calculate the cost of pairing a step number with a diagram.

    Lower cost means better pairing. Returns infinity for invalid pairings.

    Args:
        step_bbox: Bounding box of the step number
        diagram_bbox: Bounding box of the diagram
        config: Pairing configuration
        divider_bboxes: Sequence of divider bboxes to check for crossing
            (if config.check_dividers)

    Returns:
        Cost value (lower is better), or infinity for invalid pairings
    """
    # Check for divider crossing first (fast rejection)
    if (
        config.check_dividers
        and divider_bboxes
        and has_divider_between(step_bbox, diagram_bbox, divider_bboxes)
    ):
        return np.inf

    # Calculate component scores
    position_score = calculate_position_score(
        step_bbox, diagram_bbox, config.top_left_tolerance
    )
    distance_score = calculate_distance_score(
        step_bbox, diagram_bbox, config.max_distance
    )

    # If either score is zero, the pairing is invalid
    if position_score <= 0 or distance_score <= 0:
        return np.inf

    # Combined score (higher is better)
    total_score = (
        config.position_weight * position_score
        + config.distance_weight * distance_score
    )

    # Convert to cost (lower is better)
    # Use negative score so Hungarian minimization finds maximum score
    return -total_score


class PairingResult(BaseModel, frozen=True):
    """Result of a step-diagram pairing.

    Attributes:
        step_index: Index of the step number in the input list
        diagram_index: Index of the diagram in the input list
        cost: The pairing cost (lower is better)
        position_score: Position score component (0.0-1.0)
        distance_score: Distance score component (0.0-1.0)
    """

    step_index: int
    diagram_index: int
    cost: float
    position_score: float
    distance_score: float


def find_optimal_pairings(
    step_bboxes: list[BBox],
    diagram_bboxes: list[BBox],
    config: PairingConfig | None = None,
    divider_bboxes: Sequence[BBox] = (),
) -> list[PairingResult]:
    """Find optimal pairings between step numbers and diagrams.

    Uses Hungarian algorithm to find minimum-cost bipartite matching.
    Returns only valid pairings (cost < infinity).

    Args:
        step_bboxes: List of step number bounding boxes
        diagram_bboxes: List of diagram bounding boxes
        config: Pairing configuration (uses defaults if None)
        divider_bboxes: Sequence of divider bboxes to check for crossing

    Returns:
        List of PairingResult objects for valid pairings
    """
    if config is None:
        config = PairingConfig()

    n_steps = len(step_bboxes)
    n_diagrams = len(diagram_bboxes)

    if n_steps == 0 or n_diagrams == 0:
        return []

    # Build cost matrix
    cost_matrix = np.full((n_steps, n_diagrams), np.inf)

    for i, step_bbox in enumerate(step_bboxes):
        for j, diagram_bbox in enumerate(diagram_bboxes):
            cost_matrix[i, j] = calculate_pairing_cost(
                step_bbox, diagram_bbox, config, divider_bboxes
            )

    # Check if we have any valid pairings
    valid_count = np.sum(~np.isinf(cost_matrix))
    if valid_count == 0:
        log.debug("[pairing] No valid step-diagram pairs found")
        return []

    log.debug(
        "[pairing] Found %d valid pairs for %dx%d matrix",
        valid_count,
        n_steps,
        n_diagrams,
    )

    # Run Hungarian algorithm
    # Note: linear_sum_assignment can handle infinite costs
    try:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    except ValueError as e:
        log.debug("[pairing] Hungarian algorithm failed: %s", e)
        return []

    # Collect valid pairings
    results: list[PairingResult] = []
    for row_idx, col_idx in zip(row_indices, col_indices, strict=True):
        cost = cost_matrix[row_idx, col_idx]
        if np.isinf(cost):
            continue

        step_bbox = step_bboxes[row_idx]
        diagram_bbox = diagram_bboxes[col_idx]

        results.append(
            PairingResult(
                step_index=row_idx,
                diagram_index=col_idx,
                cost=float(cost),
                position_score=calculate_position_score(
                    step_bbox, diagram_bbox, config.top_left_tolerance
                ),
                distance_score=calculate_distance_score(
                    step_bbox, diagram_bbox, config.max_distance
                ),
            )
        )

    log.debug("[pairing] Hungarian matching produced %d valid pairings", len(results))
    return results
