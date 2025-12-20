"""Topological sorting utilities for classifiers.

This module provides functionality to sort classifiers based on their dependencies,
ensuring that each classifier appears after all classifiers it depends on.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier

logger = logging.getLogger(__name__)


def topological_sort(
    classifiers: Sequence[LabelClassifier],
) -> Sequence[LabelClassifier]:
    """Sort classifiers topologically based on their dependencies.

    Ensures that classifiers are ordered such that each classifier appears
    after all the classifiers it depends on. Uses Kahn's algorithm for
    topological sorting.

    Args:
        classifiers: List of classifiers to sort

    Returns:
        Topologically sorted list of classifiers

    Raises:
        ValueError: If a circular dependency is detected or if duplicate
            output labels are found
    """
    # Build a mapping from output label to classifier
    label_to_classifier: dict[str, LabelClassifier] = {}
    for classifier in classifiers:
        if classifier.output:
            if classifier.output in label_to_classifier:
                raise ValueError(
                    f"Duplicate output label '{classifier.output}' found in "
                    f"{type(classifier).__name__} and "
                    f"{type(label_to_classifier[classifier.output]).__name__}"
                )
            label_to_classifier[classifier.output] = classifier

    # Calculate in-degree (number of dependencies) for each classifier
    # Use id() as key since classifiers may not be hashable
    in_degree: dict[int, int] = {id(c): len(c.requires) for c in classifiers}

    # Find classifiers with no dependencies
    queue: list[LabelClassifier] = [c for c in classifiers if in_degree[id(c)] == 0]

    # Process classifiers in dependency order
    sorted_classifiers: list[LabelClassifier] = []
    while queue:
        # Pop classifier with no remaining dependencies
        current = queue.pop(0)
        sorted_classifiers.append(current)

        # For each classifier that depends on this one, decrement in-degree
        for classifier in classifiers:
            if current.output in classifier.requires:
                in_degree[id(classifier)] -= 1
                if in_degree[id(classifier)] == 0:
                    queue.append(classifier)

    # Check if all classifiers were processed (no cycles)
    if len(sorted_classifiers) != len(classifiers):
        # Find classifiers with non-zero in-degree (part of cycle)
        cyclic = [c for c in classifiers if in_degree[id(c)] > 0]
        cyclic_names = [type(c).__name__ for c in cyclic]
        raise ValueError(
            f"Circular dependency detected among classifiers: {cyclic_names}"
        )

    logger.debug(
        "Topologically sorted classifiers: %s",
        [type(c).__name__ for c in sorted_classifiers],
    )

    return sorted_classifiers
