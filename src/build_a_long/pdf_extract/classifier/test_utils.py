"""Test utilities for classifier tests.

This module provides helper classes and functions for testing classifiers.
"""

from build_a_long.pdf_extract.classifier.classification_result import Score, Weight


class TestScore(Score):
    """Simple Score implementation for testing purposes.

    This provides a minimal Score implementation that can be used in unit tests
    where the actual score calculation logic is not relevant.
    """

    value: float = 1.0
    """The score value (default 1.0, must be in range 0.0-1.0)."""

    def score(self) -> Weight:
        """Return the configured score value (0.0-1.0)."""
        return self.value
