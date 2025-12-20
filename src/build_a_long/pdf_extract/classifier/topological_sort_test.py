"""Tests for topological sorting of classifiers."""

from typing import ClassVar

import pytest

from build_a_long.pdf_extract.classifier import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.topological_sort import topological_sort
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElements


# Test classifiers with various dependency patterns
class TestClassifierA(LabelClassifier):
    """Test classifier with no dependencies."""

    output: ClassVar[str] = "a"
    requires: ClassVar[frozenset[str]] = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        pass

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        raise NotImplementedError


class TestClassifierB(LabelClassifier):
    """Test classifier that depends on A."""

    output: ClassVar[str] = "b"
    requires: ClassVar[frozenset[str]] = frozenset({"a"})

    def _score(self, result: ClassificationResult) -> None:
        pass

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        raise NotImplementedError


class TestClassifierC(LabelClassifier):
    """Test classifier that depends on B."""

    output: ClassVar[str] = "c"
    requires: ClassVar[frozenset[str]] = frozenset({"b"})

    def _score(self, result: ClassificationResult) -> None:
        pass

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        raise NotImplementedError


class TestClassifierD(LabelClassifier):
    """Test classifier that depends on A and B."""

    output: ClassVar[str] = "d"
    requires: ClassVar[frozenset[str]] = frozenset({"a", "b"})

    def _score(self, result: ClassificationResult) -> None:
        pass

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        raise NotImplementedError


class TestClassifierDuplicate(LabelClassifier):
    """Test classifier with duplicate output label."""

    output: ClassVar[str] = "a"  # Duplicate of TestClassifierA
    requires: ClassVar[frozenset[str]] = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        pass

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        raise NotImplementedError


class TestClassifierX(LabelClassifier):
    """Test classifier for circular dependency: X -> Y."""

    output: ClassVar[str] = "x"
    requires: ClassVar[frozenset[str]] = frozenset({"y"})

    def _score(self, result: ClassificationResult) -> None:
        pass

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        raise NotImplementedError


class TestClassifierY(LabelClassifier):
    """Test classifier for circular dependency: Y -> Z."""

    output: ClassVar[str] = "y"
    requires: ClassVar[frozenset[str]] = frozenset({"z"})

    def _score(self, result: ClassificationResult) -> None:
        pass

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        raise NotImplementedError


class TestClassifierZ(LabelClassifier):
    """Test classifier for circular dependency: Z -> X."""

    output: ClassVar[str] = "z"
    requires: ClassVar[frozenset[str]] = frozenset({"x"})

    def _score(self, result: ClassificationResult) -> None:
        pass

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        raise NotImplementedError


class TestTopologicalSort:
    """Tests for the topological_sort function."""

    def test_simple_chain(self) -> None:
        """Test sorting a simple dependency chain: A -> B -> C."""
        config = ClassifierConfig()
        classifiers: list[LabelClassifier] = [
            TestClassifierC(config=config),  # Out of order
            TestClassifierA(config=config),
            TestClassifierB(config=config),
        ]

        sorted_classifiers = topological_sort(classifiers)

        # Verify the order: A must come before B, B must come before C
        outputs = [c.output for c in sorted_classifiers]
        assert outputs.index("a") < outputs.index("b")
        assert outputs.index("b") < outputs.index("c")

    def test_diamond_dependency(self) -> None:
        """Test sorting with diamond dependency: A -> B,D and B -> C and D -> C.

        Dependency graph:
            A
           / \\
          B   (implied)
          |
          C
          |
          D (depends on A and B)
        """
        config = ClassifierConfig()
        classifiers: list[LabelClassifier] = [
            TestClassifierD(config=config),
            TestClassifierC(config=config),
            TestClassifierB(config=config),
            TestClassifierA(config=config),
        ]

        sorted_classifiers = topological_sort(classifiers)

        # Verify dependencies are satisfied
        outputs = [c.output for c in sorted_classifiers]
        # A must come before B and D
        assert outputs.index("a") < outputs.index("b")
        assert outputs.index("a") < outputs.index("d")
        # B must come before C and D
        assert outputs.index("b") < outputs.index("c")
        assert outputs.index("b") < outputs.index("d")

    def test_no_dependencies(self) -> None:
        """Test sorting classifiers with no dependencies."""
        config = ClassifierConfig()
        classifiers: list[LabelClassifier] = [TestClassifierA(config=config)]

        sorted_classifiers = topological_sort(classifiers)

        assert len(sorted_classifiers) == 1
        assert sorted_classifiers[0].output == "a"

    def test_duplicate_output_labels(self) -> None:
        """Test that duplicate output labels raise an error."""
        config = ClassifierConfig()
        classifiers: list[LabelClassifier] = [
            TestClassifierA(config=config),
            TestClassifierDuplicate(config=config),  # Same output as A
        ]

        with pytest.raises(ValueError, match="Duplicate output label 'a'"):
            topological_sort(classifiers)

    def test_empty_list(self) -> None:
        """Test sorting an empty list."""
        sorted_classifiers = topological_sort([])
        assert sorted_classifiers == []

    def test_all_dependencies_preserved(self) -> None:
        """Test that all classifiers are preserved after sorting."""
        config = ClassifierConfig()
        classifiers: list[LabelClassifier] = [
            TestClassifierD(config=config),
            TestClassifierC(config=config),
            TestClassifierB(config=config),
            TestClassifierA(config=config),
        ]

        sorted_classifiers = topological_sort(classifiers)

        # All classifiers should be present
        assert len(sorted_classifiers) == len(classifiers)
        assert set(c.output for c in sorted_classifiers) == {"a", "b", "c", "d"}

    def test_circular_dependency(self) -> None:
        """Test that circular dependencies are detected and reported with a chain."""
        config = ClassifierConfig()
        classifiers: list[LabelClassifier] = [
            TestClassifierX(config=config),  # x -> y
            TestClassifierY(config=config),  # y -> z
            TestClassifierZ(config=config),  # z -> x (creates cycle)
        ]

        with pytest.raises(ValueError) as exc_info:
            topological_sort(classifiers)

        error_msg = str(exc_info.value)
        # Check that it mentions circular dependency
        assert "Circular dependency detected" in error_msg
        # Check that it includes a dependency chain
        assert "Dependency chain:" in error_msg
        # Check that the chain includes the cycle (order may vary)
        assert "->" in error_msg
