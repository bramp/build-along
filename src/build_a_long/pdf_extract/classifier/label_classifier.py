"""
Base class for label classifiers.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElements
from build_a_long.pdf_extract.extractor.page_blocks import Text

# TODO Maybe classifers need a interface, where they have
#      either scoring functions, or filter functions.
#      * Expected a page number, filter text that isn't numeric.
#      * Expected the number in the corner, score based on position.
#      Then we can abstract out common code/functions, to keep the code DRY.


class LabelClassifier(BaseModel, ABC):
    """Abstract base class for a single label classifier.

    Classifiers are frozen Pydantic models to enforce statelessness - they cannot
    modify their attributes after initialization. All state must be stored in
    ClassificationResult.
    """

    model_config = ConfigDict(frozen=True)

    config: ClassifierConfig

    # Class-level metadata to declare pipeline dependencies.
    # Subclasses should override these at the class level
    output: ClassVar[str] = ""
    requires: ClassVar[frozenset[str]] = frozenset()

    def _score_font_size(self, block: Text, target_size: float | None) -> float:
        """Score how well text font size matches target size."""
        if target_size is None:
            return 0.5

        if block.font_size is None:
            # Fallback to bbox height if font size not explicitly set
            size = block.bbox.height
        else:
            size = block.font_size

        # Calculate difference as ratio of target size
        if target_size == 0:
            return 0.0

        diff_ratio = abs(size - target_size) / target_size

        # Linear penalty: score = 1.0 - (diff_ratio * 2.0)
        return max(0.0, 1.0 - (diff_ratio * 2.0))

    def score(self, result: ClassificationResult) -> None:
        """Score blocks and create candidates.

        This method automatically registers the classifier for all labels in
        self.output, then calls _score() to perform the actual scoring.

        Subclasses should implement _score() instead of overriding this method.

        This method should:
        1. Score each blocks for this label
        2. Create Candidate objects with scores and score_details
        3. Store candidates in the result via result.add_candidate()

        **For classifiers that depend on other classifiers:**

        Use result.get_scored_candidates() to get parent candidates, then
        store references to those candidates (not their constructed elements)
        in your score_details:

            # CORRECT - store candidate references
            parent_candidates = result.get_scored_candidates("parent_label")
            for parent_cand in parent_candidates:
                score_details = MyScore(
                    parent_candidate=parent_cand,  # Store the candidate!
                    ...
                )

        This ensures your classifier works with candidates and preserves the
        dependency chain. During construct(), you can then validate that parent
        candidates are still winners before using their constructed elements.

        This is the first phase of the two-phase classification process.
        Construction happens later in construct().

        Args:
            result: The classification result to populate with candidates
        """
        # Auto-register this classifier for its output label
        result._register_classifier(self.output, self)

        # Call the subclass implementation
        self._score(result)

    @abstractmethod
    def _score(self, result: ClassificationResult) -> None:
        """Perform the actual scoring logic.

        Subclasses should implement this method instead of score().
        See score() documentation for details on what to implement.

        Args:
            result: The classification result to populate with candidates
        """
        pass

    def build_all(self, result: ClassificationResult) -> None:
        """Build LegoPageElements from all candidates for this classifier's label.

        Iterates through all candidates for this classifier's output label
        and calls build() on each, storing the result or failure reason.

        This is the entry point called by the classification pipeline to trigger
        construction for a specific classifier.

        Args:
            result: The classification result containing candidates to build
        """
        candidates = result.get_candidates(self.output)
        for candidate in candidates:
            try:
                elem = result.build(candidate)
                candidate.constructed = elem
            except Exception as e:
                candidate.failure_reason = str(e)

    @abstractmethod
    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Construct a single candidate into a LegoPageElement.

        This method is called during the top-down construction phase.
        It should:
        1. Resolve any dependencies (e.g. ask result to construct parent candidates)
        2. Build the LegoPageElement
        3. Return it (do not set candidate.constructed, the caller does that)

        IMPORTANT: Classifiers should only create elements they are responsible for.
        A classifier should NOT create elements of types handled by other classifiers.
        For example:
        - StepClassifier should NOT create PartsList elements (that's PartsListClassifier's job)
        - If a dependency is optional and not found, set it to None instead of creating a fallback

        If you find yourself creating an element type that another classifier handles,
        that's a design smell - either:
        1. Make the field optional in the parent element, OR
        2. Ensure the dependency classifier runs first and provides the element

        Args:
            candidate: The candidate to construct
            result: The classification result context

        Returns:
            The constructed LegoPageElement
        """
        pass

    def rescore_without_blocks(
        self,
        candidate: Candidate,
        excluded_block_ids: set[int],
        result: ClassificationResult,
    ) -> Candidate | None:
        """Create a new candidate excluding specified blocks.

        This method is called when a candidate's source blocks conflict with
        another candidate that won. Instead of failing the candidate entirely,
        we try to create a reduced version without the conflicting blocks.

        The default implementation returns None (no re-scoring support).
        Subclasses can override to provide graceful degradation.

        Args:
            candidate: The original candidate to re-score
            excluded_block_ids: Set of block IDs to exclude
            result: The classification result context

        Returns:
            A new candidate without the excluded blocks, or None if the
            candidate is no longer valid without those blocks.
        """
        return None
