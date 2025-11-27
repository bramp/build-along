"""
Base class for label classifiers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElements
from build_a_long.pdf_extract.extractor.page_blocks import Text

# TODO Maybe classifers need a interface, where they have
#      either scoring functions, or filter functions.
#      * Expected a page number, filter text that isn't numeric.
#      * Expected the number in the corner, score based on position.
#      Then we can abstract out common code/functions, to keep the code DRY.


@dataclass(frozen=True)
class LabelClassifier(ABC):
    """Abstract base class for a single label classifier.

    Classifiers are frozen dataclasses to enforce statelessness - they cannot
    modify their attributes after initialization. All state must be stored in
    ClassificationResult.
    """

    config: ClassifierConfig

    # Class-level metadata to declare pipeline dependencies.
    # Subclasses should override these at the class level
    outputs: ClassVar[frozenset[str]] = frozenset()
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

    @abstractmethod
    def score(self, result: ClassificationResult) -> None:
        """Score blocks and create candidates.

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
        pass

    @abstractmethod
    def construct(self, result: ClassificationResult) -> None:
        """Construct LegoPageElements from candidates.

        This method should:
        1. Get candidates for each label this classifier outputs
        2. Select which candidates to construct (all, winners only, etc.)
        3. Build LegoPageElements from selected candidates
        4. Update candidate.constructed and candidate.failure_reason

        Default pattern for constructing all candidates:
            for label in self.outputs:
                candidates = result.get_candidates(label)
                for candidate in candidates:
                    try:
                        elem = self._construct_single(candidate, result)
                        candidate.constructed = elem
                    except Exception as e:
                        candidate.failure_reason = str(e)

        This is the second phase of the two-phase classification process.
        Scoring happens first in score().

        Args:
            result: The classification result containing candidates to construct
        """
        pass

    @abstractmethod
    def construct_candidate(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Construct a single candidate into a LegoPageElement.

        This method is called during the top-down construction phase.
        It should:
        1. Resolve any dependencies (e.g. ask result to construct parent candidates)
        2. Build the LegoPageElement
        3. Return it (do not set candidate.constructed, the caller does that)

        Args:
            candidate: The candidate to construct
            result: The classification result context

        Returns:
            The constructed LegoPageElement
        """
        pass
