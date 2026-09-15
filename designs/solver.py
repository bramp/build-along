"""
Backtracking solver for finding the optimal set of non-conflicting candidates.
"""

from collections.abc import Callable, Hashable
from typing import TypeVar

from build_a_long.pdf_extract.classifier.candidate import Candidate

T = TypeVar("T")


class Solver:
    """
    Finds the subset of candidates that maximizes the total score while satisfying
    conflict constraints.

    This is essentially a Maximum Weight Independent Set solver on the conflict graph.
    """

    def __init__(
        self,
        candidates: list[Candidate],
        get_conflicts: Callable[[Candidate], set[Hashable]],
    ):
        """
        Args:
            candidates: List of candidates to consider.
            get_conflicts: Function that returns a set of conflict identifiers
                           for a candidate. If two candidates share any identifier,
                           they are considered conflicting.
        """
        # Sort candidates by score descending for better branch-and-bound pruning
        self.candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        self.get_conflicts = get_conflicts
        self.best_score = -1.0
        self.best_solution: list[Candidate] = []

        # Pre-compute conflicts for each candidate to avoid repeated calls
        self.candidate_conflicts = [get_conflicts(c) for c in self.candidates]

        # Pre-compute suffix sums of scores for upper-bound estimation
        self.suffix_max_scores = [0.0] * (len(self.candidates) + 1)
        current_sum = 0.0
        for i in range(len(self.candidates) - 1, -1, -1):
            current_sum += self.candidates[i].score
            self.suffix_max_scores[i] = current_sum

    def solve(self) -> list[Candidate]:
        """Run the solver."""
        self.best_score = -1.0
        self.best_solution = []
        self._search(0, 0.0, set(), [])
        return self.best_solution

    def _search(
        self,
        index: int,
        current_score: float,
        consumed_conflicts: set[Hashable],
        current_path: list[Candidate],
    ) -> None:
        """Recursive backtracking search with pruning."""
        # Base case: checked all candidates
        if index == len(self.candidates):
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_solution = list(current_path)
            return

        # Pruning: Upper bound check
        # If current score + max possible remaining score <= best found so far,
        # we can't beat the best score.
        if current_score + self.suffix_max_scores[index] <= self.best_score:
            return

        candidate = self.candidates[index]
        conflicts = self.candidate_conflicts[index]

        # Option 1: Try including this candidate (if no conflicts)
        if not conflicts.intersection(consumed_conflicts):
            # Recurse with candidate included
            # We create a new set for consumed_conflicts only when necessary
            # to avoid excessive copying.
            # Python sets are mutable, so we can track changes or copy.
            # Copying is safer/simpler for now.
            new_consumed = consumed_conflicts | conflicts
            current_path.append(candidate)
            self._search(
                index + 1, current_score + candidate.score, new_consumed, current_path
            )
            current_path.pop()  # Backtrack

        # Option 2: Exclude this candidate
        # We always explore this branch (unless pruned)
        self._search(index + 1, current_score, consumed_conflicts, current_path)
