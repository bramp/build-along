"""Constraint satisfaction model for classification candidate selection.

This module provides a wrapper around OR-Tools CP-SAT solver for selecting
the optimal combination of classification candidates that satisfy all constraints
while maximizing an objective function.

Design Decisions:
-----------------
1. Expose raw CP-SAT model (self.model) to allow power users to add custom
   constraints beyond our helper methods.
2. Use Candidate.id (auto-generated unique ID) as key for variable lookup.
3. Provide high-level helpers for common patterns (at_most_one_of, if_selected_then).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ortools.sat.python import cp_model

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.candidate import Candidate

log = logging.getLogger(__name__)


class ConstraintModel:
    """Wrapper around OR-Tools CP-SAT model for classification.

    Provides high-level API for declaring constraints on candidate selection.
    Each candidate is represented as a boolean variable (0=not selected, 1=selected).

    The model finds an assignment that:
    - Satisfies all constraints
    - Maximizes the objective function (typically sum of candidate scores)

    Example:
        model = ConstraintModel()

        # Register candidates
        for candidate in candidates:
            model.add_candidate(candidate)

        # Add constraints
        model.at_most_one_of(conflicting_candidates)
        model.if_selected_then(parent, children)

        # Set objective and solve (scale float scores to int weights 0-1000)
        model.maximize([(c, int(c.score * 1000)) for c in candidates])
        success, selection = model.solve()

        if success:
            selected_ids = {cid for cid, selected in selection.items() if selected}
    """

    def __init__(self) -> None:
        """Initialize the constraint model."""
        self.model = cp_model.CpModel()
        """The underlying CP-SAT model. Exposed for advanced constraint addition."""

        self._candidate_vars: dict[int, cp_model.IntVar] = {}
        """Map from candidate ID to boolean variable."""

        self._candidates: dict[int, Candidate] = {}
        """Map from candidate ID to candidate object (for reference)."""

        self._solver = cp_model.CpSolver()
        """The solver instance (reused across solve calls)."""

    def add_candidate(self, candidate: Candidate) -> cp_model.IntVar:
        """Register a candidate and return its boolean variable.

        Creates a boolean variable for this candidate if not already registered.
        The variable represents whether the candidate is selected (1) or not (0).

        Args:
            candidate: The candidate to register

        Returns:
            The boolean variable representing this candidate's selection
        """
        cid = candidate.id

        if cid not in self._candidate_vars:
            var = self.model.NewBoolVar(f"{candidate.label}_{cid}")
            self._candidate_vars[cid] = var
            self._candidates[cid] = candidate

            log.debug(
                "Registered candidate: %s at %s (id=%d)",
                candidate.label,
                candidate.bbox,
                cid,
            )

        return self._candidate_vars[cid]

    def get_var(self, candidate: Candidate) -> cp_model.IntVar:
        """Get the boolean variable for a candidate.

        Args:
            candidate: The candidate to look up

        Returns:
            The boolean variable for this candidate

        Raises:
            KeyError: If candidate not registered via add_candidate()
        """
        cid = candidate.id
        if cid not in self._candidate_vars:
            raise KeyError(
                f"Candidate {candidate.label} at {candidate.bbox} not registered. "
                f"Call add_candidate() first."
            )
        return self._candidate_vars[cid]

    def has_candidate(self, candidate: Candidate) -> bool:
        """Check if a candidate is registered in the model.

        Args:
            candidate: The candidate to check

        Returns:
            True if the candidate was added via add_candidate(), False otherwise
        """
        return candidate.id in self._candidate_vars

    def at_most_one_of(self, candidates: list[Candidate]) -> None:
        """Add constraint: at most one of these candidates can be selected.

        Use this for mutually exclusive candidates (e.g., multiple candidates
        sharing blocks, or alternative variants for the same element).

        Args:
            candidates: List of mutually exclusive candidates
        """
        if len(candidates) <= 1:
            return  # Constraint is trivially satisfied

        vars_list = [self.get_var(c) for c in candidates]
        self.model.Add(sum(vars_list) <= 1)

        log.debug(
            "Added constraint: at_most_one_of %d candidates (%s)",
            len(candidates),
            ", ".join(c.label for c in candidates[:3]),
        )

    def exactly_one_of(self, candidates: list[Candidate]) -> None:
        """Add constraint: exactly one of these candidates must be selected.

        Use this when one (and only one) alternative must be chosen.

        Args:
            candidates: List of alternative candidates
        """
        if not candidates:
            raise ValueError("exactly_one_of requires at least one candidate")

        vars_list = [self.get_var(c) for c in candidates]
        self.model.Add(sum(vars_list) == 1)

        log.debug("Added constraint: exactly_one_of %d candidates", len(candidates))

    def if_selected_then(self, parent: Candidate, children: list[Candidate]) -> None:
        """Add constraint: if parent selected, all children must be selected.

        Use this for required parent-child dependencies.

        Args:
            parent: The parent candidate
            children: List of required child candidates
        """
        if not children:
            return

        parent_var = self.get_var(parent)

        for child in children:
            child_var = self.get_var(child)
            # If parent=1, then child=1
            self.model.Add(child_var == 1).OnlyEnforceIf(parent_var)

        log.debug(
            "Added constraint: if %s selected, then %d children must be selected",
            parent.label,
            len(children),
        )

    def if_any_selected_then_one_of(
        self, group_a: list[Candidate], group_b: list[Candidate]
    ) -> None:
        """Add constraint: if any in group_a selected, at least one in group_b.

        Use this for "no orphans" constraints. For example:
        - If any arrow is selected, at least one step must be selected
        - If any rotation symbol selected, at least one step must be selected

        Args:
            group_a: Candidates that require group_b
            group_b: Candidates that must have at least one member if group_a selected
        """
        if not group_a or not group_b:
            return

        vars_a = [self.get_var(c) for c in group_a]
        vars_b = [self.get_var(c) for c in group_b]

        # Create indicator: any_a = (sum(vars_a) > 0)
        any_a = self.model.NewBoolVar("any_a_indicator")
        self.model.Add(sum(vars_a) > 0).OnlyEnforceIf(any_a)
        self.model.Add(sum(vars_a) == 0).OnlyEnforceIf(any_a.Not())

        # If any_a, then at least one in group_b
        self.model.Add(sum(vars_b) >= 1).OnlyEnforceIf(any_a)

        log.debug(
            "Added constraint: if any of %d %s selected, "
            "then at least one of %d %s required",
            len(group_a),
            group_a[0].label if group_a else "?",
            len(group_b),
            group_b[0].label if group_b else "?",
        )

    def mutually_exclusive(
        self, candidate_a: Candidate, candidate_b: Candidate
    ) -> None:
        """Add constraint: these two candidates cannot both be selected.

        Equivalent to at_most_one_of([candidate_a, candidate_b]).

        Args:
            candidate_a: First candidate
            candidate_b: Second candidate
        """
        var_a = self.get_var(candidate_a)
        var_b = self.get_var(candidate_b)
        self.model.Add(var_a + var_b <= 1)

        log.debug(
            "Added constraint: %s and %s are mutually exclusive",
            candidate_a.label,
            candidate_b.label,
        )

    def add_block_exclusivity_constraints(
        self, all_candidates: list[Candidate]
    ) -> None:
        """Add constraints: blocks can only be used by one candidate.

        This is the fundamental constraint that prevents candidates from
        consuming the same blocks. For each block, at most one candidate
        using that block can be selected.

        Args:
            all_candidates: All candidates in the classification
        """
        # Group candidates by block
        block_to_candidates: dict[int, list[Candidate]] = {}

        for candidate in all_candidates:
            for block in candidate.source_blocks:
                if block.id not in block_to_candidates:
                    block_to_candidates[block.id] = []
                block_to_candidates[block.id].append(candidate)

        # Add at_most_one constraint for each block with multiple candidates
        constraint_count = 0
        for _block_id, candidates in block_to_candidates.items():
            if len(candidates) > 1:
                self.at_most_one_of(candidates)
                constraint_count += 1

        log.info(
            "Added %d block exclusivity constraints for %d blocks",
            constraint_count,
            len(block_to_candidates),
        )

    def maximize(self, objective_terms: list[tuple[Candidate, int]]) -> None:
        """Set objective: maximize sum of (candidate_var * weight).

        CP-SAT requires integer coefficients. Weights should be in the range
        0-1000 (or similar scale). Callers should scale float scores before
        passing them here.

        Args:
            objective_terms: List of (candidate, weight) tuples where weight
                is an integer (typically 0-1000)
        """
        terms = []
        for candidate, weight in objective_terms:
            var = self.get_var(candidate)
            terms.append(var * weight)

        self.model.Maximize(sum(terms))

        log.debug("Set objective: maximize sum of %d terms", len(objective_terms))

    def solve(self) -> tuple[bool, dict[int, bool]]:
        """Solve the constraint satisfaction problem.

        Returns:
            Tuple of (success, selection) where:
            - success: True if feasible solution found
            - selection: Map from candidate_id to whether selected (True/False)
        """
        # Configure solver
        self._solver.parameters.log_search_progress = False

        # Solve
        log.info(
            "Solving constraint model with %d candidates...", len(self._candidates)
        )
        status = self._solver.Solve(self.model)

        # Extract results
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            selection = {
                cid: self._solver.Value(var) == 1
                for cid, var in self._candidate_vars.items()
            }

            selected_count = sum(1 for s in selection.values() if s)
            log.info(
                "Solver succeeded: status=%s, selected %d/%d candidates",
                "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                selected_count,
                len(self._candidates),
            )

            return True, selection

        else:
            # Get status name (handle both old and new API)
            try:
                status_name = cp_model.OPTIMAL  # Try to access to check API
                status_name = cp_model.CpSolverStatus_Name(status)  # type: ignore[attr-defined]
            except AttributeError:
                status_name = str(status)

            log.error("Solver failed: status=%s", status_name)
            return False, {}

    def get_candidates_by_label(self, label: str) -> list[Candidate]:
        """Get all registered candidates with the given label.

        Utility method for constraint declaration.

        Args:
            label: The label to filter by

        Returns:
            List of candidates with this label
        """
        return [c for c in self._candidates.values() if c.label == label]
