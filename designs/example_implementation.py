"""
Proof-of-Concept Implementation for Global Solver Architecture.

This script demonstrates:
1.  Hierarchical Conflict Resolution (Part vs PartCount).
2.  Subtractive Candidate Generation (Diagram vs Arrow).
3.  The Global Solver (Backtracking MWIS).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- 1. Data Structures ---
@dataclass(frozen=True)
class Block:
    id: str
    type: str  # 'text', 'image'

    def __repr__(self):
        return f"[{self.id}]"


@dataclass
class Candidate:
    id: str
    label: str
    score: float
    source_blocks: set[Block]
    children: list[Candidate] = field(default_factory=list)

    def __repr__(self):
        blocks = ",".join(b.id for b in self.source_blocks)
        return f"<{self.label}:{self.id} Score={self.score} Blks={blocks}>"


# --- 2. The Solver ---
class Solver:
    """Finds the subset of candidates with max total score and no shared blocks."""

    def solve(self, candidates: list[Candidate]) -> list[Candidate]:
        # Sort by score descending (optimization for backtracking)
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        best_score, best_solution = self._backtrack(sorted_candidates, 0, set())
        return best_solution

    def _backtrack(
        self,
        candidates: list[Candidate],
        index: int,
        consumed_blocks: set[Block],
    ) -> tuple[float, list[Candidate]]:
        if index >= len(candidates):
            return 0.0, []

        current = candidates[index]

        # Branch 1: Exclude 'current'
        # We always explore this (in a real implementation, we'd prune based on max possible remaining score)
        score_exclude, sol_exclude = self._backtrack(
            candidates, index + 1, consumed_blocks
        )

        # Branch 2: Include 'current' (only if valid)
        conflict = False
        for b in current.source_blocks:
            if b in consumed_blocks:
                conflict = True
                break

        if not conflict:
            # Recurse
            new_consumed = consumed_blocks | current.source_blocks
            score_include_sub, sol_include_sub = self._backtrack(
                candidates, index + 1, new_consumed
            )
            score_include = current.score + score_include_sub

            if score_include > score_exclude:
                return score_include, [current] + sol_include_sub

        return score_exclude, sol_exclude


# --- 3. Classifiers (The Logic) ---


def run_example_hierarchy():
    print("\n--- Scenario 1: Hierarchy (Part vs PartCount) ---")
    # Data: A "2x" text and a "Brick" image
    blk_txt = Block("txt_2x", "text")
    blk_img = Block("img_brick", "image")

    # Phase 1: Atomic Classifiers
    # StepCount says "2x" looks like a step multiplier (weak match)
    c_step_cnt = Candidate("c1", "StepCount", score=10.0, source_blocks={blk_txt})
    # PartCount says "2x" looks like a part count (strong match)
    c_part_cnt = Candidate("c2", "PartCount", score=20.0, source_blocks={blk_txt})
    # PartImage says "Brick" is a part
    c_part_img = Candidate("c3", "PartImage", score=30.0, source_blocks={blk_img})

    # Phase 2: Composite Classifiers
    # PartClassifier looks at PartCount and PartImage candidates.
    # It creates a PART that claims BOTH blocks.
    # Score logic: Sum of parts + Bonus for correct layout
    c_part = Candidate(
        "c4",
        "Part",
        score=(c_part_cnt.score + c_part_img.score + 15.0),  # 20+30+15 = 65
        source_blocks={blk_txt, blk_img},
        children=[c_part_cnt, c_part_img],
    )

    candidates = [c_step_cnt, c_part_cnt, c_part_img, c_part]

    print("Candidates:")
    for c in candidates:
        print(f"  {c}")

    # Phase 3: Solve
    solver = Solver()
    winners = solver.solve(candidates)

    print("Winners:")
    for w in winners:
        print(f"  {w}")

    # Explanation:
    # c_step_cnt (10) vs c_part_cnt (20) -> c_part_cnt wins atomic battle.
    # But c_part (65) consumes {txt, img}.
    # Alternative: c_part_cnt (20) + c_part_img (30) = 50.
    # 65 > 50. The composite Part wins.


def run_example_arrow_subtraction():
    print("\n--- Scenario 2: Arrow Split (Subtraction) ---")
    # Data: [Image] -- [Arrow] -- [Image]
    b_left = Block("img_left", "image")
    b_mid = Block("img_arrow", "image")
    b_right = Block("img_right", "image")

    # Phase 1: Atomic
    # ArrowClassifier identifies the middle block
    c_arrow = Candidate("c1", "Arrow", score=50.0, source_blocks={b_mid})

    # Phase 2: Diagram Classifier
    # Sees all 3 are connected images.
    # 1. Emits FULL Diagram
    # Note: Score is lowered because b_mid looks like an arrow, reducing diagram confidence
    c_diag_full = Candidate(
        "c2", "DiagramFull", score=80.0, source_blocks={b_left, b_mid, b_right}
    )

    # 2. Subtractive Logic
    # "I see b_mid is claimed by c_arrow. What happens if I remove it?"
    # Result: b_left and b_right become disconnected.
    c_diag_left = Candidate("c3", "DiagramPart", score=40.0, source_blocks={b_left})
    c_diag_right = Candidate("c4", "DiagramPart", score=40.0, source_blocks={b_right})

    candidates = [c_arrow, c_diag_full, c_diag_left, c_diag_right]

    print("Candidates:")
    for c in candidates:
        print(f"  {c}")

    # Phase 3: Solve
    solver = Solver()
    winners = solver.solve(candidates)

    print("Winners:")
    for w in winners:
        print(f"  {w}")

    # Explanation:
    # Option A: c_diag_full (Score 80)
    # Option B: c_arrow (50) + c_diag_left (40) + c_diag_right (40) = 130.
    # Option B wins.


if __name__ == "__main__":
    run_example_hierarchy()
    run_example_arrow_subtraction()
