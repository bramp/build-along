# Global Optimization Architecture for LEGO Instruction Extraction

## 1. Core Philosophy: Maximum Weight Independent Set (MWIS)

We are shifting from a **Greedy Pipeline** (score -> construct -> consume -> repeat) to a **Global Solver** model.

The core problem is modeled as finding the **Maximum Weight Independent Set** on a conflict graph:
*   **Nodes:** All possible Candidates (Interpretations of data).
*   **Weights:** The confidence score of that candidate.
*   **Edges (Conflicts):** A connection exists between two candidates if they claim the same source pixel/vector block.

We want to pick the set of candidates that maximizes the total Score, such that no two candidates share a block.

## 2. The Workflow

### Phase 1: Atomic Scoring
Run lightweight classifiers to score atomic elements.
*   *Input:* Raw Blocks.
*   *Output:* `List[Candidate]` (e.g., `PartCount`, `StepNumber`, `Arrow`, `Icon`).
*   *Note:* Over-generate. If "2x" looks like a PartCount (0.8) and a StepCount (0.4), emit **both**.

### Phase 2: Composite & Alternative Generation
Run aggregate classifiers. These consume the Atomic candidates to build larger structures.
*   *Input:* Raw Blocks + Atomic Candidates.
*   *Output:* `List[Candidate]` (e.g., `Part`, `PartList`, `Diagram`).

**The "Subtraction" Strategy (For Arrows/Diagrams):**
When a classifier identifies a large structure (like a Diagram), it checks if any of its constituent blocks are *also* claimed by highly-scored Atomic candidates (like Arrows).
1.  Generate **Candidate A**: The full structure (Diagram + Arrow block).
2.  Generate **Candidate B**: The structure *minus* the contested block. (If this splits the structure, generate multiple candidates).

### Phase 3: The Solver
Feed **ALL** candidates (Atomic and Composite) into the MWIS Solver.
*   The solver identifies conflicts (shared blocks).
*   It returns the optimal subset of candidates.

## 3. Handling Hierarchies (The "Bubbling Up" Rule)

How do we decide between a `PartCount` ("2x") and a `Part` (which contains that "2x")?

**The Rule:** A Composite Candidate **inherits** the conflicts of its children.

*   **Block:** `[Txt: "2x"]`, `[Img: Brick]`
*   **Candidate 1 (Atomic):** `PartCount` claims `[Txt: "2x"]`. Score: 10.
*   **Candidate 2 (Composite):** `Part` composed of (`PartCount` + `PartImage`). Claims `[Txt: "2x", Img: Brick]`. Score: 25.

**Conflict:** Candidate 1 and Candidate 2 conflict because they share `[Txt: "2x"]`.
**Resolution:** The Solver picks Candidate 2 because 25 > 10.

**Implication:**
You do not need logic to "fold" hierarchies. You simply define the `Part` candidate. If the `Part` wins, the `PartCount` (as an independent entity) "loses", but the `Part` object *contains* that data, so it is preserved in the final tree.

---

## 4. Example Scenario: The Arrow Split

**Input:** Three image blocks in a row: `[Block A] -- [Block B] -- [Block C]`.
*   `Block B` looks like an Arrow.
*   `A, B, C` together look like a Diagram.

**Execution:**
1.  **ArrowClassifier:**
    *   See `Block B`.
    *   Emits **Candidate 1 (Arrow)**. Blocks: `{B}`. Score: **50**.

2.  **DiagramClassifier:**
    *   Finds connected component `{A, B, C}`.
    *   *Action:* Emits **Candidate 2 (Full Diagram)**. Blocks: `{A, B, C}`. Score: **80** (Lower confidence because B looks weird).
    *   *Check:* Sees `Block B` is contested by `Cand 1`.
    *   *Subtraction:* Removes `B`. Remaining: `{A}`, `{C}`.
    *   *Action:* Emits **Candidate 3 (Diagram Left)**. Blocks: `{A}`. Score: **40**.
    *   *Action:* Emits **Candidate 4 (Diagram Right)**. Blocks: `{C}`. Score: **40**.

3.  **Solver:**
    *   **Option X:** Pick `Cand 2` (Full Diagram).
        *   Total Score: **80**.
    *   **Option Y:** Pick `Cand 1` + `Cand 3` + `Cand 4` (Arrow + Left + Right).
        *   Total Score: 50 + 40 + 40 = **130**.

    **Result:** Solver picks Option Y. The Arrow is successfully extracted, and the diagram is split.
