# Constraint Solver Architecture - Design Document

**Date:** December 21, 2025 (Updated: December 22, 2025)  
**Branch:** `feature/constraint-solver-classification`  
**Status:** Phase 2 OpenBag Complete

## Current Implementation Summary

The constraint solver uses a **generic `Candidate[T]`** approach for type-safe
constraint mapping. Key components:

```python
# Generic Candidate class - T indicates what element type it produces
class Candidate(BaseModel, Generic[T]):
    \"\"\"A candidate that produces a LegoPageElement of type T when built.\"\"\"
    ...

# Score classes use generics to map to schema fields
class _PartsListScore(Score):
    part_candidates: list[Candidate[Part]] = []  # Maps to PartsList.parts

class _PartPairScore(Score):
    part_count_candidate: Candidate[PartCount]    # Maps to Part.count
    part_image_candidate: Candidate[PartImage]    # Maps to Part.diagram
```

**SchemaConstraintGenerator** introspects `Candidate[T]` types and matches them
to schema fields by element type (not field name), providing type-safe constraint
generation without string literals.

---

## Problem Statement

The current classification architecture has an orphaned elements issue where:

1. Elements are built speculatively (arrows, parts_lists) before knowing if their parents (steps) will succeed
2. When parents fail due to conflicts, children are left orphaned
3. Cascade rollback doesn't help because the parent-child relationship is inverted

Root cause: **StepClassifier builds children before knowing if steps will exist**, violating the proper parent-child hierarchy.

## Proposed Solution: Constraint Satisfaction Problem (CSP)

Model classification as a constraint satisfaction problem using OR-Tools CP-SAT:

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│             Phase 1: Scoring                    │
│  All classifiers generate candidates            │
│  (Multiple alternatives per element)            │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│        Phase 2: Constraint Solving              │
│  CP-SAT selects optimal candidate combination   │
│  - Block exclusivity                            │
│  - Parent-child dependencies                    │
│  - Uniqueness constraints                       │
│  - No orphaned elements                         │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│          Phase 3: Building                      │
│  Build only selected candidates                 │
│  - Simple elements built directly               │
│  - Spatial assignments via Hungarian matching   │
└─────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Schema-Driven Constraints

**Leverage Pydantic schema to auto-generate constraints:**

```python
class Step(LegoPageElement):
    """Step with schema metadata for constraint generation."""

    # Constraint metadata embedded in schema
    __constraint_rules__: ClassVar[dict[str, Any]] = {
        'step_number': {'unique_by': 'value'},     # Each step_number.value unique
        'diagram': {'assignment': 'spatial'},      # Diagram assigned via Hungarian
        'arrows': {'no_orphans': True},            # Arrows need parent step
    }

    step_number: StepNumber                        # Required (no None)
    parts_list: PartsList | None = None            # Optional
    diagram: Diagram | None = None                 # Optional, spatial
    arrows: Sequence[Arrow] = Field(default_factory=list)
```

**Auto-generation logic:**

- Required fields → parent selected ⇒ exactly one child selected
- Optional fields → parent selected ⇒ at most one child selected
- Sequence fields → zero or more allowed
- `__constraint_rules__` → custom constraints (uniqueness, no-orphans, spatial)

### 2. Automatic vs Manual Constraints

The `SchemaConstraintGenerator` auto-generates **structural constraints** that can
be derived directly from the schema. Classifiers should only implement
`declare_constraints()` for **semantic constraints** that require domain knowledge.

**Automatic (do NOT implement manually):**

| Constraint Type         | Description                                                | Example                                    |
| ----------------------- | ---------------------------------------------------------- | ------------------------------------------ |
| Parent-child dependency | If parent selected, children must be selected              | Step selected → step_number selected       |
| Child uniqueness        | Each child element can have at most one parent             | A `StepNumber` can't belong to two `Step`s |
| Block exclusivity       | Two candidates using the same block are mutually exclusive | Handled by `ConstraintModel`               |

**Manual (implement in `declare_constraints()`):**

| Constraint Type       | Description                                              | Example                                       |
| --------------------- | -------------------------------------------------------- | --------------------------------------------- |
| Semantic uniqueness   | Elements with same semantic value are mutually exclusive | Only one Step with `step_value=3` per page    |
| Singleton constraints | At most one of this element type per page                | Only one `PageNumber` per page                |
| No-orphan constraints | If child elements exist, parent must also exist          | If arrows selected, at least one step must be |
| Custom business rules | Domain-specific constraints                              | Specific spatial relationships                |

**Example:** `StepClassifier` handles step_value uniqueness and no-orphan constraints:

```python
def declare_constraints(self, model: ConstraintModel, result: ClassificationResult) -> None:
    # Semantic constraint: unique step values
    candidates_by_step_value = self._group_by_step_value(candidates)
    for step_value, cands in candidates_by_step_value.items():
        if len(cands) > 1:
            model.at_most_one_of([c.var for c in cands])

    # No-orphan constraints: arrows, diagrams, etc. need a parent step
    orphan_labels = ["arrow", "rotation_symbol", "subassembly", "substep", "diagram"]
    for label in orphan_labels:
        child_candidates = result.get_candidates(label)
        if child_candidates and step_candidates:
            model.if_any_selected_then_one_of(child_candidates, step_candidates)
```

### 3. Separation of Selection vs Assignment

**Selection (CP-SAT):** Discrete yes/no decisions

- Which candidates to build?
- No block conflicts
- Parent-child dependencies satisfied
- Uniqueness constraints met

**Assignment (Hungarian Matching):** Continuous optimization

- Which diagram goes with which step? (spatial proximity)
- Which arrow points to which step? (direction + distance)
- Minimize total assignment cost

### 4. Progressive Rollout

Start with simple classifiers, add complexity incrementally:

**Phase 1:** Simple parent-child (PartsList → Parts) ✅  
**Phase 2:** OpenBag with Candidate[T] generics ✅  
**Phase 3:** Add spatial assignment (Step + Diagram)  
**Phase 4:** Full integration (all classifiers)

## Implementation Phases

### Phase 0: Infrastructure (Week 1) ✅ COMPLETE

**Goal:** Set up constraint solver infrastructure without changing existing behavior

**Tasks:**

- [x] Add `ortools` dependency to `requirements.txt`
- [x] Create `src/build_a_long/pdf_extract/classifier/constraint_model.py`
  - `ConstraintModel` wrapper around CP-SAT
  - Helper methods: `add_candidate()`, `at_most_one_of()`, `if_selected_then()`, etc.
- [x] Create `src/build_a_long/pdf_extract/classifier/schema_constraint_generator.py`
  - `SchemaConstraintGenerator` for auto-generating constraints from Pydantic
- [x] Add `declare_constraints()` method to `LabelClassifier` base class (default no-op)
- [x] Add `__constraint_rules__` support to element classes
- [x] Add `use_constraint_solver` flag to `Classifier` (default `False`)

**Testing:**

- Unit tests for `ConstraintModel` basic operations
- Unit tests for schema introspection
- No integration changes yet

**Success Criteria:**

- Infrastructure compiles
- Tests pass
- No behavior changes (solver not used)

### Phase 1: PartsList + Parts (Week 2) ✅ INFRASTRUCTURE COMPLETE

**Goal:** Prove out the approach with a simple parent-child relationship

**Status:** Infrastructure done. Tests for full constraint solver integration
skipped pending incremental rollout.

**Why PartsList?**

- Simple hierarchy: PartsList → Parts → PartCount + PartImage
- No spatial assignment complexity
- Self-contained (doesn't affect Steps)
- Easy to validate

**Changes:**

1. **Update PartsList schema:**

```python
class PartsList(LegoPageElement):
    __constraint_rules__: ClassVar[dict] = {
        'parts': {'min_count': 1},  # Must have at least 1 part
    }

    parts: Sequence[Part]
```

1. **Update Part schema:**

```python
class Part(LegoPageElement):
    __constraint_rules__: ClassVar[dict] = {
        'count': {'required': True},  # Count is mandatory
    }

    count: PartCount
    diagram: PartImage | None = None
```

1. **PartsListClassifier uses declarative constraints:**

Constraints are auto-generated by `SchemaConstraintGenerator` from:

- `PartsList.__constraint_rules__['parts']['min_count'] = 1`
- `_PartsListScore.part_candidates: list[Candidate[Part]]` generic type

No manual `declare_constraints()` needed - `Candidate[T]` generic type enables automatic mapping:

```python
from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.extractor.lego_page_elements import Part

class _PartsListScore(Score):
    # Generic type Candidate[Part] tells SchemaConstraintGenerator that
    # these candidates produce Part elements, which maps to PartsList.parts
    part_candidates: list[Candidate[Part]] = []

class PartsListClassifier(LabelClassifier):
    # No declare_constraints() override needed!
    # SchemaConstraintGenerator introspects Candidate[Part] and matches
    # to PartsList.parts field which has type Sequence[Part]
    pass
```

**Key Design Decision:** Using generic `Candidate[T]` instead of annotation-based
`ChildOf("field_name")`. Benefits:

- Type-safe: IDE catches wrong element types
- No string literals that can get out of sync
- Works with Python typing infrastructure

1. **Enable solver for parts_list label only:**

```python
class Classifier:
    def __init__(self, classifiers, use_solver_for: set[str] | None = None):
        self.use_solver_for = use_solver_for or set()

    def _solve_constraints(self, result):
        if not self.use_solver_for:
            return set()  # All candidates allowed

        # Only solve for specified labels
        candidates = [
            c for c in result.get_all_candidates()
            if c.label in self.use_solver_for
        ]
        # ... run solver on filtered candidates
```

**Testing:**

- Test parts_list with 0 parts → rejected by solver
- Test parts_list with parts consuming conflicting blocks
- Compare results with/without solver (should be identical for simple cases)

**Success Criteria:**

- PartsList + Parts work correctly with constraint solver
- No orphaned parts
- No regression in existing tests
- Can toggle solver on/off per label

### Phase 2: OpenBag with Alternatives (Week 3)

**Goal:** Add multiple candidate variants per element

**Changes:**

1. **Update OpenBagClassifier to generate alternatives:**

```python
class OpenBagScore(Score):
    bag_number_candidate: Candidate[BagNumber]  # Maps to OpenBag.bag_number
    parts_candidates: list[Candidate[Part]] = []  # Maps to OpenBag.parts
    variant: str = "default"  # Track which variant this is

def _score(self, result: ClassificationResult):
    """Generate greedy + conservative variants."""

    for bag_num_cand in bag_numbers:
        nearby_parts = self._find_nearby_parts(bag_num_cand)

        # Variant 1: Greedy (all nearby parts)
        result.add_candidate(Candidate(
            label="open_bag",
            source_blocks=[bag_num_block],
            score=0.9,
            score_details=OpenBagScore(
                bag_number_candidate=bag_num_cand,
                parts_candidates=nearby_parts,  # Note: parts_candidates
                variant="greedy"
            ),
        ))

        # Variant 2: Conservative (high-confidence only)
        conservative = [p for p in nearby_parts if self._is_high_confidence(p)]
        result.add_candidate(Candidate(
            label="open_bag",
            source_blocks=[bag_num_block],
            score=0.85,
            score_details=OpenBagScore(
                bag_number_candidate=bag_num_cand,
                parts_candidates=conservative,  # Note: parts_candidates
                variant="conservative"
            ),
        ))
```

1. **OpenBag constraints (mostly auto-generated):**

```python
class OpenBag(LegoPageElement):
    __constraint_rules__: ClassVar[dict] = {
        'bag_number': {'unique_by': 'value'},  # Only one bag per number
    }

# Most constraints auto-generated by SchemaConstraintGenerator:
# - If open_bag selected, bag_number_candidate must be selected
# - If open_bag selected, parts_candidates must be selected (from field types)
#
# Custom constraint (if needed) for variant exclusivity:
def declare_constraints(self, model, result):
    # Group by bag_number (only one variant per bag)
    by_bag = self._group_by_bag_number(candidates)
    for variants in by_bag.values():
        if len(variants) > 1:
            model.at_most_one_of(variants)
```

**Testing:**

- Test page 20 (the failing case) with alternatives
- Verify solver picks conservative variant when greedy conflicts with step_number
- Measure performance (how long does solving take?)

**Success Criteria:**

- Page 20 passes validation (no orphaned elements)
- Solver automatically picks better variant when conflict exists
- Solve time < 1 second for typical page

### Phase 3: Step + Spatial Assignment (Week 4)

**Goal:** Integrate steps with Hungarian matching for diagram assignment

**Changes:**

1. **Update Step schema:**

```python
class Step(LegoPageElement):
    __constraint_rules__: ClassVar[dict] = {
        'step_number': {'unique_by': 'value'},
        'diagram': {'assignment': 'spatial'},
        'arrows': {'no_orphans': True},
        'subassemblies': {'assignment': 'spatial'},
    }

    step_number: StepNumber
    parts_list: PartsList | None = None
    diagram: Diagram | None = None  # Assigned post-solve
    arrows: Sequence[Arrow] = Field(default_factory=list)
    subassemblies: Sequence[SubAssembly] = Field(default_factory=list)
```

1. **Update StepClassifier scoring:**

```python
def _score(self, result):
    """Create step candidates WITHOUT diagram (assigned later)."""

    for step_num_cand in step_numbers:
        for pl_cand in parts_lists:
            if self._is_compatible(step_num_cand, pl_cand):
                result.add_candidate(Candidate(
                    label="step",
                    source_blocks=step_num_cand.source_blocks,
                    score=self._compute_score(step_num_cand, pl_cand),
                    score_details=_StepScore(
                        step_number_candidate=step_num_cand,
                        parts_list_candidate=pl_cand,
                        # NO diagram here - assigned in build_all()
                    ),
                ))
```

1. **Add Step constraints:**

```python
def declare_constraints(self, model, result):
    # Auto-generated:
    # - If step selected, step_number must be selected
    # - If step selected, parts_list (if present) must be selected

    # Custom: Unique step numbers
    by_value = self._group_by_step_value(candidates)
    for candidates_with_same_value in by_value.values():
        if len(candidates_with_same_value) > 1:
            model.at_most_one_of(candidates_with_same_value)

    # Custom: No orphaned arrows
    arrow_candidates = result.get_scored_candidates("arrow")
    step_candidates = result.get_scored_candidates("step")
    if arrow_candidates and step_candidates:
        model.if_any_selected_then_one_of(arrow_candidates, step_candidates)
```

1. **Implement spatial assignment in build_all():**

```python
def build_all(self, result):
    # Build selected steps (partial - no diagram)
    steps = []
    for step_cand in result.get_scored_candidates("step"):
        try:
            step = result.build(step_cand)
            steps.append(step)
        except CandidateFailedError:
            continue  # Solver didn't select this one

    # Build selected diagrams
    diagrams = []
    for diag_cand in result.get_scored_candidates("diagram"):
        try:
            diagram = result.build(diag_cand)
            diagrams.append(diagram)
        except CandidateFailedError:
            continue

    # Hungarian matching: assign diagrams to steps
    self._assign_diagrams_to_steps(steps, diagrams, result)

    # Similar for arrows, subassemblies

    return steps
```

**Testing:**

- Test step + diagram assignment on simple pages
- Test multiple steps competing for same diagram
- Test steps with no compatible diagrams
- Performance benchmark (solve + assignment time)

**Success Criteria:**

- Steps correctly paired with diagrams
- No orphaned arrows/diagrams
- Hungarian matching produces sensible pairings
- Total time (solve + assign) < 2 seconds per page

### Phase 4: Full Integration (Week 5+)

**Goal:** Enable constraint solver for all classifiers

**Changes:**

1. Add `__constraint_rules__` to remaining element types
2. Add `declare_constraints()` to remaining classifiers
3. Enable solver by default (`use_constraint_solver=True`)
4. Remove old speculative building code from StepClassifier
5. Clean up cascade rollback code (no longer needed)

**Testing:**

- Run full test suite with solver enabled
- Process all example PDFs
- Compare results with/without solver
- Performance benchmarking across all PDFs
- Memory profiling

**Success Criteria:**

- All tests pass
- No orphaned elements in any PDF
- No performance regression (< 10% slower)
- Memory usage reasonable (< 2x increase)

## Performance Considerations

**Expected bottlenecks:**

1. CP-SAT solve time (depends on # candidates and constraints)
2. Hungarian matching (O(n³) but n typically small)
3. Schema introspection overhead

**Optimizations:**

- Cache schema constraint analysis
- Pre-filter candidates before solving (remove impossible ones)
- Use beam search for very large candidate sets
- Parallel solving for multi-page batches

**Target performance:**

- Simple page (< 50 candidates): < 100ms
- Complex page (< 200 candidates): < 500ms
- Very complex page (< 500 candidates): < 2s

## Migration Strategy

**Backward Compatibility:**

- Keep `use_constraint_solver` flag (default enabled after Phase 4)
- Preserve old behavior when disabled
- Allow per-label solver enabling for gradual migration

**Rollback Plan:**

- If performance unacceptable, disable solver by default
- If correctness issues, revert to previous approach
- Git branch allows clean revert if needed

## Testing Strategy

### Unit Tests

- `ConstraintModel` operations
- Schema introspection
- Constraint generation for each element type
- Hungarian matching logic

### Integration Tests

- Each phase has dedicated integration test
- Test known failure cases (page 20, etc.)
- Test edge cases (0 candidates, all conflicts, etc.)

### Regression Tests

- Process all example PDFs with/without solver
- Compare outputs (should be identical or better)
- Performance benchmarks

### Validation Tests

- No orphaned elements
- All required children present
- Uniqueness constraints satisfied
- Block consumption correct

## Open Questions

1. **How to handle unsatisfiable constraints?**

   - Log detailed error with which constraints conflict
   - Fall back to greedy approach?
   - Return empty page?

2. **Should we support soft constraints?**

   - E.g., "prefer diagrams below parts_list" as optimization hint
   - CP-SAT supports weighted objectives

3. **How to debug constraint failures?**

   - Log constraint model to file
   - Visualize candidate graph
   - Incremental constraint addition with bisection

4. **Memory usage for large documents?**
   - Process pages in batches?
   - Stream results to disk?

## Success Metrics

### Correctness

- ✅ No orphaned elements in any processed PDF
- ✅ All uniqueness constraints satisfied
- ✅ No block conflicts

### Performance

- ✅ < 2s average per page
- ✅ < 10% slower than current approach
- ✅ Memory usage < 2x current

### Maintainability

- ✅ Constraints co-located with element schemas
- ✅ Easy to add new constraints
- ✅ Clear error messages when constraints fail

## Dependencies

- `ortools` (Google OR-Tools) - Apache 2.0 license, ~50MB installed
- No other new dependencies

## Timeline

- **Week 1:** Infrastructure (Phase 0)
- **Week 2:** PartsList (Phase 1)
- **Week 3:** OpenBag alternatives (Phase 2)
- **Week 4:** Step + spatial (Phase 3)
- **Week 5+:** Full integration (Phase 4)

**Total estimated time:** 5-6 weeks for full implementation

## Next Steps

1. Review this design doc
2. Implement Phase 0 (infrastructure)
3. Write tests for `ConstraintModel`
4. Implement Phase 1 (PartsList)
5. Iterate based on learnings

---

**Notes:**

- This is a significant architectural change
- Progressive rollout reduces risk
- Can be disabled via flag if issues arise
- Schema-driven approach is elegant and maintainable
