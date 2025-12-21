# Constraint Solver Implementation TODO

**Branch:** `feature/constraint-solver-classification`  
**Created:** December 21, 2025

## Phase 0: Infrastructure ✋ NOT STARTED

### Core Components

- [ ] **Add ortools dependency**
  - File: `3rdparty/requirements.txt`
  - Add: `ortools>=9.8.0`
  - Run: `pants generate-lockfiles ::`

- [ ] **Create ConstraintModel wrapper**
  - File: `src/build_a_long/pdf_extract/classifier/constraint_model.py`
  - Class: `ConstraintModel`
  - Methods:
    - [ ] `__init__()`
    - [ ] `add_candidate(candidate) -> IntVar`
    - [ ] `get_var(candidate) -> IntVar`
    - [ ] `at_most_one_of(candidates)`
    - [ ] `exactly_one_of(candidates)`
    - [ ] `if_selected_then(parent, children)`
    - [ ] `if_any_selected_then_one_of(group_a, group_b)`
    - [ ] `mutually_exclusive(candidate_a, candidate_b)`
    - [ ] `add_block_exclusivity_constraints(candidates)`
    - [ ] `maximize(objective_terms)`
    - [ ] `solve() -> (bool, dict[int, bool])`

- [ ] **Create SchemaConstraintGenerator**
  - File: `src/build_a_long/pdf_extract/classifier/schema_constraint_generator.py`
  - Class: `SchemaConstraintGenerator`
  - Methods:
    - [ ] `generate_for_classifier(classifier, model, result)`
    - [ ] `_get_element_class(label) -> type[LegoPageElement]`
    - [ ] `_generate_field_constraints(element_class, candidates, model)`
    - [ ] `_generate_uniqueness_constraints(element_class, candidates, model)`
    - [ ] `_generate_orphan_constraints(element_class, candidates, model, result)`
    - [ ] `_parse_field_type(field_type) -> (child_type, cardinality)`
    - [ ] `_get_child_candidates(candidate, field_name) -> list[Candidate]`

### Base Class Updates

- [ ] **Update LabelClassifier**
  - File: `src/build_a_long/pdf_extract/classifier/label_classifier.py`
  - Add method: `declare_constraints(model: ConstraintModel, result: ClassificationResult) -> None`
  - Default implementation: no-op (returns without adding constraints)

- [ ] **Update Classifier**
  - File: `src/build_a_long/pdf_extract/classifier/classifier.py`
  - Add: `use_constraint_solver: bool = False` parameter to `__init__`
  - Add: `use_solver_for: set[str] | None = None` parameter to `__init__`
  - Add method: `_solve_constraints(result) -> set[int]`
  - Update: `classify()` to call solver if enabled

- [ ] **Update ClassificationResult**
  - File: `src/build_a_long/pdf_extract/classifier/classification_result.py`
  - Add: `_solver_selected_ids: set[int] | None = PrivateAttr(default=None)`
  - Add method: `_set_solver_selection(selected_ids: set[int])`
  - Update: `build()` to check solver selection before building
  - Add method: `get_all_candidates() -> list[Candidate]`

### Testing

- [ ] **Unit tests for ConstraintModel**
  - File: `src/build_a_long/pdf_extract/classifier/constraint_model_test.py`
  - Tests:
    - [ ] `test_add_candidate`
    - [ ] `test_at_most_one_of`
    - [ ] `test_exactly_one_of`
    - [ ] `test_if_selected_then`
    - [ ] `test_block_exclusivity`
    - [ ] `test_solve_simple`
    - [ ] `test_solve_infeasible`
    - [ ] `test_maximize`

- [ ] **Unit tests for SchemaConstraintGenerator**
  - File: `src/build_a_long/pdf_extract/classifier/schema_constraint_generator_test.py`
  - Tests:
    - [ ] `test_parse_required_field`
    - [ ] `test_parse_optional_field`
    - [ ] `test_parse_sequence_field`
    - [ ] `test_generate_field_constraints`
    - [ ] `test_generate_uniqueness_constraints`

- [ ] **Integration test**
  - File: `src/build_a_long/pdf_extract/classifier/tests/test_constraint_solver_infrastructure.py`
  - Test: Create simple classifier with constraints, verify solver runs without errors

### Documentation

- [ ] Add docstrings to all new classes/methods
- [ ] Add type hints everywhere
- [ ] Add inline comments for complex logic

---

## Phase 1: PartsList + Parts 🎯 CURRENT FOCUS

### Schema Updates

- [ ] **Update PartsList**
  - File: `src/build_a_long/pdf_extract/extractor/lego_page_elements.py`
  - Add: `__constraint_rules__` class variable
  - Rules:
    - `'parts': {'min_count': 1}` - Must have at least 1 part

- [ ] **Update Part**
  - File: `src/build_a_long/pdf_extract/extractor/lego_page_elements.py`
  - Add: `__constraint_rules__` class variable
  - Rules:
    - `'count': {'required': True}` - Count is mandatory

### Classifier Updates

- [ ] **Update PartsListClassifier**
  - File: `src/build_a_long/pdf_extract/classifier/parts/parts_list_classifier.py`
  - Implement: `declare_constraints(model, result)`
  - Constraints:
    - If parts_list selected, its parts must be selected
    - At least 1 part required

- [ ] **Update PartClassifier**
  - File: `src/build_a_long/pdf_extract/classifier/parts/parts_classifier.py`
  - Implement: `declare_constraints(model, result)`
  - Constraints:
    - If part selected, part_count must be selected
    - If part selected and has part_image, part_image must be selected

### Testing

- [ ] **Unit test: parts_list with 0 parts**
  - Should be rejected by solver

- [ ] **Unit test: parts with conflicting blocks**
  - Solver should pick non-conflicting subset

- [ ] **Integration test: compare with/without solver**
  - Results should be identical for simple cases
  - File: `src/build_a_long/pdf_extract/classifier/tests/test_parts_list_solver.py`

- [ ] **Enable solver for parts_list in main classifier**
  - File: `src/build_a_long/pdf_extract/classifier/classifier.py`
  - Add: `use_solver_for={'parts_list', 'part', 'part_count', 'part_image'}`
  - Test on real PDFs

---

## Phase 2: OpenBag with Alternatives

### Classifier Updates

- [ ] **Update OpenBagClassifier scoring**
  - File: `src/build_a_long/pdf_extract/classifier/bags/open_bag_classifier.py`
  - Generate multiple variants per bag:
    - Greedy variant (all nearby parts)
    - Conservative variant (high-confidence only)
  - Update: `OpenBagScore` to include `variant: str` field

- [ ] **Update OpenBagClassifier constraints**
  - Implement: `declare_constraints(model, result)`
  - Constraints:
    - Group by bag_number - at most one variant per bag
    - If bag selected, its parts must be selected
    - If bag selected, bag_number must be selected

### Schema Updates

- [ ] **Update OpenBag**
  - File: `src/build_a_long/pdf_extract/extractor/lego_page_elements.py`
  - Add: `__constraint_rules__` if needed

### Testing

- [ ] **Test page 20 specifically**
  - File: `src/build_a_long/pdf_extract/classifier/tests/test_page20_orphaned_elements.py`
  - Verify: No orphaned arrows/parts_lists
  - Verify: Solver picks conservative variant when greedy conflicts

- [ ] **Performance test**
  - Measure solve time for typical pages
  - Target: < 1 second

- [ ] **Enable solver for open_bag**
  - Update: `use_solver_for` set in classifier initialization

---

## Phase 3: Step + Spatial Assignment

### Schema Updates

- [ ] **Update Step**
  - File: `src/build_a_long/pdf_extract/extractor/lego_page_elements.py`
  - Add: `__constraint_rules__`
  - Rules:
    - `'step_number': {'unique_by': 'value'}` - Unique step numbers
    - `'diagram': {'assignment': 'spatial'}` - Post-solve assignment
    - `'arrows': {'no_orphans': True}` - Arrows need parent
    - `'subassemblies': {'assignment': 'spatial'}` - Post-solve assignment

### Classifier Updates

- [ ] **Update StepClassifier scoring**
  - File: `src/build_a_long/pdf_extract/classifier/steps/step_classifier.py`
  - Create step candidates WITHOUT diagram
  - Remove diagram pre-assignment logic from `_StepScore`

- [ ] **Update StepClassifier constraints**
  - Implement: `declare_constraints(model, result)`
  - Constraints:
    - Unique step numbers
    - If step selected, step_number must be selected
    - If step selected, parts_list (if present) must be selected
    - No orphaned arrows (arrows need at least one step)
    - No orphaned rotation symbols

- [ ] **Refactor StepClassifier.build_all()**
  - Remove speculative building of arrows/parts_lists
  - Build only solver-selected candidates
  - Add spatial assignment after building:
    - `_assign_diagrams_to_steps(steps, diagrams, result)`
    - `_assign_arrows_to_steps(steps, arrows, result)`
    - `_assign_subassemblies_to_steps(steps, subassemblies, result)`

- [ ] **Implement spatial assignment helpers**
  - Method: `_compute_assignment_cost(step, diagram, result) -> float`
  - Consider: distance, vertical alignment, divider obstruction
  - Use: scipy's `linear_sum_assignment` for Hungarian matching

### Testing

- [ ] **Test step + diagram assignment**
  - Simple page with 1 step, 1 diagram
  - Page with 2 steps, 2 diagrams
  - Page with 3 steps, 2 diagrams (one shared)

- [ ] **Test step uniqueness**
  - Page with duplicate step_number candidates
  - Solver should pick at most one

- [ ] **Test no orphaned arrows**
  - Page with arrows but no steps → should fail or not select arrows

- [ ] **Performance benchmark**
  - Solve time + assignment time
  - Target: < 2 seconds total per page

---

## Phase 4: Full Integration

### All Remaining Elements

- [ ] **Add constraint rules to all elements**
  - [ ] `StepNumber` - unique by value
  - [ ] `Diagram` - can be shared or assigned spatially
  - [ ] `Arrow` - needs parent step
  - [ ] `RotationSymbol` - needs parent step
  - [ ] `SubAssembly` - spatial assignment to step
  - [ ] `SubStep` - needs parent SubAssembly or Step
  - [ ] `BagNumber` - unique by value
  - [ ] `PageNumber` - at most one per page
  - [ ] `ProgressBar` - at most one per page
  - [ ] `Background` - at most one per page

### Classifier Updates

- [ ] Implement `declare_constraints()` for all classifiers:
  - [ ] `StepNumberClassifier`
  - [ ] `DiagramClassifier`
  - [ ] `ArrowClassifier`
  - [ ] `RotationSymbolClassifier`
  - [ ] `SubAssemblyClassifier`
  - [ ] `SubStepClassifier`
  - [ ] `BagNumberClassifier`
  - [ ] `PageNumberClassifier`
  - [ ] `ProgressBarClassifier`
  - [ ] `BackgroundClassifier`

### Enable by Default

- [ ] **Update Classifier initialization**
  - Set: `use_constraint_solver=True` by default
  - Set: `use_solver_for=None` (all labels)

### Cleanup

- [ ] **Remove old speculative building**
  - Clean up StepClassifier phases
  - Remove build stack depth checks
  - Remove `has_step_candidates` guards

- [ ] **Remove unnecessary cascade rollback code**
  - Simplify ClassificationResult
  - Remove `_fail_candidate_tree()`
  - Keep basic rollback for build failures

### Testing

- [ ] **Run full test suite**
  - All existing tests should pass
  - No regressions

- [ ] **Process all example PDFs**
  - Compare with baseline (solver disabled)
  - Verify no orphaned elements
  - Check for performance regressions

- [ ] **Performance benchmarking**
  - Average time per page
  - Memory usage
  - Identify slowest pages

- [ ] **Memory profiling**
  - Check for memory leaks
  - Verify reasonable memory usage

---

## Documentation & Polish

- [ ] **Update DESIGN.md**
  - Document constraint solver approach
  - Update architecture diagrams

- [ ] **Update classifier README**
  - Explain constraint solver
  - How to add constraints to new classifiers

- [ ] **Add troubleshooting guide**
  - How to debug unsatisfiable constraints
  - How to visualize candidate graph
  - How to disable solver if needed

- [ ] **Update orphaned_elements_issue.md**
  - Mark as resolved
  - Link to new architecture

---

## Performance Monitoring

### Metrics to Track

- [ ] Solve time per page (average, p50, p95, p99)
- [ ] Assignment time per page
- [ ] Total classification time per page
- [ ] Memory usage per page
- [ ] Number of candidates per page
- [ ] Number of constraints per page
- [ ] Solver iterations / decisions

### Optimization Opportunities

- [ ] Cache schema constraint analysis
- [ ] Pre-filter impossible candidates
- [ ] Parallel solving for batch processing
- [ ] Incremental solving for similar pages

---

## Success Criteria

### Must Have (Phase 4 Complete)
- ✅ No orphaned elements in any processed PDF
- ✅ All uniqueness constraints satisfied
- ✅ No block conflicts
- ✅ All tests pass
- ✅ < 2s average per page

### Nice to Have
- ✅ < 10% slower than current approach
- ✅ Memory usage < 2x current
- ✅ Clear error messages for constraint failures
- ✅ Easy to add new constraints

---

## Notes

- Start with Phase 0 - infrastructure only
- Each phase can be tested independently
- Can rollback via feature flag if issues arise
- Schema-driven approach should make maintenance easier

**Current Status:** Phase 0 not started, infrastructure design complete
