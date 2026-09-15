# Orphaned Elements Issue - Investigation Report

**Date:** December 21, 2025  
**Issue:** `AssertionError: Page 20: 3 constructed elements not on Page (programming error)`

## Problem Summary

After fixing the `open_bag` validation error (empty `source_blocks`), we encountered a new issue where certain elements (arrows, parts_lists) were being successfully constructed but left "orphaned" - they exist with `.constructed` set, but are not attached to any parent Page element. This triggered a validation error designed to catch programming errors.

### Specific Failing Case

**PDF:** `data/10216/4596701.pdf`, Page 20  
**Orphaned Elements:**

- `arrow` at (370.7, 97.6, 409.0, 115.6)
- `arrow` at (365.6, 87.4, 397.7, 102.9)
- `parts_list` at (22.7, 22.7, 178.6, 109.5)

## Root Cause Analysis

### The Classification Architecture

The classification system uses a **candidate-based approach** with these key mechanisms:

1. **Scoring Phase**: Classifiers score candidates and add them to `ClassificationResult.candidates`
2. **Building Phase**: `ClassificationResult.build()` constructs elements from candidates
3. **Conflict Resolution**: When multiple candidates overlap (share blocks), the system:

   - Marks consumed blocks when a candidate is built
   - Fails competing candidates that use those blocks
   - Uses snapshot/rollback for transactional semantics during build

4. **Parent-Child Relationships**: We added dependency tracking:
   - `_build_stack`: Tracks nested `build()` calls via candidate IDs
   - `_build_dependencies`: Maps `parent_id -> [child_ids]` for all nested builds
   - **Cascade Rollback**: When a parent fails a conflict, recursively clear `.constructed` on all children

### The Step Classifier's Unusual Architecture

The `StepClassifier.build_all()` method has a **phased approach** that breaks the normal parent-child pattern:

```
Phase 1: Pre-check (determine if page has valid steps)
Phase 2: Build subassemblies
Phase 3: Build ALL arrows eagerly (if has_step_candidates)
Phase 4: Build substeps
Phase 5: Build steps (partial - just step_number + parts_list)
Phase 6: Build diagrams
Phase 7: Assign diagrams/arrows to steps via Hungarian matching
```

**The Problem:** Phases 3 and 5 build arrows and parts_lists **BEFORE** knowing if any steps will successfully build. Steps are built in Phase 5, but they can fail for various reasons (e.g., their step_number lost a conflict with a bag_number).

### What Actually Happened on Page 20

Based on debug logs with `--log-level DEBUG`, here's the exact sequence:

1. **Open Bag Construction** (depth=5-30):

   - `open_bag` classifier builds a bag with bag_number=10
   - Bag builds multiple `part` elements
   - Each part builds part_count → part_image
   - Creates a deep nesting chain (depth up to 30)

2. **Step Classifier's build_all()** is invoked (depth=31):

   - `has_step_candidates = True` (there ARE step candidates on the page)
   - Phase 3 should build arrows, but log shows they're built later
   - Phase 5 builds `parts_list` at (22.7, 22.7, 178.6, 109.5)
     - **Parent:** `part_image` (depth=30), NOT a step!
     - This parts_list builds two arrows at (370.7, 97.6) and (365.6, 87.4)
     - **Parent of arrows:** `parts_list` (depth=31)

3. **Arrows Try to Build Steps** (depth=34):

   - The arrows attempt to build step candidates
   - All step candidates FAIL: their step_number at (22.7, 119.7, 57.4, 157.7) lost a conflict to `bag_number` at the same location
   - Exception: `CandidateFailedError: Lost conflict to 'bag_number'`
   - Step classifier catches the exception, logs it, continues

4. **Result:**
   - Steps: 0 built successfully
   - Arrows: 2 built successfully, parent=parts_list
   - Parts_list: 1 built successfully, parent=part_image
   - **All are orphaned** - no step owns them

### Why Cascade Rollback Didn't Help

We implemented cascade rollback to handle this pattern:

```
Parent builds → Child builds → Parent fails conflict → Cascade rolls back child
```

But the actual pattern on Page 20 was:

```
A builds successfully → B builds successfully → C TRIES to build → C fails → A and B left orphaned
```

The cascade rollback triggers when:

- A candidate that HAS CHILDREN loses a conflict (handled in `_fail_conflicting_candidates`)
- A candidate build raises an exception (handled in exception handlers)

But it DOESN'T trigger when:

- A candidate is built successfully
- That candidate tries to build something else
- That other thing fails
- The original candidate is never marked as failed

### The Build Stack Anomaly

A key discovery: When `step_classifier.build_all()` runs, **it's not at the top level**. The build stack shows:

```
build_stack depth=31: parent=4616163248 ('part_image')
```

This means `build_all()` for step was called WHILE building part_image! This is unexpected. The normal pattern is:

1. Top-level classifier loop calls `build_all()` for each label
2. `build_all()` is called at stack depth 0

But here, `build_all()` for step is being invoked at depth 31, nested inside a part_image build. This suggests either:

- A classifier is calling `build_all()` for another label (unusual)
- The build_all invocation timing is wrong
- There's recursive/nested build_all calls happening

## Original Architecture Design

### ClassificationResult Build Mechanism

```python
def build(candidate, classifier, **kwargs) -> Element:
    """Build an element from a candidate with transactional semantics."""

    # 1. Check if already built or failed
    if candidate.constructed: return candidate.constructed
    if candidate.failure_reason: raise CandidateFailedError

    # 2. Take snapshot for rollback
    snapshot = self._take_snapshot()

    # 3. Track in build stack
    candidate_id = id(candidate)
    parent_id = self._build_stack[-1] if self._build_stack else None
    self._build_stack.append(candidate_id)

    try:
        # 4. Call classifier's build method
        element = classifier.build(candidate, self, **kwargs)
        candidate.constructed = element

        # 5. Record parent-child relationship
        if parent_id:
            self._build_dependencies[parent_id].append(candidate_id)

        # 6. Mark blocks as consumed
        for block in candidate.source_blocks:
            self._consumed_blocks.add(block.id)

        # 7. Fail conflicting candidates
        self._fail_conflicting_candidates(candidate)

        return element

    except CandidateFailedError:
        # Rollback and propagate
        self._restore_snapshot(snapshot)
        self._build_stack.pop()
        raise

    except Exception:
        # Rollback and propagate
        self._restore_snapshot(snapshot)
        self._build_stack.pop()
        raise
```

### Snapshot/Rollback System

Snapshots capture:

- `consumed_blocks`: Set of block IDs marked as consumed
- `candidates`: Deep copy of all candidate objects (with their `.constructed` and `.failure_reason`)

When an exception occurs during `build()`:

1. Restore consumed_blocks to snapshot state
2. Restore all candidate states (clear `.constructed`, `.failure_reason`)
3. Remove candidate from build stack
4. Propagate exception

**Important:** Snapshot rollback only affects state changes made during THIS `build()` call. Nested `build()` calls that completed successfully are NOT rolled back by the snapshot.

### Cascade Rollback (Implemented for Conflict Resolution)

When a candidate wins a conflict (in `_fail_conflicting_candidates`):

```python
# Mark losers as failed
for losing_candidate in conflicting_candidates:
    self._fail_candidate_tree(losing_candidate, reason)

def _fail_candidate_tree(candidate, reason):
    """Recursively rollback this candidate and all its children."""
    candidate_id = id(candidate)

    # Recursively rollback children first
    if candidate_id in self._build_dependencies:
        for child_id in self._build_dependencies[candidate_id]:
            child_candidate = find_by_id(child_id)
            if child_candidate and child_candidate.constructed:
                self._fail_candidate_tree(child_candidate, f"Parent '{candidate.label}' failed")

    # Clear this candidate
    candidate.constructed = None
    candidate.failure_reason = reason
```

This works when a parent loses a conflict AFTER building children. But it doesn't help when:

- Parent builds successfully
- Parent tries to build something unrelated
- That unrelated thing fails
- Parent is never marked as failed

## Why Standard Solutions Don't Work

### Option 1: Extend Cascade Rollback to Exceptions

We tried adding `_rollback_children()` calls in exception handlers:

```python
except Exception:
    self._restore_snapshot(snapshot)
    self._rollback_children(candidate_id)  # NEW
    self._build_stack.pop()
    raise
```

**Result:** Didn't help. The arrows and parts_list were built successfully. When the step candidate failed, it raised `CandidateFailedError` which was caught by `step_classifier.build_all()`, not propagated up to the parent's exception handler.

### Option 2: Let Exceptions Propagate

Don't catch exceptions in `step_classifier.build_all()` - let them propagate to trigger rollback.

**Problem:** This would break the design. Step classifier intentionally tries multiple step candidates and expects some to fail. It catches exceptions to continue processing.

### Option 3: Manual Cleanup in Step Classifier

After Phase 5, check if any steps were built. If not, manually fail all arrows and parts_lists.

**Problem:**

- Need to track which arrows/parts_lists were built during THIS invocation
- Complex to implement correctly
- Doesn't solve the architectural issue

## The Real Architecture Problem

The step classifier's phased approach violates a key assumption:

**Assumption:** Elements are built in a parent-child hierarchy where:

- Parent builds first
- Parent calls `result.build()` for children
- If parent build fails/raises, children are rolled back

**Reality in StepClassifier:**

- Arrows/parts_lists built BEFORE their intended parent (steps)
- Built "speculatively" hoping a step will claim them
- When no step succeeds, they're orphaned

This is a **build-time vs assignment-time** mismatch:

- Build-time: Arrows/parts_lists are built eagerly
- Assignment-time: Later assigned to steps via Hungarian matching
- Gap: If no steps are built, assignment never happens

## Questions for Solution Design

1. **Why is build_all() for step called at depth 31?** Is there code that calls `result.build_all()` from within a classifier's `build()` method? This seems wrong.

2. **Should arrows/parts_lists be built by step_classifier at all?** Perhaps they should be built by their own classifiers' `build_all()`, and step_classifier should only ASSIGN them?

3. **Can we defer arrow/parts_list building until after steps succeed?** Refactor Phase 5 to:

   - Try to build steps
   - If successful, THEN build their arrows/parts_lists
   - This would make them proper children

4. **Is the Hungarian matching approach necessary?** Could steps identify their arrows/parts_lists during their own `build()` method instead of a later assignment phase?

5. **Should we track "speculative" builds?** Add a mechanism to mark candidates as "tentative" until confirmed by a parent?

## Relevant Code Locations

- **Classification Result:** [src/build_a_long/pdf_extract/classifier/classification_result.py](../src/build_a_long/pdf_extract/classifier/classification_result.py)

  - `build()` method: Lines 232-395
  - `_fail_conflicting_candidates()`: Lines 413-456
  - `_fail_candidate_tree()`: Lines 458-541
  - Build stack tracking: Lines 275-290

- **Step Classifier:** [src/build_a_long/pdf_extract/classifier/steps/step_classifier.py](../src/build_a_long/pdf_extract/classifier/steps/step_classifier.py)

  - `build_all()` method: Lines 135-558
  - Phase 3 (arrows): Lines 260-287
  - Phase 5 (steps): Lines 333-376
  - Step build: Lines 605-642

- **Validation:** [src/build_a_long/pdf_extract/validation/rules.py](../src/build_a_long/pdf_extract/validation/rules.py)
  - `assert_constructed_elements_on_page()`: Lines 108-145

## Test Cases

Currently no automated test reproduces this issue. To create one:

```python
def test_orphaned_elements_when_all_steps_fail():
    """Test that arrows/parts_lists are cleaned up when no steps are built."""
    # Setup: Page with step candidates that will fail (conflict with bag_number)
    # Expected: No orphaned arrows or parts_lists
    # Actual: Arrows and parts_lists remain constructed
```

## Debug Commands

To reproduce with full logging:

```bash
pants run src/build_a_long/pdf_extract:main -- \
    data/10216/4596701.pdf \
    --pages 20 \
    --output-dir debug/test \
    --log-level DEBUG 2>&1 | tee debug.log
```

Search for key patterns:

```bash
# Find build stack entries
grep '\[build_stack\]' debug.log

# Find dependency recordings
grep '\[dependency\]' debug.log

# Find cascade rollbacks
grep '\[cascade\]' debug.log

# Find the failing elements
grep -E '370\.7,97\.6|365\.6,87\.4|22\.7,22\.7,178\.6' debug.log
```

## Next Steps

1. Investigate why `build_all()` for step is called at depth 31
2. Decide on architectural approach:
   - Refactor step classifier to build children only after parent succeeds?
   - Add speculative build tracking?
   - Change validation to allow certain orphaned elements?
3. Add regression tests
4. Document the chosen solution
