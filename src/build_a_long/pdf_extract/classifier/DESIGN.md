# Classifier Design Principles

This document outlines the architectural principles for all classifiers in the PDF extraction pipeline.

## Core Philosophy

**Scoring should be based on intrinsic properties of a block, not on pre-determined relationships with other blocks.**

This avoids conflicts that arise when:

1. Multiple classifiers compete for the same child candidates
2. Build-time conditions change, invalidating pre-scored relationships
3. Rollback scenarios leave state inconsistent

## Scoring Phase Principles

### 1. Score Based on Intrinsic Properties Only

Scoring should evaluate a block's inherent characteristics:

- **Geometric properties**: size, aspect ratio, position on page
- **Visual properties**: color, opacity, line thickness
- **Content properties**: text content, font size, numeric patterns
- **Spatial properties**: location relative to page regions (top, bottom, margins)

**Good example:**

```python
def score(self, block: Block) -> float:
    # Score based on intrinsic properties
    score = 0.0
    if block.is_image and block.width > 100 and block.height > 100:
        score += 0.5
    if block.bbox.is_in_region(self.expected_region):
        score += 0.3
    return score
```

**Bad example:**

```python
def score(self, block: Block, all_blocks: list[Block]) -> float:
    # DON'T pre-determine child relationships
    children = self._find_children(block, all_blocks)  # ❌
    if len(children) >= 3:
        score += 0.5
```

### 2. Score Potential, Not Actuality

When a classified element may contain other elements, it's fine to *observe* whether potential children exist—this informs the score. However, don't *pre-assign* specific children or assume they'll still be available at build time.

**Good example:**

```python
def score(self, block: Block, all_blocks: list[Block]) -> float:
    score = 0.0
    # Check if block COULD contain children (has enough space)
    if block.area > MIN_CONTAINER_AREA:
        score += 0.2
    
    # OK: Count potential children to inform score
    potential_parts = [b for b in all_blocks if self._could_be_part(b, block)]
    if len(potential_parts) >= 3:
        score += 0.3  # Likely a parts list
    elif len(potential_parts) == 0:
        score -= 0.5  # Clearly NOT a parts list
    
    return score
```

**Bad example:**

```python
def score(self, block: Block, all_blocks: list[Block]) -> float:
    # DON'T pre-assign specific children
    self.best_part = self._find_best_part(block, all_blocks)  # ❌ Pre-assignment
    self.children = self._find_all_children(block, all_blocks)  # ❌ Stored for later
    if self.best_part:
        score += 0.3
```

The key difference:

- ✅ **Observe**: "There are 5 potential parts here" → use count for scoring
- ❌ **Pre-assign**: "Part X is THE child" → stored reference may be stale at build time

### 3. Keep Score Objects Lightweight

Score objects should contain:

- The candidate block reference
- Computed intrinsic scores
- Optionally, metadata about *why* it scored well (for debugging)

Score objects should NOT contain:

- References to other candidate blocks
- Pre-computed child assignments
- Mutable state that changes during build

## Build Phase Principles

### 4. Discover Relationships at Build Time

During the build phase, find and claim related elements:

```python
def build(self, candidate: Block, result: ClassificationResult) -> Element | None:
    # Find children NOW, when we know what's still available
    available_children = self._find_available_children(candidate, result)
    
    if not self._can_build_with(available_children):
        return None  # Can't build, let candidate be used elsewhere
    
    # Claim children
    for child in available_children:
        result.consume(child)
    
    return MyElement(candidate, children=available_children)
```

### 5. Fail Gracefully, Don't Force

If required children are no longer available at build time:

- Return `None` to indicate build failure
- Don't try to "steal" from other built elements
- Let the candidate potentially be claimed by another classifier

### 6. Consume Blocks Atomically

When building composite elements:

- Consume all blocks together, or none at all
- Use rollback/transaction patterns if partial builds can fail
- Ensure consumed blocks are properly tracked

## Classifier Interaction Principles

### 7. Order Classifiers by Specificity

More specific classifiers should run before general ones:

1. **Highly specific**: Elements with unique visual signatures (rotation symbols, step numbers)
2. **Moderately specific**: Elements defined by spatial patterns (parts lists, callouts)  
3. **General**: Catch-all elements (diagrams, generic images)

### 8. Respect the Consumed Set

During build, always check if blocks are already consumed:

```python
def _find_available_children(self, parent: Block, result: ClassificationResult) -> list[Block]:
    return [
        block for block in potential_children
        if not result.is_consumed(block)  # Always check!
    ]
```

### 9. Claim Associated Elements Proactively

If an element has tightly associated sub-elements (e.g., rotation symbol + dropshadow), claim them immediately during build to prevent other classifiers from incorrectly clustering them.

## Score Design Guidelines

### What Makes a Good Score

| Property | Good for Scoring | Bad for Scoring |
|----------|-----------------|-----------------|
| Block size | ✅ Intrinsic | |
| Block position | ✅ Intrinsic | |
| Block color | ✅ Intrinsic | |
| Text content | ✅ Intrinsic | |
| "Has 3 children" | | ❌ Relationship |
| "Best diagram is X" | | ❌ Pre-assignment |
| "Overlaps with Y" | ⚠️ Use cautiously | |

### Overlap Considerations

Overlap-based scoring is borderline:

- ✅ OK: "Block overlaps expected region" (region is fixed)
- ⚠️ Caution: "Block overlaps another candidate" (candidate may be consumed)
- ❌ Avoid: "Block's best overlapping partner is X" (pre-assignment)

## Migration Checklist

When refactoring existing classifiers:

- [ ] Remove child candidate fields from score classes
- [ ] Remove pre-assignment logic from `score()` methods
- [ ] Move relationship discovery to `build()` methods
- [ ] Add availability checks (`is_consumed`) in build
- [ ] Ensure atomic consumption of related blocks
- [ ] Add graceful failure when children unavailable
- [ ] Update tests to reflect new behavior

## Examples

### Before (Anti-pattern)

```python
@dataclass
class StepScore:
    candidate: Block
    diagram_candidate: Block | None  # ❌ Pre-assigned
    parts_list_candidate: Block | None  # ❌ Pre-assigned
    
    def overall_score(self) -> float:
        score = 0.5
        if self.diagram_candidate:  # ❌ Based on relationship
            score += 0.3
        return score
```

### After (Correct pattern)

```python
@dataclass  
class StepScore:
    candidate: Block
    intrinsic_score: float  # Based on candidate properties only
    
    def overall_score(self) -> float:
        return self.intrinsic_score

class StepClassifier:
    def build(self, score: StepScore, result: ClassificationResult) -> Step | None:
        # Find diagram NOW
        diagram = self._find_available_diagram(score.candidate, result)
        if diagram is None:
            return None  # Can't build without diagram
        
        result.consume(diagram)
        return Step(candidate=score.candidate, diagram=diagram)
```

---

## Current Classifier Audit

This section documents the current state of each classifier and how well it aligns with the design principles above.

### ✅ Good - Intrinsic Scoring Only

| Classifier | Scoring Criteria | Notes |
|------------|------------------|-------|
| **StepNumberClassifier** | Text pattern, font size, position (not in bottom 10%) | ✅ Pure intrinsic properties |
| **PageNumberClassifier** | Text pattern, font size, bottom band position, corner distance | ✅ Pure intrinsic properties |
| **BagNumberClassifier** | Text pattern, font size, top-left position | ✅ Pure intrinsic properties |
| **PartCountClassifier** | Text pattern ("2x"), font size | ✅ Pure intrinsic properties |
| **ArrowClassifier** | Shape (3-4 line items), size, fill color, finds shaft at score time | ✅ Intrinsic - shaft is an extension of the same element |
| **DiagramClassifier** | Image blocks, filters out >95% page area | ✅ Intrinsic - clustering is correctly deferred to build time |
| **RotationSymbolClassifier** | Size (~46px), aspect ratio (~1.0) | ✅ Recently refactored - proximity to diagram removed |

### ⚠️ Mixed - Some Pre-assignment but Acceptable

| Classifier | Scoring Criteria | Issues |
|------------|------------------|--------|
| **StepClassifier** | Step number + parts_list proximity/alignment. Diagrams found at build time | ✅ Recently refactored! But still stores `parts_list_candidate` in score object. |

### ❌ Problematic - Pre-assigns Child Candidates

| Classifier | Scoring Criteria | Issues |
|------------|------------------|--------|
| **PartsListClassifier** | Contains Parts candidates | ❌ `_PartsListScore.part_candidates: list[Candidate]` - Pre-assigns which parts belong to this list. If parts get consumed elsewhere, build fails. |
| **PartsClassifier** | Pairs part_count with part_image (above + aligned) | ❌ `_PartPairScore` stores `part_count_candidate`, `part_image_candidate`, `part_number_candidate`, `piece_length_candidate` - Full pre-assignment of all children. |
| **SubAssemblyClassifier** | White/light box, contains step_count, diagrams | ❌ **Most problematic.** Pre-assigns: `step_count_candidate`, `diagram_candidate`, `step_number_candidates`, `diagram_candidates`, `images_inside`, `arrow_candidate` - Stores 6 different child references! |

### Migration Priority

1. **SubAssemblyClassifier** - Highest priority, most pre-assignment
2. **PartsClassifier** - High priority, pre-assigns 4 children
3. **PartsListClassifier** - Medium priority, pre-assigns parts list
