# LEGO Instruction Classifier

This module provides classification and structured page building for LEGO instruction PDFs. It applies rule-based heuristics to label extracted PDF elements and constructs a hierarchical representation of LEGO-specific components.

## Overview

The classifier transforms flat lists of extracted PDF elements into labeled, structured LEGO instruction pages using a **candidate-based architecture**:

1. **Classification** - Scores elements, constructs `LegoPageElement` objects, and selects the best candidates
2. **Page Building** - Uses pre-constructed elements to assemble hierarchical `Page` objects

This single-parse approach guarantees consistency between classification and construction, enables re-evaluation, and provides rich debugging information.

See the [parent README](../README.md) for overall pipeline architecture.

## Quick Start

```python
import pymupdf
from build_a_long.pdf_extract.classifier import classify_pages
from build_a_long.pdf_extract.extractor import extract_bounding_boxes

# 1. Extract elements from PDF
with pymupdf.open("instructions.pdf") as doc:
    pages = extract_bounding_boxes(doc, None)

# 2. Classify elements
results = classify_pages(pages)

# 3. Build structured hierarchy
for result in results:
    page = result.page
    
    # Access structured LEGO elements
    if page.page_number:
        print(f"Page {page.page_number.value}")
    
    for step in page.steps:
        print(f"  Step {step.step_number.value}")
        for part in step.parts_list.parts:
            print(f"    {part.count.count}x")
```

## Classification Pipeline

The classifier runs in topologically-sorted order based on declared dependencies. Here are some key examples:

**Atomic Classifiers** (identify individual elements from source blocks):

- **PageNumberClassifier** - Identifies page numbers from Text blocks
- **PartCountClassifier** - Detects part-count text (e.g., "2x", "3X")
- **PartsImageClassifier** - Wraps Image blocks as potential part diagrams

**Composite Classifiers** (combine other elements):

- **PartsClassifier** - Pairs PartCount + PartImage to create Part elements
  - Requires: `"part_count"`, `"part_image"`, `"part_number"`, `"piece_length"`
- **StepClassifier** - Combines StepNumber + PartsList + Diagram
  - Requires: `"step_number"`, `"parts_list"`, `"diagram"`

The pipeline automatically determines execution order via topological sort of the dependency graph. See the "Classifier Reference" section below for detailed descriptions of each classifier.

## Candidate-Based Architecture

### Core Types

#### Candidate

Represents a potential classification decision with complete metadata:

```python
@dataclass
class Candidate:
    source_blocks: Blocks.                     # PDF element being classified
    label: str                                 # Classification label
    score: float                               # Confidence score (0.0-1.0)
    score_details: Any                         # Detailed score breakdown
    constructed: Optional[LegoPageElement]     # Constructed element (or None if failed)
    failure_reason: Optional[str]              # Why construction failed
```

#### ClassificationResult

Enhanced to preserve all classification decisions:

```python
@dataclass
class ClassificationResult:
    labeled_elements: Dict[Element, str]                 # Element → label
    removal_reasons: Dict[int, RemovalReason]            # Element → removal reason
    constructed_elements: Dict[Element, LegoPageElement] # Pre-constructed objects
    candidates: Dict[str, List[Candidate]]               # All candidates by label
```

**Key Methods**:

- `get_best_candidate(label)`: Get the winning candidate for a label
- `get_alternative_candidates(label)`: Get non-winning candidates for debugging

### Implementation Pattern

Each classifier implements two core methods:

#### Method 1: Scoring (`_score`)

Scores elements and creates candidates:

```python
class MyClassifier(LabelClassifier):
    output = "my_label"
    requires = frozenset({"dependency_label"})  # Or empty set
    
    def _score(self, result: ClassificationResult) -> None:
        """Create and score candidates."""
        # 1. Get dependency candidates if needed
        dep_candidates = result.get_scored_candidates("dependency_label")
        
        # 2. Score and create candidates
        for element in result.page_data.blocks:
            score_details = self._calculate_score(element)
            
            result.add_candidate(
                Candidate(
                    bbox=element.bbox,
                    label="my_label",
                    score=score_details.total_score,
                    score_details=score_details,  # MUST be non-None
                    source_blocks=[element],  # Or [] for composite elements
                )
            )
```

#### Method 2: Construction (`build`)

Constructs LegoPageElement from a winning candidate:

```python
    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> MyLegoElement:
        """Construct element from candidate."""
        score_details = candidate.score_details
        assert isinstance(score_details, MyScoreDetails)
        
        # For composite elements, build children
        child = result.build(score_details.child_candidate)
        assert isinstance(child, ChildElement)
        
        return MyLegoElement(
            bbox=candidate.bbox,
            child=child,
        )
```

See `PartsClassifier` or `PartsImageClassifier` for complete reference implementations.

### Benefits

1. **Single Parse**: Text and elements parsed once during scoring → no duplicate parsing
2. **Consistency**: Builders use pre-constructed objects from classification
3. **Better Debugging**: All candidates preserved with scores, failures, rejection reasons
4. **Re-evaluation**: Can try second-best candidate if first fails downstream
5. **Hints Support**: External code can guide classification (e.g., user corrections)
6. **Type Safety**: Constructed elements have known, validated types
7. **Clear Separation**: Scoring creates candidates; classification picks winners

### Migration Status

✅ **Complete**: All core classifiers migrated to two-phase pattern:

- `PageNumberClassifier` - Candidates with rejection reasons
- `StepNumberClassifier` - Candidates with size/position filtering
- `PartCountClassifier` - Candidates with pattern matching
- `PartsListClassifier` - Candidates with proximity scoring
- `PartsImageClassifier` - Compatible signature (uses old pattern internally)

## Classifier Reference

### Page Number Classifier

**Purpose**: Identifies text elements that represent page numbers.

**Heuristics**:

- Scores text based on patterns like `"5"` or `"01"`.
- Scores position based on proximity to the bottom corners of the page.
- Prefers candidates whose numeric value matches the actual page number.

**Label**: `"page_number"`

### Part Count Classifier

**Purpose**: Detects part-count text like "2x", "3X", or "5×".

**Heuristics**:

- Matches text that fits the pattern `\d{1,3}\s*[x×]`.

**Label**: `"part_count"`

### Step Number Classifier

**Purpose**: Identifies step numbers in instruction sequences.

**Heuristics**:

- Matches numeric text (e.g., "1", "23").
- Scores based on font size relative to the page number (expects step numbers to be larger).

**Label**: `"step_number"`

### Parts Image Classifier

**Purpose**: Identifies images that could be part diagrams.

**Architecture**: Atomic element (tracks Image source blocks)

**Heuristics**:

- Creates PartImage candidates from all Image blocks on the page
- No filtering or scoring logic (score = 1.0 for all images)
- PartsClassifier handles pairing with part counts

**Label**: `"part_image"`

**Dependencies**: None

### Parts Classifier

**Purpose**: Pairs PartCount + PartImage candidates to create Part elements.

**Architecture**: Composite element (no source blocks, composed of child elements)

**Heuristics**:

- Finds PartImage candidates above and aligned with PartCount candidates
- Scores based on vertical distance and horizontal alignment
- Also incorporates PartNumber and PieceLength if found nearby
- One PartCount + PartImage pair = one Part candidate

**Label**: `"part"`

**Dependencies**: `"part_count"`, `"part_number"`, `"piece_length"`, `"part_image"`

## Implementation Architecture

The classifier is architected as a pipeline of independent classifier modules managed by an orchestrator.

```text
classifier/
├── __init__.py                      # Public API and logging setup
├── BUILD                            # Pants build configuration
├── README.md                        # This file
├── TEXT_EXTRACTION_REFACTORING.md   # Text extraction refactoring details
├── classifier.py                    # Main classification orchestrator
├── classifier_test.py               # Unit tests
├── label_classifier.py              # Abstract base class for classifiers
├── page_number_classifier.py        # Page number classification logic
├── part_count_classifier.py         # Part count classification logic
├── parts_image_classifier.py        # Part image classification logic
├── parts_list_classifier.py         # Parts list classification logic
├── step_number_classifier.py        # Step number classification logic
├── text_extractors.py               # Shared text parsing functions
└── types.py                         # Shared data types
```

### Key Components

- `classify_elements(pages)`: The main entry point that runs the entire classification pipeline on all pages.
- `ClassificationOrchestrator`: Manages the stateful, iterative classification process for a single page.
- `Classifier`: A stateless class that holds the list of individual classifiers and runs them in order.
- `LabelClassifier`: An abstract base class that all specific classifiers (like `PageNumberClassifier`) must implement. It defines the interface for calculating scores and assigning labels.

## Adding New Classifiers

To add a new classifier:

1. **Create a new file** (e.g., `my_new_classifier.py`)
2. **Inherit from `LabelClassifier`**
3. **Define class attributes**:
   - `output`: The label this classifier produces (string)
   - `requires`: frozenset of labels this classifier depends on
4. **Implement `_score()`**:
   - Get dependency candidates via `result.get_scored_candidates(label)`
   - Score elements and create Candidate objects
   - Set `score_details` to a dataclass (never None)
   - Set `source_blocks` appropriately (atomic vs composite)
   - Use `result.add_candidate()` to register candidates
5. **Implement `build()`**:
   - Extract score_details from candidate
   - Build child elements via `result.build(child_candidate)` for composites
   - Assert types of child elements
   - Construct and return the LegoPageElement
6. **Add to pipeline**: Register in classifier list (topological sort handles order)
7. **Write tests**: Unit tests for scoring, construction, and edge cases
8. **Verify**: Run `classifier_rules_test.py` to check source block tracking
9. **Update documentation**: Add classifier details to this README

See `PartsImageClassifier` (atomic) or `PartsClassifier` (composite) for reference implementations.

## Architecture Status

The candidate-based architecture with composite element pattern is fully implemented:

### ✅ Completed

- **Core Infrastructure**:
  - `classification_result.py`: Candidate tracking, `get_scored_candidates()`, `build()` method
  - `label_classifier.py`: Abstract base with `_score()` and `build()` interface
  - `topological_sort.py`: Dependency resolution for classifier pipeline
  - `text_extractors.py`: Shared text parsing functions

- **Classifiers** (score + build pattern):
  - `page_number_classifier.py`: Atomic element, tracks Text source
  - `step_number_classifier.py`: Atomic element, tracks Text source
  - `part_count_classifier.py`: Atomic element, tracks Text source
  - `part_number_classifier.py`: Atomic element, tracks Text source
  - `piece_length_classifier.py`: Atomic element, tracks Text source
  - `parts_image_classifier.py`: Atomic element, tracks Image source
  - `parts_classifier.py`: **Composite element**, pairs PartCount + PartImage
  - `parts_list_classifier.py`: **Composite element**, groups Part elements
  - `diagram_classifier.py`: Atomic element, tracks Drawing source
  - `step_classifier.py`: **Composite element**, combines StepNumber + PartsList + Diagram

- **Source Block Tracking**:
  - Atomic elements set `source_blocks=[source]`
  - Composite elements set `source_blocks=[]`
  - Enforced by `classifier_rules_test.py`

### ⏳ Future Work

- **Enhancements**:
  - Hints support for user corrections
  - Re-evaluation of alternative candidates
  - Enhanced debugging UI showing candidate alternatives
  - Progress bars and bag number classification

## Testing

Run all tests:

```bash
pants test src/build_a_long/pdf_extract/classifier::
```

Run specific tests:

```bash
pants test src/build_a_long/pdf_extract/classifier/classifier_test.py
```

## Composite Elements and Source Block Tracking

### Key Concepts

The classifier architecture distinguishes between **atomic elements** (single source block) and **composite elements** (multiple child elements). This distinction is critical for proper source block tracking and conflict resolution.

### Source Block Tracking Rules

**RULE 1: One Source Block → One Element**

Each source block (Text, Image, Drawing) should map to at most one element in the final Page tree. This is enforced by `classifier_rules_test.py`.

**RULE 2: Atomic Elements Track Sources**

Atomic elements (created from a single source block) should set `source_blocks=[source]`:

- `PartCount` → tracks the Text block
- `PartImage` → tracks the Image block  
- `StepNumber` → tracks the Text block
- `PageNumber` → tracks the Text block

**RULE 3: Composite Elements Don't Track Sources**

Composite elements (made of other LegoPageElements) should set `source_blocks=[]`:

- `Part` → composed of PartCount + PartImage + PartNumber + PieceLength
- `Step` → composed of StepNumber + PartsList + Diagram
- `PartsList` → composed of multiple Part elements

**WHY**: If a Part has `source_blocks=[image]`, and its child PartImage also has `source_blocks=[image]`, the same Image would be mapped to two elements, violating RULE 1.

### Example: Part/PartImage Architecture

The `Part` element demonstrates the composite pattern:

```python
# WRONG - Part claims the Image source block
result.add_candidate(
    Candidate(
        bbox=bbox,
        label="part",
        score=1.0,
        score_details=ps,
        source_blocks=[ps.image],  # ❌ Conflict with PartImage
    )
)

# CORRECT - Part is composite, only children track sources
result.add_candidate(
    Candidate(
        bbox=bbox,
        label="part",
        score=1.0,
        score_details=ps,
        source_blocks=[],  # ✅ Composite element
    )
)
```

The PartImage (child element) properly tracks the source:

```python
result.add_candidate(
    Candidate(
        bbox=img.bbox,
        label="part_image",
        score=1.0,
        score_details=_PartImageScore(image=img),
        source_blocks=[img],  # ✅ Atomic element tracks source
    )
)
```

### Dependency Management and Circular Dependencies

**RULE 4: Topological Sort Requirements**
Classifiers declare dependencies via the `requires` attribute. The pipeline uses topological sort to determine execution order. Circular dependencies will cause a runtime error.

**Example Circular Dependency Problem**:

```python
# BROKEN - Creates cycle
class PartsImageClassifier(LabelClassifier):
    requires = frozenset({"parts_list", "part_count"})  # Depends on parts_list

class PartsListClassifier(LabelClassifier):
    requires = frozenset({"part"})  # Depends on part

class PartsClassifier(LabelClassifier):
    requires = frozenset({"part_image"})  # Depends on part_image

# Cycle: PartsClassifier → PartsImageClassifier → PartsListClassifier → PartsClassifier
```

**Solution**: Break the cycle by removing unnecessary dependencies. Often, a classifier can work with lower-level blocks instead of depending on higher-level classifiers:

```python
# FIXED - No cycle
class PartsImageClassifier(LabelClassifier):
    requires = frozenset()  # No dependencies - works on raw Image blocks
    
    def _score(self, result: ClassificationResult) -> None:
        # Create PartImage candidates from all Images
        images = [e for e in result.page_data.blocks if isinstance(e, Image)]
        for img in images:
            result.add_candidate(...)
```

### Candidate Visibility with score_details

**RULE 5: Candidates Need score_details for Visibility**
When a classifier depends on another classifier's candidates, those candidates MUST have `score_details != None` to be returned by `get_scored_candidates()`.

**Example Problem**:

```python
# PartsImageClassifier creates candidates
result.add_candidate(
    Candidate(
        bbox=img.bbox,
        label="part_image",
        score=1.0,
        score_details=None,  # ❌ Will be filtered out!
        source_blocks=[img],
    )
)

# PartsClassifier tries to use them
part_image_candidates = result.get_scored_candidates("part_image")
# Returns [] because score_details=None
```

**Solution**: Always provide score_details:

```python
result.add_candidate(
    Candidate(
        bbox=img.bbox,
        label="part_image",
        score=1.0,
        score_details=_PartImageScore(image=img),  # ✅ Visible to dependents
        source_blocks=[img],
    )
)
```

### Building Composite Elements

When constructing composite elements in `build()`, use `result.build()` to properly track child elements:

```python
def build(
    self, candidate: Candidate, result: ClassificationResult
) -> Part:
    """Construct a Part from its child candidates."""
    ps = candidate.score_details
    assert isinstance(ps, _PartPairScore)
    
    # Build child elements through result.build()
    part_image = result.build(ps.part_image_candidate)
    assert isinstance(part_image, PartImage)
    
    part_count = result.build(ps.part_count_candidate)
    assert isinstance(part_count, PartCount)
    
    # Composite Part element
    return Part(
        bbox=candidate.bbox,
        count=part_count,
        diagram=part_image,
        number=...,
        piece_length=...,
    )
```

### Checklist for New Classifiers

When creating a new classifier, verify:

- [ ] Is this an atomic or composite element?
  - Atomic: Set `source_blocks=[source_block]` in candidates
  - Composite: Set `source_blocks=[]` in candidates
  
- [ ] Does this depend on other classifiers?
  - Add to `requires` set
  - Verify no circular dependencies
  - Use `result.get_scored_candidates()` to get candidates
  
- [ ] Do I create candidates for downstream classifiers?
  - Always set `score_details` to a dataclass (never None)
  - Store candidate references, not constructed elements, in score details
  
- [ ] Does `build()` construct child elements correctly?
  - Use `result.build(child_candidate)` for children
  - Assert type of built children
  
- [ ] Run `classifier_rules_test.py` to verify:
  - No source block conflicts
  - All candidates properly tracked
  - Page tree is valid

## Design Principles

- **Rule-based** - Uses deterministic heuristics rather than ML models
- **Single Parse** - Text parsed once during classification, not again during building
- **Candidate Preservation** - All candidates stored, not just winners (enables debugging and re-evaluation)
- **Type Safety** - Constructed elements have known, validated types
- **Extensible** - Pipeline architecture and hints enable easy addition of new rules
- **Testable** - Each classifier is independently testable with clear inputs/outputs
- **Dependency-aware** - Pipeline enforces execution order based on declared dependencies
- **Non-fatal errors** - Parsing errors produce warnings instead of failing
- **Auditable** - Full decision trail with scores, alternatives, and failure reasons
- **Composite Pattern** - Atomic elements track sources, composite elements compose children
- **Provenance Tracking** - Source blocks traceable through entire classification pipeline
