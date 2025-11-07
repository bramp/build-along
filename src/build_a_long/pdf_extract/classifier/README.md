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
from build_a_long.pdf_extract.classifier.lego_page_builder import build_page
from build_a_long.pdf_extract.extractor import extract_bounding_boxes

# 1. Extract elements from PDF
with pymupdf.open("instructions.pdf") as doc:
    pages = extract_bounding_boxes(doc, None)

# 2. Classify elements
results = classify_pages(pages)

# 3. Build structured hierarchy
for page_data, result in zip(pages, results):
    page = build_page(page_data, result)
    
    # Access structured LEGO elements
    if page.page_number:
        print(f"Page {page.page_number.value}")
    
    for step in page.steps:
        print(f"  Step {step.step_number.value}")
        for part in step.parts_list.parts:
            print(f"    {part.count.count}x")
```

## Classification Pipeline

The classifier runs in a fixed order, with later stages depending on earlier ones:

1. **PageNumberClassifier** - Identifies page numbers
   - Outputs: `"page_number"`
   - Requires: None

2. **PartCountClassifier** - Detects part-count text (e.g., "2x", "3X")
   - Outputs: `"part_count"`
   - Requires: None

3. **StepNumberClassifier** - Identifies step numbers
   - Outputs: `"step_number"`
   - Requires: `"page_number"` (uses page number size as context)

4. **PartsListClassifier** - Identifies the drawing region for the parts list
   - Outputs: `"parts_list"`
   - Requires: `"step_number"`, `"part_count"`

5. **PartsImageClassifier** - Associates part counts with their corresponding images
   - Outputs: `"part_image"`
   - Requires: `"parts_list"`, `"part_count"`

## Candidate-Based Architecture

### Why Candidates?

The classifier uses a **candidate-based architecture** where classifiers construct `LegoPageElement` objects during the scoring phase rather than deferring to separate classification or building phases. This solves several problems:

**Before (Two-Phase)**:

- Scoring: Calculate scores for elements
- Classification: Parse and label elements based on scores
- Building: Re-parse labeled elements to construct objects
- **Problem**: Duplicate parsing, information lost, inconsistent results, no re-evaluation

**After (Single-Phase with Candidates)**:

- Scoring: Calculate scores AND construct candidates with LegoPageElements
- Classification: Pick winners from pre-built candidates
- Building: Use pre-constructed objects directly
- **Benefit**: Single parse, consistency guaranteed, full decision trail with rejection reasons, re-evaluation enabled

### Core Types

#### Candidate

Represents a potential classification decision with complete metadata:

```python
@dataclass
class Candidate:
    source_element: Element                    # PDF element being classified
    label: str                                 # Classification label
    score: float                               # Confidence score (0.0-1.0)
    score_details: Any                         # Detailed score breakdown
    constructed: Optional[LegoPageElement]     # Constructed element (or None if failed)
    failure_reason: Optional[str]              # Why construction failed
    is_winner: bool = False                    # Whether selected as best
```

#### ClassificationResult

Enhanced to preserve all classification decisions:

```python
@dataclass
class ClassificationResult:
    labeled_elements: Dict[Element, str]                 # Element → label
    removal_reasons: Dict[int, RemovalReason]            # Element → removal reason
    constructed_elements: Dict[Element, LegoPageElement] # NEW: Pre-constructed objects
    candidates: Dict[str, List[Candidate]]               # NEW: All candidates by label
```

**Key Methods**:

- `get_best_candidate(label)`: Get the winning candidate for a label
- `get_alternative_candidates(label)`: Get non-winning candidates for debugging

### Implementation Pattern

Each classifier follows a two-phase pattern:

#### Phase 1: Evaluation (`evaluate`)

Evaluates elements and creates candidates with constructed LegoElements:

```python
def evaluate(self, page_data, labeled_elements, candidates):
    candidate_list = []
    
    for element in page_data.elements:
        # 1. Score the element
        score_obj = self._calculate_score(element)
        
        # 2. Try to construct LegoElement
        constructed = None
        failure_reason = None
        
        try:
            constructed = self._construct_element(element)
        except Exception as e:
            failure_reason = f"Construction failed: {e}"
        
        # 3. Create candidate with all metadata
        candidate = Candidate(
            source_element=element,
            label=self.label,
            score=score_obj.combined_score(self.config),
            score_details=score_obj,
            constructed=constructed,
            failure_reason=failure_reason,
            is_winner=False,  # Will be set in classify()
        )
        candidate_list.append(candidate)
    
    # Store candidates for classification phase
    candidates[self.label] = candidate_list
```

#### Phase 2: Classification (`classify`)

Picks winners from pre-built candidates:

```python
def classify(self, page_data, labeled_elements, removal_reasons,
             hints, constructed_elements, candidates):
    # 1. Get pre-built candidates
    candidate_list = candidates.get(self.label, [])
    
    # 2. Apply hints to filter/re-rank (optional)
    candidate_list = self._apply_hints(candidate_list, hints)
    
    # 3. Select the winner
    winner = self._select_winner(candidate_list)
    
    # 4. Store results
    if winner:
        winner.is_winner = True
        labeled_elements[winner.source_element] = self.label
        constructed_elements[winner.source_element] = winner.constructed
        
        # Cleanup: remove similar/child elements
        self.classifier._remove_child_bboxes(...)
        self.classifier._remove_similar_bboxes(...)
```

See `PageNumberClassifier` for a complete reference implementation.

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


## Classifier Details

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

### Parts List Classifier

**Purpose**: Identifies the drawing region(s) that represent a step's parts list.

**Heuristics**:

- Looks for `Drawing` elements located above a detected step number.
- Requires the drawing to contain at least one `part_count` text.
- Prefers the drawing with the closest vertical proximity to the step number.

**Label**: `"parts_list"`

### Parts Image Classifier

**Purpose**: Associates each `part_count` text with its corresponding part `Image`.

**Heuristics**:

- For each `part_count`, it looks for an `Image` that is directly above it and roughly left-aligned.
- It enforces a one-to-one mapping between part counts and images.

**Label**: `"part_image"`

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
├── lego_page_builder.py             # Builds Page from classification results
└── types.py                         # Shared data types
```

### Key Components

- `classify_elements(pages)`: The main entry point that runs the entire classification pipeline on all pages.
- `ClassificationOrchestrator`: Manages the stateful, iterative classification process for a single page.
- `Classifier`: A stateless class that holds the list of individual classifiers and runs them in order.
- `LabelClassifier`: An abstract base class that all specific classifiers (like `PageNumberClassifier`) must implement. It defines the interface for calculating scores and assigning labels.

## Adding New Classifiers

To add a new classifier following the two-phase candidate pattern:

1. **Create a new file** (e.g., `my_new_classifier.py`)
2. **Inherit from `LabelClassifier`**
3. **Define class attributes**:
   - `outputs`: Set of labels this classifier produces
   - `requires`: Set of labels this classifier depends on
4. **Implement `evaluate`**:
   - Score all elements
   - Attempt to construct LegoPageElements
   - Create Candidate objects with scores, constructed elements, and failure reasons
   - Store candidates in the `candidates` dict
5. **Implement `classify`**:
   - Get pre-built candidates from `candidates` dict
   - Apply hints to filter candidates (optional)
   - Select the winning candidate(s)
   - Mark winners and store in `labeled_elements` and `constructed_elements`
   - Clean up similar/child bboxes
6. **Add to pipeline**: Instantiate in `Classifier.__init__` in dependency order
7. **Write tests**: Unit tests for scoring, construction, candidate selection, and edge cases
8. **Update documentation**: Add classifier details to this README

See `PageNumberClassifier`, `StepNumberClassifier`, or `PartCountClassifier` for reference implementations.

## Page Builder

After classification, the `lego_page_builder` module constructs a structured hierarchy of LEGO-specific elements. With the candidate-based architecture, builders can use pre-constructed `LegoPageElement` objects from `ClassificationResult.constructed_elements`, eliminating duplicate parsing and guaranteeing consistency.

The builder returns a `Page` object (a `LegoPageElement`) that represents the complete structured view of the page.

### Transformations

```python
# Classifiers construct elements during classification
Text("5") + scores → Candidate(constructed=PageNumber(value=5))

# Builders use pre-constructed elements
result.constructed_elements[text_element]  →  PageNumber(value=5)
page.page_number = result.constructed_elements[text_element]
```

Traditional transformations (still supported for backwards compatibility):

```python
# Text with page_number label becomes PageNumber
Text("5") + label="page_number"  →  PageNumber(value=5)

# Text with step_number label becomes StepNumber inside a Step
Text("2") + label="step_number"  →  StepNumber(value=2) inside Step

# Part count text and image become Part inside PartsList
Text("3x") + Image() + labels  →  Part(count=PartCount(3), ...)

# Drawing with parts_list label becomes PartsList container
Drawing() + label="parts_list"  →  PartsList(parts=[...])
```

### Spatial Relationships

The page builder uses `bbox.fully_inside()` to determine containment. For example, parts are only included in a parts list if their bounding boxes are fully inside the parts list's bounding box:

```text
┌────────────────────────────────────┐
│  PartsList (Drawing)               │
│                                    │
│  ┌──────┐  ┌──────────┐           │
│  │ 2x   │  │  Image   │  ← Part 1 │
│  └──────┘  └──────────┘           │
│                                    │
│  ┌──────┐  ┌──────────┐           │
│  │ 1x   │  │  Image   │  ← Part 2 │
│  └──────┘  └──────────┘           │
└────────────────────────────────────┘
```

### Type Hierarchy

```text
LegoPageElement (base)
├── Page (top-level container)
│   ├── page_number: PageNumber | None
│   ├── steps: List[Step]
│   ├── parts_lists: List[PartsList]
│   ├── warnings: List[str]
│   └── unprocessed_elements: List[Element]
├── PageNumber
│   └── value: int
├── Step
│   ├── step_number: StepNumber
│   ├── parts_list: PartsList
│   └── diagram: Diagram
├── StepNumber
│   └── value: int
├── PartsList
│   └── parts: List[Part]
├── Part
│   ├── count: PartCount
│   ├── name: str | None
│   └── number: str | None
├── PartCount
│   └── count: int
└── Diagram
    └── bbox: BBox
```

### Usage Examples

```python
from build_a_long.pdf_extract.classifier.lego_page_builder import build_page

# After extracting and classifying
page = build_page(page_data, result)

# Access structured elements
if page.page_number:
    print(f"Page {page.page_number.value}")

# Steps
for step in page.steps:
    print(f"Step {step.step_number.value}")
    for part in step.parts_list.parts:
        print(f"  {part.count.count}x")

# Standalone parts lists
for parts_list in page.parts_lists:
    total = sum(p.count.count for p in parts_list.parts)
    print(f"Parts list with {total} total items")

# Check for issues
if page.warnings:
    for warning in page.warnings:
        print(f"Warning: {warning}")
```

### Error Handling

All parsing errors are non-fatal and result in warnings:

```python
page = build_page(page_data, result)

# Check for parsing errors
for warning in page.warnings:
    if "Could not parse" in warning:
        # Handle parse error
        pass
```

The `Page` object includes:

- **warnings**: List of issues encountered (e.g., multiple page numbers, parse failures)
- **unprocessed_elements**: Elements that were labeled but not converted (helps identify gaps)

### Current Limitations

The page builder is under active development. Current limitations:

1. **Step Construction** - Steps currently only have basic structure. Future work will:
   - Associate diagrams with steps based on spatial proximity
   - Link parts lists to steps when appropriate
   - Handle multi-step layouts

2. **Part Details** - Parts currently only extract count. Future enhancements:
   - Extract part name from nearby text
   - Extract part number (LEGO piece ID)
   - Associate part color information

3. **Missing Elements** - Some LEGO-specific elements are not yet extracted:
   - Bag numbers
   - New bag graphics
   - Sub-assemblies
   - Progress bars
   - Information callouts

## Current Migration Status

The candidate-based architecture has been successfully implemented:

### ✅ Completed

- **Core Infrastructure**:
  - `types.py`: `Candidate` dataclass and enhanced `ClassificationResult`
  - `classifier.py`: Creates and passes `constructed_elements` and `candidates` dicts
  - `label_classifier.py`: Updated `calculate_scores()` signature with `candidates` parameter
  - `text_extractors.py`: Shared text parsing functions

- **Classifiers** (two-phase pattern with candidates):
  - `page_number_classifier.py`: Scores, constructs candidates, picks winners with rejection reasons
  - `step_number_classifier.py`: Scores with size filtering, constructs candidates, picks winners
  - `part_count_classifier.py`: Pattern matching, constructs candidates, picks all valid matches
  - `parts_list_classifier.py`: Proximity scoring, constructs candidates, picks best per step
  - `parts_image_classifier.py`: Compatible signature (uses pairing logic internally)

### ⏳ Future Work

- **Builders** (to be updated to use `constructed_elements`):
  - `lego_page_builder.py`: Currently parses elements; should use pre-constructed objects

- **Enhancements**:
  - Implement hints support for re-evaluation
  - Add user correction API
  - Expand debugging UI with candidate alternatives

All classifiers now create candidates during scoring, enabling single-parse consistency and rich debugging information.

## Testing

Run all tests:

```bash
pants test src/build_a_long/pdf_extract/classifier::
```

Run specific tests:

```bash
pants test src/build_a_long/pdf_extract/classifier/lego_page_builder_test.py
pants test src/build_a_long/pdf_extract/classifier/classifier_test.py
```

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
