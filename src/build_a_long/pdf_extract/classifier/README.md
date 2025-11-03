# LEGO Instruction Classifier & Page Builder

This module provides a complete pipeline for extracting, classifying, and structuring LEGO instruction PDF pages. It transforms raw PDF elements into a structured hierarchy of LEGO-specific components.

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

## Architecture Overview

The processing pipeline has three stages:

```text
┌───────────┐
│    PDF    │
└─────┬─────┘
      │
      ▼
┌─────────────────────┐
│    EXTRACTOR        │  Extracts raw elements
│  (pymupdf-based)    │  - Text, Image, Drawing
└──────────┬──────────┘  - BBox for each
           │
           ▼
┌─────────────────────┐
│     PageData        │  Flat list of elements
│  - Text, Image,     │  with bounding boxes
│    Drawing          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   CLASSIFIER        │  Applies heuristics
│  (rule-based)       │  to label elements
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ClassificationResult│  Labeled elements
│  - page_number      │  with relationships
│  - step_number      │
│  - parts_list       │
│  - part_count       │
│  - part_image       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ LEGO PAGE BUILDER   │  Builds structured
│  (lego_page_builder)│  hierarchy
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│        Page         │  Structured LEGO
│  - PageNumber       │  elements
│  - Step[]           │
│  - PartsList[]      │
└─────────────────────┘
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

## Example

```python
from build_a_long.pdf_extract.classifier import classify_elements
from build_a_long.pdf_extract.extractor import extract_bounding_boxes

pages = extract_bounding_boxes(doc, start_page, end_page)
classify_elements(pages)

# After classification, elements will have labels
for page in pages:
    for element in page.elements:
        if element.label:
            print(f"Found {element.label}: {getattr(element, 'text', 'Image/Drawing')}")
```

## Architecture

The classifier is architected as a pipeline of independent classifier modules managed by an orchestrator.

```
classifier/
├── __init__.py                 # Public API and logging setup
├── BUILD                       # Pants build configuration
├── classifier.py               # Main classification orchestrator
├── classifier_test.py          # Unit tests
├── label_classifier.py         # Abstract base class for classifiers
├── page_number_classifier.py   # Page number classification logic
├── part_count_classifier.py    # Part count classification logic
├── parts_image_classifier.py   # Part image classification logic
├── parts_list_classifier.py    # Parts list classification logic
├── step_number_classifier.py   # Step number classification logic
└── types.py                    # Shared data types
```

### Key Components

- `classify_elements(pages)`: The main entry point that runs the entire classification pipeline on all pages.
- `ClassificationOrchestrator`: Manages the stateful, iterative classification process for a single page.
- `Classifier`: A stateless class that holds the list of individual classifiers and runs them in order.
- `LabelClassifier`: An abstract base class that all specific classifiers (like `PageNumberClassifier`) must implement. It defines the interface for calculating scores and assigning labels.

## Adding New Classifiers

To add a new classifier:

1. Create a new file (e.g., `my_new_classifier.py`).
2. Implement a class that inherits from `LabelClassifier`.
3. Define the `outputs` and `requires` class attributes to declare its position in the pipeline.
4. Implement the `calculate_scores` and `classify` methods.
5. Instantiate the new classifier in the `Classifier.__init__` method in `classifier.py`, ensuring it is added in an order that respects its dependencies.
6. Write comprehensive unit tests.
7. Update this README.

## Integration

The classifier is automatically integrated into the main bounding box extraction pipeline:

1. `extract_bounding_boxes()` extracts all elements from the PDF.
2. `classify_elements()` labels the elements in-place.
3. Results can be saved to JSON with labels included.
4. Annotated images can be rendered showing the classifications.

## Page Builder

After classification, the `lego_page_builder` module constructs a structured hierarchy of LEGO-specific elements from the flat list of classified elements. It returns a `Page` object (a `LegoPageElement`) that represents the complete structured view of the page.

### Transformations

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

## Project Structure

```text
classifier/
├── __init__.py                    # Public API
├── classifier.py                  # Classification orchestrator
├── classifier_test.py             # Unit tests
├── lego_page_builder.py           # Hierarchy builder
├── lego_page_builder_test.py      # Hierarchy tests
├── example_hierarchy.py           # Complete working example
├── label_classifier.py            # Abstract base class
├── page_number_classifier.py      # Page number classification
├── part_count_classifier.py       # Part count classification
├── parts_image_classifier.py      # Part image classification
├── parts_list_classifier.py       # Parts list classification
├── step_number_classifier.py      # Step number classification
└── types.py                       # Shared data types
```

## Testing

Run all tests:

```bash
./pants test src/build_a_long/pdf_extract/classifier::
```

Run specific tests:

```bash
./pants test src/build_a_long/pdf_extract/classifier/lego_page_builder_test.py
./pants test src/build_a_long/pdf_extract/classifier/classifier_test.py
```

## See Also

- `example_hierarchy.py` - Complete working example showing the full pipeline
- `lego_page_elements.py` - Definitions of all LEGO-specific element types
- `types.py` - Classification result types

## Design Principles

- **Rule-based** - Uses deterministic heuristics rather than ML models
- **In-place modification** - Labels are added to existing `Element` objects
- **Extensible** - Pipeline architecture makes it easy to add new classification rules
- **Testable** - Each classifier is independently testable
- **Dependency-aware** - Pipeline enforces execution order based on declared dependencies
- **Non-fatal errors** - Parsing errors produce warnings instead of failing
