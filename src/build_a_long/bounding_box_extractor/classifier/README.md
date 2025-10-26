# Element Classifier

The element classifier provides rule-based heuristics to automatically label and categorize elements extracted from PDF pages.

## Overview

After the bounding box extractor identifies elements on a page (text, images, drawings), the classifier applies a pipeline of heuristic rules to determine what each element represents. This is done by analyzing:

- Position on the page
- Text content patterns
- Size and dimensions
- Relationships with other elements

The classifiers run in a fixed, enforced order because later stages depend on labels produced by earlier stages.

## Classifier Pipeline

The classifier pipeline runs in a fixed order, enforced at initialization:

1.  **PageNumberClassifier**: Identifies page numbers.
    -   **Outputs**: `"page_number"`
    -   **Requires**: None
2.  **PartCountClassifier**: Detects part-count text (e.g., "2x", "3X").
    -   **Outputs**: `"part_count"`
    -   **Requires**: None
3.  **StepNumberClassifier**: Identifies step numbers.
    -   **Outputs**: `"step_number"`
    -   **Requires**: `"page_number"` (uses page number size as context)
4.  **PartsListClassifier**: Identifies the drawing region for the parts list.
    -   **Outputs**: `"parts_list"`
    -   **Requires**: `"step_number"`, `"part_count"`
5.  **PartsImageClassifier**: Associates part counts with their corresponding images.
    -   **Outputs**: `"part_image"`
    -   **Requires**: `"parts_list"`, `"part_count"`

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
from build_a_long.bounding_box_extractor.classifier import classify_elements
from build_a_long.bounding_box_extractor.extractor import extract_bounding_boxes

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

-   `classify_elements(pages)`: The main entry point that runs the entire classification pipeline on all pages.
-   `ClassificationOrchestrator`: Manages the stateful, iterative classification process for a single page.
-   `Classifier`: A stateless class that holds the list of individual classifiers and runs them in order.
-   `LabelClassifier`: An abstract base class that all specific classifiers (like `PageNumberClassifier`) must implement. It defines the interface for calculating scores and assigning labels.

## Adding New Classifiers

To add a new classifier:

1.  Create a new file (e.g., `my_new_classifier.py`).
2.  Implement a class that inherits from `LabelClassifier`.
3.  Define the `outputs` and `requires` class attributes to declare its position in the pipeline.
4.  Implement the `calculate_scores` and `classify` methods.
5.  Instantiate the new classifier in the `Classifier.__init__` method in `classifier.py`, ensuring it is added in an order that respects its dependencies.
6.  Write comprehensive unit tests.
7.  Update this README.

## Integration

The classifier is automatically integrated into the main bounding box extraction pipeline:

1.  `extract_bounding_boxes()` extracts all elements from the PDF.
2.  `classify_elements()` labels the elements in-place.
3.  Results can be saved to JSON with labels included.
4.  Annotated images can be rendered showing the classifications.

## Testing

Run tests with:

```bash
pants test src/build_a_long/bounding_box_extractor/classifier:tests
```

## Design Principles

-   **Rule-based**: Uses deterministic heuristics rather than ML models.
-   **In-place modification**: Labels are added to existing `Element` objects.
-   **Extensible**: The pipeline architecture makes it easy to add new classification rules.
-   **Testable**: Each classifier is independently testable.
-   **Dependency-aware**: The pipeline enforces the order of execution based on declared dependencies.