# Element Classifier

The element classifier provides rule-based heuristics to automatically label and categorize elements extracted from PDF pages.

## Overview

After the bounding box extractor identifies elements on a page (text, images, drawings), the classifier applies heuristic rules to determine what each element represents. This is done by analyzing:

- Position on the page
- Text content patterns
- Size and dimensions
- Relationships with other elements

## Current Classifiers

### Page Number Classifier

**Purpose**: Identifies text elements that represent page numbers.

**Heuristics**:

- Located in the bottom 10% of the page
- Contains numeric text (1-3 digits)
- Positioned in lower-left or lower-right corner (preferred)
- Supports formats like: `"5"`, `"01"`, `"Page 5"`, `"p.5"`

**Label**: `"page_number"`

**Example**:

```python
from build_a_long.bounding_box_extractor.classifier import classify_elements
from build_a_long.bounding_box_extractor.extractor import extract_bounding_boxes

pages = extract_bounding_boxes(doc, start_page, end_page)
classify_elements(pages)

# After classification, text elements may have labels
for page in pages:
    for element in page.elements:
        if isinstance(element, Text) and element.label == "page_number":
            print(f"Page number found: {element.text}")
```

## Future Classifiers

The following classifiers are planned for future implementation:

- **Step Number Classifier**: Identify step numbers in instruction sequences
- **Parts List Classifier**: Detect and extract parts lists
- **Part Count Classifier**: Identify quantity indicators (e.g., "x3", "2x")
- **Title/Heading Classifier**: Identify section titles and headings

## Architecture

The classifier module is structured as:

```
classifier/
├── __init__.py           # Public API
├── classifier.py         # Main classification logic
├── classifier_test.py    # Unit tests
└── BUILD                 # Pants build configuration
```

### Key Functions

- `classify_elements(pages)`: Main entry point that applies all classification rules
- `_classify_page_number(page_data)`: Page number classification logic
- `_is_likely_page_number(text)`: Pattern matching for page number text

## Adding New Classifiers

To add a new classifier:

1. Create a helper function to test if an element matches the pattern (e.g., `_is_likely_step_number`)
2. Create a classification function that finds and labels elements (e.g., `_classify_step_numbers`)
3. Add the classification function call to `classify_elements()`
4. Write comprehensive unit tests
5. Update this README

Example skeleton:

```python
def _is_likely_step_number(text: str) -> bool:
    """Check if text looks like a step number."""
    # Your pattern matching logic
    pass

def _classify_step_numbers(page_data: PageData) -> None:
    """Identify and label step number elements."""
    for element in page_data.elements:
        if isinstance(element, Text) and _is_likely_step_number(element.text):
            # Additional position/context checks
            element.label = "step_number"
```

## Integration

The classifier is automatically integrated into the main bounding box extraction pipeline:

1. `extract_bounding_boxes()` extracts all elements from PDF
2. `classify_elements()` labels the elements
3. Results are saved to JSON with labels included
4. Annotated images can be rendered showing the classifications

## Testing

Run tests with:

```bash
pants test src/build_a_long/bounding_box_extractor/classifier:tests
```

## Design Principles

- **Rule-based**: Uses deterministic heuristics rather than ML models
- **In-place modification**: Labels are added to existing Element objects
- **Defensive**: Handles missing data gracefully (empty pages, no candidates)
- **Testable**: Each classifier function is independently testable
- **Extensible**: Easy to add new classification rules
