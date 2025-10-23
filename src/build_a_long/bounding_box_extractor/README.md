# Bounding Box Extractor

This module extracts bounding boxes and structural information from LEGO instruction PDFs.

## Directory Structure

```text
bounding_box_extractor/
├── main.py                 # CLI entry point for the extractor
├── drawing/                # Drawing and visualization submodule
│   ├── drawing.py         # Functions to draw bounding boxes on PDF pages
│   └── BUILD              # Pants build configuration
├── extractor/              # PDF extraction submodule
│   ├── bbox.py            # BBox dataclass for bounding box representation
│   ├── bbox_test.py       # Tests for BBox class
│   ├── page_elements.py   # Page element type definitions (StepNumber, Drawing, etc.)
│   ├── page_elements_test.py  # Tests for page elements
│   ├── hierarchy.py       # Containment hierarchy building logic
│   ├── hierarchy_test.py  # Tests for hierarchy building
│   ├── extractor.py       # Core extraction logic using PyMuPDF
│   ├── extractor_test.py  # Tests for extraction functionality
│   └── BUILD              # Pants build configuration
└── parser/                 # Input parsing submodule
    ├── parser.py          # Page range parsing utilities
    ├── parser_test.py     # Tests for parser functionality
    └── BUILD              # Pants build configuration
```

## Module Organization

### `main.py`

Command-line interface for extracting bounding boxes from PDFs. Handles argument parsing and orchestrates the extraction process.

### Submodules

#### `drawing/`

Handles visualization and rendering of extracted bounding boxes:

- Draws bounding boxes on PDF page images
- Color-codes elements by nesting depth
- Labels elements with their types

#### `extractor/`

Core PDF extraction functionality and data models:

**Data Models:**

- `bbox.py`: Defines the `BBox` dataclass representing a bounding box with coordinates (x0, y0, x1, y1) and spatial relationship methods
- `page_elements.py`: Defines typed page element classes:
  - `StepNumber`: Instruction step numbers
  - `Drawing`: Image blocks
  - `PathElement`: Vector drawing paths
  - `Unknown`: Unclassified elements
- `hierarchy.py`: Builds containment hierarchies from page elements based on spatial relationships

**Extraction Logic:**

- `extractor.py`: Uses PyMuPDF (fitz) to parse PDF structure, extract text/images/vector graphics, classify elements, and build hierarchical representations

#### `parser/`

Input parsing utilities:

- Parses page range specifications (e.g., "5", "5-10", "10-", "-5")
- Validates and normalizes user input
- Provides flexible page selection for processing

## Usage

```bash
# Extract from entire PDF
pants run src/build_a_long/bounding_box_extractor:main path/to/file.pdf

# Extract specific pages
pants run src/build_a_long/bounding_box_extractor:main path/to/file.pdf --pages 5-10

# Specify output directory
pants run src/build_a_long/bounding_box_extractor:main path/to/file.pdf --output-dir ./output

# Filter by element types
pants run src/build_a_long/bounding_box_extractor:main path/to/file.pdf --include-types text,image
```

## Testing

Run all tests:

```bash
pants test src/build_a_long/bounding_box_extractor::
```

Run specific submodule tests:

```bash
pants test src/build_a_long/bounding_box_extractor/extractor:tests
pants test src/build_a_long/bounding_box_extractor/parser:tests
```

## Output

The extractor produces:

1. **JSON file**: Structured data containing all extracted elements and hierarchies
2. **PNG images** (if `--output-dir` specified): Visual representation with bounding boxes drawn and labeled
