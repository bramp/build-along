# Bounding Box Extractor

This module extracts bounding boxes and structural information from LEGO instruction PDFs.

## Directory Structure

```text
bounding_box_extractor/
├── main.py                 # CLI entry point for the extractor
├── classifier/             # Element classification submodule
│   ├── classifier.py      # Main classification logic
│   ├── page_number_classifier.py # Page number classification logic
│   └── ...                # Other individual classifiers
├── drawing/                # Drawing and visualization submodule
│   ├── drawing.py         # Functions to draw bounding boxes on PDF pages
│   └── BUILD              # Pants build configuration
├── extractor/              # PDF extraction submodule
│   ├── bbox.py            # BBox dataclass for bounding box representation
│   ├── page_elements.py   # Raw page element type definitions (Text, Image, etc.)
│   ├── lego_page_elements.py # Structured LEGO element type definitions (Step, Part, etc.)
│   ├── hierarchy.py       # Containment hierarchy building logic
│   ├── extractor.py       # Core extraction logic using PyMuPDF
│   └── BUILD              # Pants build configuration
└── parser/                 # Input parsing submodule
    ├── parser.py          # Page range parsing utilities
    └── BUILD              # Pants build configuration
```

## Module Organization

The general flow of the bounding box extractor is as follows:

1. **PDF Extraction**: The `extractor` submodule uses PyMuPDF to parse the PDF and extract raw page elements (text blocks, images, drawings) along with their bounding boxes.
2. **Element Classification**: The `classifier` submodule applies rule-based heuristics to label the raw elements (e.g., identifying page numbers, step numbers, parts lists).
3. **Output Generation**: The extracted and classified data is output as structured JSON. Optionally

### `main.py`

Command-line interface for extracting bounding boxes from PDFs. Handles argument parsing and orchestrates the extraction and classification process.

### Submodules

#### `extractor/` - PDF Extraction

Core PDF extraction functionality and data models.

**Data Models:**

This project uses a two-stage data model:

1. **Raw Page Elements** (`page_elements.py`): Defines the basic, low-level elements extracted directly from the PDF.
    - `Text`: A block of text.
    - `Image`: A raster image.
    - `Drawing`: A vector graphic.

**Extraction Logic:**

- `extractor.py`: Uses [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) to parse the PDF and extract the raw `Text`, `Image`, and `Drawing` elements.
- `hierarchy.py`: Builds a containment hierarchy from the flat list of raw elements.

#### `classifier/` - Element Classification

Handles the classification of raw page elements. It takes the raw elements from the `extractor` and applies a series of rule-based heuristics to assign labels (e.g., "page_number", "step_number"). This labeling is the first step in transforming the raw data into the structured LEGO elements. See the `classifier/README.md` for more details.

1. **Structured LEGO Elements** (`lego_page_elements.py`): Defines the high-level, domain-specific data model representing the logical structure of a LEGO instruction manual.
    - `Step`: A single instruction step.
    - `PartsList`: A list of parts required for a step.
    - `Part`: A single part in a parts list.
    - `PageNumber`, `StepNumber`, `PartCount`, etc.

#### `drawing/`

Handles visualization and rendering of extracted bounding boxes.

#### `parser/`

Input parsing utilities for things like page ranges.

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
pants test src/build_a_long/bounding_box_extractor/classifier:tests
pants test src/build_a_long/bounding_box_extractor/parser:tests
```

## Output

The extractor produces:

1. **JSON file**: Structured data containing all extracted and labeled raw elements.
2. **PNG images** (if `--output-dir` specified): Visual representation with bounding boxes drawn and labeled.
