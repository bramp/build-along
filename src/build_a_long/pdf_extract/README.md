# PDF Extract Pipeline

This module provides a complete pipeline for extracting, classifying, and structuring LEGO instruction PDF pages. It transforms raw PDF files into structured hierarchies of LEGO-specific components.

## Architecture Overview

The processing pipeline has three main stages:

```text
┌───────────┐
│    PDF    │
└─────┬─────┘
      │
      ▼
┌─────────────────────┐
│    EXTRACTOR        │  Extracts raw blocks
│  (pymupdf-based)    │  - Text, Image, Drawing
└──────────┬──────────┘  - BBox for each
           │
           ▼
┌─────────────────────┐
│     PageData        │  Flat list of blocks
│  - Text, Image,     │  with bounding boxes
│    Drawing          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   CLASSIFIER        │  Scores blocks,
│  (rule-based)       │  constructs LegoElements,
│                     │  selects best candidates
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ClassificationResult│  Pre-constructed elements
│  - labeled_blocks   │  + Labels
│  - constructed_     │  + All candidates
│    elements         │  + Decision trail
│  - candidates       │
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

**Key Difference**: The classifier now **constructs** `LegoPageElement` objects during classification (single parse). This guarantees consistency and preserves all decision context.

## Quick Start

```python
import pymupdf
from build_a_long.pdf_extract.classifier import classify_pages
from build_a_long.pdf_extract.extractor import extract_page_data

# 1. Extract blocks from PDF
with pymupdf.open("instructions.pdf") as doc:
    pages = extract_page_data(doc, None)

# 2. Classify blocks
results = classify_pages(pages)

# 3. Build structured hierarchy
for page_data, result in zip(pages, results):
    page = result.page
    
    # Access structured LEGO elements
    if page.page_number:
        print(f"Page {page.page_number.value}")
    
    for step in page.steps:
        print(f"  Step {step.step_number.value}")
        for part in step.parts_list.parts:
            print(f"    {part.count.count}x")
```

## Directory Structure

```text
pdf_extract/
├── main.py                 # CLI entry point for the extractor
├── analyze_classifier.py   # Classifier analysis tool
├── ANALYZE_CLASSIFIER.md   # Classifier analysis documentation
├── classifier/             # Element classification submodule
│   ├── classifier.py      # Main classification logic
│   ├── page_number_classifier.py # Individual classifiers
│   ├── step_number_classifier.py
│   ├── part_count_classifier.py
│   ├── parts_list_classifier.py
│   ├── parts_image_classifier.py
│   └── README.md          # Classifier implementation details
├── drawing/                # Drawing and visualization submodule
│   └── drawing.py         # Functions to draw bounding boxes on PDF pages
├── extractor/              # PDF extraction submodule
│   ├── bbox.py            # BBox dataclass for bounding box representation
│   ├── page_elements.py   # Raw page element type definitions (Text, Image, etc.)
│   ├── lego_page_elements.py # Structured LEGO element type definitions (Step, Part, etc.)
│   ├── hierarchy.py       # Containment hierarchy building logic
│   └── extractor.py       # Core extraction logic using PyMuPDF
└── parser/                 # Input parsing submodule
    └── parser.py          # Page range parsing utilities
```

## Data Models

This project uses a two-stage data model to represent the transformation from raw PDF data to structured LEGO instructions:

### 1. Raw Page Elements (`extractor/page_elements.py`)

Defines the basic, low-level elements extracted directly from the PDF:

- `Text`: A block of text with font, size, and position
- `Image`: A raster image with dimensions
- `Drawing`: A vector graphic region

These elements are flat and unstructured, representing the PDF's raw content.

### 2. Structured LEGO Elements (`extractor/lego_page_elements.py`)

Defines the high-level, domain-specific data model representing the logical structure of a LEGO instruction manual:

- `Page`: Top-level container for a single page
- `PageNumber`: The page number
- `Step`: A single instruction step
- `StepNumber`: The step number
- `PartsList`: A collection of parts required for a step
- `Part`: A single part with count, image, and optional name/number
- `PartCount`: The quantity of a part (e.g., "2x")
- `Diagram`: An illustration showing how to assemble parts

[LEGO Page Elements Layout Diagram](lego_page_layout.png): Visual diagram showing the hierarchical structure and spatial arrangement of LEGO page elements. See [lego_page_elements.py](src/build_a_long/pdf_extract/extractor/lego_page_elements.py) for detailed type definitions with positional context. To regenerate: `pants run src/build_a_long/pdf_extract/tools/lego_page_layout.py`

### Pipeline Flow

1. **PDF Extraction**: The `extractor` submodule uses PyMuPDF to parse the PDF and extract raw page blocks (text blocks, images, drawings) along with their bounding boxes.
2. **Element Classification**: The `classifier` submodule applies rule-based heuristics to score elements, **construct** `LegoPageElement` objects for valid candidates, and select the best match. All candidates and decision context are preserved.
3. **Output Generation**: The extracted and classified data is output as structured JSON. Optionally, annotated PNG images can be generated showing the bounding boxes.

## Submodules

### `main.py`

Command-line interface for extracting bounding boxes from PDFs. Handles argument parsing and orchestrates the extraction and classification process.

### `extractor/` - PDF Extraction

Core PDF extraction functionality using [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/).

**Key files:**

- `extractor.py`: Main extraction logic
- `page_elements.py`: Raw element type definitions
- `lego_page_elements.py`: Structured element type definitions
- `bbox.py`: Bounding box utilities
- `hierarchy.py`: Containment hierarchy building

### `classifier/` - Element Classification

Applies rule-based heuristics to identify and label LEGO-specific elements. The classifier runs multiple passes in a specific order, with later passes depending on earlier classifications.

**Classification Pipeline:**

1. **PageNumberClassifier** - Identifies page numbers
2. **PartCountClassifier** - Detects part-count text (e.g., "2x", "3X")
3. **StepNumberClassifier** - Identifies step numbers
4. **PartsListClassifier** - Identifies the drawing region for the parts list
5. **PartsImageClassifier** - Associates part counts with their images

See [`classifier/README.md`](classifier/README.md) for detailed classifier implementation information.

### `drawing/` - Visualization

Handles rendering of extracted bounding boxes onto PDF pages for debugging and visualization.

### `parser/` - Input Parsing

Input parsing utilities for things like page ranges.

## Usage

```bash
# Extract from entire PDF
pants run src/build_a_long/pdf_extract:main path/to/file.pdf

# Extract specific pages
pants run src/build_a_long/pdf_extract:main path/to/file.pdf --pages 5-10

# Specify output directory
pants run src/build_a_long/pdf_extract:main path/to/file.pdf --output-dir ./output

# Filter by element types
pants run src/build_a_long/pdf_extract:main path/to/file.pdf --include-types text,image
```

## Testing

Run all tests:

```bash
pants test src/build_a_long/pdf_extract::
```

Run specific submodule tests:

```bash
pants test src/build_a_long/pdf_extract/extractor:tests
pants test src/build_a_long/pdf_extract/classifier:tests
pants test src/build_a_long/pdf_extract/parser:tests
```

## Output

The extractor produces:

1. **JSON file**: Structured data containing all extracted and labeled raw elements.
2. **PNG images** (if `--output-dir` specified): Visual representation with bounding boxes drawn and labeled.

## Classifier Analysis Tool

The `analyze_classifier.py` tool helps evaluate classifier performance across multiple PDF files. See [ANALYZE_CLASSIFIER.md](ANALYZE_CLASSIFIER.md) for detailed usage information.

**Quick usage:**

```bash
# Analyze all PDFs in the data directory
pants run src/build_a_long/pdf_extract:analyze_classifier

# Analyze with detailed per-document report
pants run src/build_a_long/pdf_extract:analyze_classifier -- --detailed

# Test with limited PDFs
pants run src/build_a_long/pdf_extract:analyze_classifier -- --max-pdfs 10
```

The tool provides coverage statistics, showing how many pages have page numbers successfully identified and highlighting problematic documents.
