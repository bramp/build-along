# Classifier Test Fixtures

## Overview

This directory contains test fixtures for the PDF block classifier.

## Fixture Types

### Raw Input Fixtures

#### Individual Page Fixtures (`*_raw.json`)

- **Source**: From various LEGO® instruction manuals
- **Format**: JSON serialization of `PageData` objects extracted from PDFs
- **Usage**: Input to the classifier for testing individual pages

Example files:

- `6509377_page_010_raw.json` - Page 10 from instruction manual 6509377
- `6509377_page_011_raw.json` - Page 11 from instruction manual 6509377
- etc.

#### Full Document Fixtures (`*_raw.json.bz2`)

- **Source**: Complete LEGO® instruction manuals
- **Format**: Bz2-compressed JSON serialization of `ExtractionResult` with multiple pages
- **Usage**: Input for tests that analyze entire documents (e.g., font size hints)

Example files:

- `6055741_raw.json.bz2` - Complete instruction manual 6055741
- `6509377_raw.json.bz2` - Complete instruction manual 6509377
- etc.
-

### Golden Output Fixtures

#### Classifier Output (`*_expected.json`)

- **Format**: JSON serialization of `ClassificationResult` objects
- **Usage**: Expected output for golden file testing
- **Generation**: Created/updated using the `generate-golden-files` script

#### Font Size Hints Output (`*_font_hints_expected.json`)

- **Format**: JSON serialization of `FontSizeHints` objects
- **Usage**: Expected output for font size hint extraction testing
- **Generation**: Created/updated using the `generate-golden-hints` script

#### Page Hints Output (`*_page_hints_expected.json`)

- **Format**: JSON serialization of `PageHintCollection` objects
- **Usage**: Expected output for page type classification hints
- **Generation**: Created/updated using the `generate-golden-hints` script

## Test Files Using These Fixtures

### `classifier_rules_test.py` - Invariant Tests

Tests universal rules that must always hold:

- Every parts list contains at least one part image
- No two parts lists overlap
- Each part image is inside a parts list
- No labeled block is deleted

**Usage**: `pants test src/build_a_long/pdf_extract/classifier/classifier_rules_test.py`

### `classifier_golden_test.py` - Golden File Tests

Tests that classification output matches expected golden files:

- Validates exact classification output
- Also runs all invariant checks
- Supports comparing against known-good golden files

**Usage**:

```bash
# Run tests (compare against golden files)
pants test src/build_a_long/pdf_extract/classifier/classifier_golden_test.py
```

### `font_size_hints_test.py` - Font Size Hints Tests

Tests the `FontSizeHints.from_pages` method:

- Unit tests for various scenarios (all sizes, two sizes, one size, etc.)
- Golden file tests that validate font size extraction from full documents
- Uses compressed `.json.bz2` fixtures for testing complete instruction manuals

**Usage**:

```bash
# Run all font size hints tests
pants test src/build_a_long/pdf_extract/classifier/font_size_hints_test.py

# Run only golden file tests
pants test src/build_a_long/pdf_extract/classifier/font_size_hints_test.py -- -k "test_from_pages_matches_golden"
```

## Workflow

### Adding New Fixtures

#### For Individual Page Fixtures

1. **Generate raw input file**: Extract pages from PDF using the main CLI:

   ```bash
   # Extract specific pages with extended drawing metadata
   pants run src/build_a_long/pdf_extract:main -- \
     path/to/instruction_manual.pdf \
     --pages 10-17,180 \
     --output-dir src/build_a_long/pdf_extract/fixtures \
     --debug-json \
     --debug-extra-json
   ```

   This creates `*_page_NNN_raw.json` files directly in the fixtures directory.

   Optionally add `--compress-json` to create compressed `.json.bz2` files.

2. **Generate golden file**: Run the generation script (see below)
3. **Verify output**: Review the generated `*_expected.json` file
4. **Commit both files**: Include both raw and expected files in version control

**Important**: Use `--debug-extra-json` to include full metadata (drawing paths, colors, dimensions, etc.) in raw fixtures. This ensures Drawing objects have all fields including `original_bbox` for proper clipping detection.

#### Regenerating All Current Fixtures

To regenerate all existing fixture files (raw input and golden output files):

```bash
./src/build_a_long/pdf_extract/classifier/tools/regenerate_fixtures.sh
```

This script regenerates:

- Full document fixtures (`*_raw.json.bz2`) from source PDFs
- Individual page fixtures (`*_raw.json`) from source PDFs
- Hints golden files (`*_font_hints_expected.json`, `*_page_hints_expected.json`)
- Classifier golden files (`*_expected.json`)

### Regenerating Golden Files

When you intentionally change classifier logic and want to update the expected outputs:

```bash
# Generate/update classifier golden files
pants run src/build_a_long/pdf_extract/classifier/tools/generate_golden_files.py

# Generate/update hints golden files (font hints and page hints)
pants run src/build_a_long/pdf_extract/classifier/tools/generate_golden_hints.py
```

**Important**: Use the `.py` file path (not the `:target` name) to run without sandboxing. This allows the scripts to read/write files directly in the repository.

## Fixture Details

### LEGO® Sets Represented

- **10237** The Lord of the Rings™ The Tower of Orthanc
  - 2359 pieces, released in 2013
  - **`6055739_*`** Manual 1 of 3
  - **`6055740_*`** Manual 2 of 3
  - **`6055741_*`** Manual 3 of 3

- **40573** Ideas Christmas Tree
  - 784 pieces, released in 2022
  - **`6433200_*`** Manual 1 of 1 (pages 4, 5, 7, 31)

- **75375** Star Wars™ Millennium Falcon™
  - 921 pieces, released in 2024
  - **`6509377_*`** Manual 1 of 1

- **75406** Star Wars™ Kylo Ren's Command Shuttle
  - 386 pieces, released in 2025
  - **`6580053_*`** Manual 1 of 1

### File Naming Convention

- `{element_id}_page_{NNN}_raw.json` - Single page raw extraction
- `{element_id}_raw.json.bz2` - Full document raw extraction (compressed)
- `{element_id}_page_{NNN}_expected.json` - Single page classifier golden output
- `{element_id}_font_hints_expected.json` - Font size hints golden output
- `{element_id}_page_hints_expected.json` - Page hints golden output

Note: An element ID identifies a LEGO element (manual, piece, sticker sheet, etc.).
