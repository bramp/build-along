# Classifier Test Fixtures

## Overview

This directory contains test fixtures for the PDF block classifier.

## Fixture Types

### Raw Input Fixtures (`*_raw.json`)

- **Source**: From various LEGO® instruction manuals
- **Format**: JSON serialization of `PageData` objects extracted from PDFs
- **Usage**: Input to the classifier for testing

Example files:

- `6509377_page_010_raw.json` - Page 10 from instruction manual 6509377
- `6509377_page_011_raw.json` - Page 11 from instruction manual 6509377
- etc.

### Golden Output Fixtures (`*_expected.json`)

- **Format**: JSON serialization of `ClassificationResult` objects
- **Usage**: Expected output for golden file testing
- **Generation**: Created/updated using the `generate-golden-files` script

## Test Files Using These Fixtures

### `classifier_rules_test.py` - Invariant Tests

Tests universal rules that must always hold:

- Every parts list contains at least one part image
- No two parts lists overlap
- Each part image is inside a parts list
- No labeled block is deleted

**Status**: Currently skipped due to known classifier issues.

**Usage**: `pants test src/build_a_long/pdf_extract/classifier:classifier_rules_test`

### `classifier_golden_test.py` - Golden File Tests

Tests that classification output matches expected golden files:

- Validates exact classification output
- Also runs all invariant checks
- Supports comparing against known-good golden files

**Status**: Currently skipped (same classifier issues as rules test).

**Usage**:

```bash
# Run tests (compare against golden files)
pants test src/build_a_long/pdf_extract/classifier:classifier_golden_test
```

## Workflow

### Adding New Fixtures

1. **Add raw input file**: Place a new `*_raw.json` file in this directory
2. **Generate golden file**: Run the generation script (see below)
3. **Verify output**: Review the generated `*_expected.json` file
4. **Commit both files**: Include both raw and expected files in version control

### Regenerating Golden Files

When you intentionally change classifier logic and want to update the expected outputs:

```bash
# Generate/update all golden files
pants run src/build_a_long/pdf_extract/classifier/tools:generate-golden-files
```

This script runs without sandboxing, allowing it to write files directly to the `fixtures/` directory.

### Enabling the Tests

Once the classifier issues are fixed:

1. Remove the `@pytest.mark.skip` decorator from both test files
2. Regenerate golden files if needed
3. Run tests to verify they pass

## Fixture Details

- `6509377_*` files are from set 75375 LEGO® Star Wars™ Millennium Falcon™
