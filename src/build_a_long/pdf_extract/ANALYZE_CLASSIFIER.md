# Classifier Analysis Tool

This tool analyzes the performance of the element classifier across all PDF files in the data directory.

## Usage

### Basic Analysis

Run on all PDFs in the data directory:

```bash
pants run src/build_a_long/pdf_extract:analyze_classifier
```

### Test with Limited PDFs

Analyze only the first N PDFs (useful for quick testing):

```bash
pants run src/build_a_long/pdf_extract:analyze_classifier -- --max-pdfs 10
```

### Detailed Report

Show detailed per-document analysis with pages that are missing classifications:

```bash
pants run src/build_a_long/pdf_extract:analyze_classifier -- --detailed
```

### Custom Data Directory

Specify a different data directory:

```bash
pants run src/build_a_long/pdf_extract:analyze_classifier -- --data-dir /path/to/pdfs
```

### All Options

```bash
pants run src/build_a_long/pdf_extract:analyze_classifier -- --help
```

## Output

The tool provides:

1. **Summary Report**:
   - Total PDFs and pages analyzed
   - Overall coverage percentage
   - Distribution by coverage bucket (100%, 75-99%, etc.)

2. **Detailed Report** (with `--detailed`):
   - Per-document coverage statistics
   - Specific pages missing classifications
   - Element counts for problematic pages

## Example Output

```text
================================================================================
CLASSIFIER ANALYSIS SUMMARY
================================================================================

Total PDFs analyzed: 116
Total pages analyzed: 5432
Pages with page number identified: 5301
Overall coverage: 97.6%

COVERAGE DISTRIBUTION

100% coverage (89 PDFs):
  10237/6055739.pdf: 68/68 pages
  ...

75-99% coverage (15 PDFs):
  10237/6055740.pdf: 71/72 pages
  ...
```

## Use Cases

