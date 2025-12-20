# Scripts

This directory contains utility scripts for the project.

## `all-sets.txt`

A text file containing a list of LEGO set IDs (one per line). This is used as a reference or input for bulk downloading operations.

## `scan_sets.py`

A Python script that scans the `data/` directory for downloaded LEGO set metadata and PDFs. It generates a CSV output (printed to stdout) listing a selection of sets (up to 4 themes per year) to be used for testing or analysis.

**Usage:**

```bash
python3 scripts/scan_sets.py > scripts/example_pdfs.csv
```

## `process_all_examples.sh`

A Bash script that:

1. Builds the `pdf_extract` tool using `pants package`.
2. Reads `example_pdfs.csv` (located in the repository root).
3. Runs the extracted tool against each PDF listed in the CSV.
4. Outputs debug information to `debug/all/`.

**Usage:**

```bash
pants run scripts:process-all-examples
```
