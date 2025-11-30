#!/usr/bin/env bash
# Regenerate all raw fixture files from source PDFs.
#
# This script extracts pages from the source PDFs to create the raw fixture files
# used by tests. Run this when you need to update fixtures after changes to the
# extraction logic.
#
# Usage:
#   ./src/build_a_long/pdf_extract/classifier/tools/regenerate_fixtures.sh
#

set -euo pipefail

echo "Regenerating all raw fixture files..."
echo ""

FIXTURES_DIR="src/build_a_long/pdf_extract/fixtures"

# Individual page fixtures from 6509377 (pages 10-17, 180)
echo "=== 6509377: Star Wars Millennium Falcon (75375) - Individual page fixtures (pages 10-17, 180) ==="
pants run src/build_a_long/pdf_extract:main -- \
	data/75375/6509377.pdf \
	--pages 10-17,180 \
	--output-dir "$FIXTURES_DIR" \
	--raw-json \
	--no-json \
	--no-summary

# Individual page fixtures from 6433200 (pages 4, 5, 7, 31)
echo ""
echo "=== 6433200: Ideas Christmas Tree (40573) - Individual page fixtures (pages 4, 5, 7, 31) ==="
pants run src/build_a_long/pdf_extract:main -- \
	data/40573/6433200.pdf \
	--pages 4,5,7,31 \
	--output-dir "$FIXTURES_DIR" \
	--raw-json \
	--no-json \
	--no-summary

# Full document fixtures (compressed)
echo ""
echo "=== 6055739: Tower of Orthanc (10237) - Full document fixture (manual 1/3) ==="
pants run src/build_a_long/pdf_extract:main -- \
	data/10237/6055739.pdf \
	--output-dir "$FIXTURES_DIR" \
	--raw-json \
	--no-json \
	--no-summary \
	--compress-json

echo ""
echo "=== 6055740: Tower of Orthanc (10237) - Full document fixture (manual 2/3) ==="
pants run src/build_a_long/pdf_extract:main -- \
	data/10237/6055740.pdf \
	--output-dir "$FIXTURES_DIR" \
	--raw-json \
	--no-json \
	--no-summary \
	--compress-json

echo ""
echo "=== 6055741: Tower of Orthanc (10237) - Full document fixture (manual 3/3) ==="
pants run src/build_a_long/pdf_extract:main -- \
	data/10237/6055741.pdf \
	--output-dir "$FIXTURES_DIR" \
	--raw-json \
	--no-json \
	--no-summary \
	--compress-json

echo ""
echo "=== 6509377: Star Wars Millennium Falcon (75375) - Full document fixture ==="
pants run src/build_a_long/pdf_extract:main -- \
	data/75375/6509377.pdf \
	--output-dir "$FIXTURES_DIR" \
	--raw-json \
	--no-json \
	--no-summary \
	--compress-json

echo ""
echo "=== 6580053: Star Wars Kylo Ren's Command Shuttle (75406) - Full document fixture ==="
pants run src/build_a_long/pdf_extract:main -- \
	data/75406/6580053.pdf \
	--output-dir "$FIXTURES_DIR" \
	--raw-json \
	--no-json \
	--no-summary \
	--compress-json

echo ""
echo "✓ All raw fixtures regenerated successfully!"
echo ""
echo "Next steps:"
echo "  1. Run: pants run src/build_a_long/pdf_extract/classifier/tools/generate_golden_hints.py"
echo "  2. Run: pants run src/build_a_long/pdf_extract/classifier/tools/generate_golden_files.py"
echo "  3. Run: pants test ::"
