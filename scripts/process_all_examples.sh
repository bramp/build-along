#!/bin/bash
set -e

INPUT_CSV="scripts/example_pdfs.csv"

if [ ! -f "$INPUT_CSV" ]; then
	echo "Error: $INPUT_CSV not found in $REPO_ROOT!"
	exit 1
fi

# Build the tool first
echo "Building pdf_extract tool..."
PANTS_CONCURRENT=True pants package src/build_a_long/pdf_extract:main

TOOL_PATH=${CHROOT:-dist}
TOOL="$TOOL_PATH/src.build_a_long.pdf_extract/main.pex"

if [ ! -f "$TOOL" ]; then
	echo "Error: Built tool not found at $TOOL"
	exit 1
fi

# Ensure output directory exists
mkdir -p debug/all

# Initialize results log with timestamp in filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_LOG="debug/all/processing_results_${TIMESTAMP}.txt"
ERROR_LOG="debug/all/processing_errors_${TIMESTAMP}.txt"

{
	echo "Processing started at $(date)"
	echo "Format: [STATUS] Year | Theme | Set | Path"
	echo ""
} | tee "$RESULTS_LOG"

echo "Error details started at $(date)" >"$ERROR_LOG"
echo "" >>"$ERROR_LOG"

# Counters
SUCCESS_COUNT=0
FAILURE_COUNT=0
TOTAL_COUNT=0

# Read CSV, skip header
# CSV format: year,theme,set,filename,path
tail -n +2 "$INPUT_CSV" | while IFS=, read -r year theme set_id _ path; do
	# Remove any carriage returns just in case
	path=$(echo "$path" | tr -d '\r')

	if [ -z "$path" ]; then
		continue
	fi

	TOTAL_COUNT=$((TOTAL_COUNT + 1))

	{
		echo "----------------------------------------------------------------"
		echo "Processing [$TOTAL_COUNT]: Year: $year, Theme: $theme, Set: $set_id"
	} | tee -a "$RESULTS_LOG"

	# Run the tool and capture output and exit code
	# Temporarily disable errexit for this command
	ERROR_OUTPUT=$(mktemp)
	set +e
	"$TOOL" --output-dir debug/all "$path" 2>"$ERROR_OUTPUT"
	EXIT_CODE=$?
	set -e

	if [ $EXIT_CODE -eq 0 ]; then
		echo "[SUCCESS] $year | $theme | $set_id | $path" | tee -a "$RESULTS_LOG"
		SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
		echo "✓ SUCCESS" | tee -a "$RESULTS_LOG"
	else
		echo "[FAILURE] $year | $theme | $set_id | $path | Exit code: $EXIT_CODE" | tee -a "$RESULTS_LOG"
		echo "----------------------------------------------------------------" >>"$ERROR_LOG"
		echo "[FAILURE] $year | $theme | $set_id | $path" >>"$ERROR_LOG"
		echo "Exit code: $EXIT_CODE" >>"$ERROR_LOG"
		cat "$ERROR_OUTPUT" >>"$ERROR_LOG"
		echo "" >>"$ERROR_LOG"
		FAILURE_COUNT=$((FAILURE_COUNT + 1))
		{
			echo "✗ FAILED (see $ERROR_LOG for details)"
			# Show first few lines of error
			head -3 "$ERROR_OUTPUT"
		} | tee -a "$RESULTS_LOG"
	fi
	rm -f "$ERROR_OUTPUT"

	echo "" | tee -a "$RESULTS_LOG"
done

# Summary
{
	echo ""
	echo "================================================================"
	echo "Processing completed at $(date)"
	echo "Total: $TOTAL_COUNT | Success: $SUCCESS_COUNT | Failures: $FAILURE_COUNT"
	echo ""
	echo "Results logged to: $RESULTS_LOG"
	echo "Error details in: $ERROR_LOG"
} | tee -a "$RESULTS_LOG"
