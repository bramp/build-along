#!/bin/bash
set -e

INPUT_CSV="scripts/example_pdfs.csv"

if [ ! -f "$INPUT_CSV" ]; then
	echo "Error: $INPUT_CSV not found in $REPO_ROOT!"
	exit 1
fi

TOOL_PATH=${CHROOT:-dist}
TOOL="$TOOL_PATH/src.build_a_long.pdf_extract/main.pex"

if [ ! -f "$TOOL" ]; then
	echo "Error: Built tool not found at $TOOL"
	exit 1
fi

# Ensure output directory exists
mkdir -p debug/all

# Read CSV, skip header
# CSV format: year,theme,set,filename,path
tail -n +2 "$INPUT_CSV" | while IFS=, read -r year theme set_id filename path; do
	# Remove any carriage returns just in case
	path=$(echo "$path" | tr -d '\r')

	if [ -z "$path" ]; then
		continue
	fi

	echo "----------------------------------------------------------------"
	echo "Processing Year: $year, Theme: $theme, Set: $set_id"

	"$TOOL" --output-dir debug/all "$path"

	echo ""
done

echo "All done!"
