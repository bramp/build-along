import csv
import json
import os
import sys
from collections import defaultdict

# Determine repo root relative to this script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def scan_sets():
    # year -> theme -> list of {set, filename, path}
    sets_by_year = defaultdict(lambda: defaultdict(list))

    set_dirs = [
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    # print(f"Found {len(set_dirs)} set directories.", file=sys.stderr)

    for set_id in set_dirs:
        metadata_path = os.path.join(DATA_DIR, set_id, "metadata.json")
        if not os.path.exists(metadata_path):
            continue

        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)

            year = data.get("year")
            theme = data.get("theme")
            pdfs = data.get("pdfs", [])

            if not year or not theme or not pdfs:
                continue

            # Prefer the first PDF
            pdf_filename = pdfs[0].get("filename")
            if not pdf_filename:
                continue

            pdf_path = os.path.join(DATA_DIR, set_id, pdf_filename)

            if not os.path.exists(pdf_path):
                continue

            sets_by_year[year][theme].append(
                {
                    "set": set_id,
                    "filename": pdf_filename,
                    "path": pdf_path,
                    "year": year,
                    "theme": theme,
                }
            )

        except Exception as e:
            print(f"Error reading {metadata_path}: {e}", file=sys.stderr)

    # Strategy:
    # For each year, pick up to 4 themes.
    # To ensure "spread", prioritize themes that have been used less often globally so far.

    theme_usage_counts = defaultdict(int)
    csv_rows = []

    for year in sorted(sets_by_year.keys()):
        themes_in_year = list(sets_by_year[year].keys())

        # Sort themes by how many times we've already picked them (ascending)
        # This encourages picking new/rare themes.
        # Secondary sort by theme name for stability.
        themes_in_year.sort(key=lambda t: (theme_usage_counts[t], t))

        # Pick top 4
        selected_themes = themes_in_year[:4]

        for theme in selected_themes:
            # Pick the first set for this theme in this year
            # (We could randomise this too, but first is deterministic)
            set_info = sets_by_year[year][theme][0]

            csv_rows.append(set_info)
            theme_usage_counts[theme] += 1

    # Write CSV to stdout
    fieldnames = ["year", "theme", "set", "filename", "path"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)


if __name__ == "__main__":
    scan_sets()
