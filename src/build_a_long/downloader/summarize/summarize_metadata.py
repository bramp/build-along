"""Summarizes all metadata files into a yearly index."""

import argparse
import collections
import json
from pathlib import Path


def summarize_metadata(data_dir: Path, output_dir: Path) -> int:
    """Summarize all metadata.json files into a yearly index.

    Args:
        data_dir: The directory containing the data folders.
        output_dir: The directory to write the indices to.

    Returns:
        0 on success, 1 on error.
    """
    yearly_metadata = collections.defaultdict(list)
    metadata_files = list(data_dir.glob("*/metadata.json"))

    if not metadata_files:
        print(f"No metadata.json files found in {data_dir}")
        return 1

    for metadata_file in metadata_files:
        with open(metadata_file) as f:
            try:
                metadata = json.load(f)
                year = metadata.get("year")
                if year:
                    yearly_metadata[year].append(metadata)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {metadata_file}")
                continue

    output_dir.mkdir(exist_ok=True)

    all_years_summary = []
    sorted_years = sorted(yearly_metadata.keys())

    for year in sorted_years:
        metadata_list = yearly_metadata[year]
        metadata_list.sort(key=lambda x: x.get("set_id", ""))
        filename = f"index-{year}.json"
        output_file = output_dir / filename
        with open(output_file, "w") as f:
            json.dump(metadata_list, f, indent=2)

        all_years_summary.append(
            {
                "year": year,
                "count": len(metadata_list),
                "filesize": output_file.stat().st_size,
                "filename": filename,
            }
        )

    index_file = output_dir / "index.json"
    with open(index_file, "w") as f:
        json.dump(all_years_summary, f, indent=2)

    print(f"Indexed {len(metadata_files)} files into {len(sorted_years)} year(s).")
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the summarizer script."""
    parser = argparse.ArgumentParser(
        description="Summarize LEGO metadata into a yearly index."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the downloaded LEGO set data.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/indices",
        help="Directory to store the generated index files.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for the metadata summarizer."""
    args = _parse_args()
    return summarize_metadata(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
