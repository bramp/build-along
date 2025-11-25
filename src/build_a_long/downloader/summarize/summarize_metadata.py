"""Summarizes all metadata files into a yearly index."""

import argparse
import collections
import json
from pathlib import Path

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from build_a_long.downloader.metadata import read_metadata
from build_a_long.schemas import InstructionMetadata


def _load_single_metadata(metadata_file: Path) -> InstructionMetadata | None:
    """Load and parse a single metadata file.

    Args:
        metadata_file: Path to the metadata.json file.

    Returns:
        The parsed InstructionMetadata object, or None if there was an error.
    """
    try:
        return read_metadata(metadata_file)
    except (OSError, ValueError) as e:
        tqdm.write(f"Warning: Could not read {metadata_file}: {e}; ignoring")
        return None


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

    # Use process_map for parallel loading of metadata files
    metadata_list = process_map(
        _load_single_metadata,
        metadata_files,
        desc="Loading metadata",
        unit="file",
        max_workers=4,
        chunksize=10,
    )

    # Group metadata by year
    for metadata in metadata_list:
        if metadata is None:
            continue
        year = metadata.year
        if year:
            yearly_metadata[year].append(metadata)

    output_dir.mkdir(exist_ok=True)

    all_years_summary = []
    sorted_years = sorted(yearly_metadata.keys())

    for year in sorted_years:
        metadata_list = yearly_metadata[year]
        metadata_list.sort(key=lambda x: x.set)
        filename = f"index-{year}.json"
        output_file = output_dir / filename
        with open(output_file, "w") as f:
            # Convert to dict for JSON serialization with mode='json' to serialize URLs
            json.dump(
                [m.model_dump(mode="json", exclude_unset=True) for m in metadata_list],
                f,
                indent=2,
            )

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
