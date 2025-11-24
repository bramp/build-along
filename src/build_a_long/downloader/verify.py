"""Verify the integrity of downloaded data."""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError
from tqdm.auto import tqdm  # Keep tqdm.auto for tqdm.write
from tqdm.contrib.concurrent import process_map

from build_a_long.schemas import InstructionMetadata


def _verify_single_metadata(metadata_path: Path) -> list[str]:
    """
    Verifies the integrity of a single LEGO instruction file against its metadata.

    Args:
        metadata_path: The path to the metadata.json file.

    Returns:
        A list of error messages. An empty list means no errors were found.
    """
    errors = []
    try:
        with open(metadata_path) as f:
            data = json.load(f)
        metadata = InstructionMetadata.model_validate(data)
    except (ValidationError, json.JSONDecodeError) as e:
        errors.append(
            f"Error: Could not validate or parse metadata in {metadata_path}: {e}"
        )
        return errors

    set_dir = metadata_path.parent
    for pdf_entry in metadata.pdfs:
        pdf_path = set_dir / pdf_entry.filename

        if not pdf_path.exists():
            errors.append(f"Error: Missing file {pdf_path} for set {metadata.set}")
            continue

        # Verify filesize
        if pdf_entry.filesize is not None:
            actual_size = pdf_path.stat().st_size
            if actual_size != pdf_entry.filesize:
                errors.append(
                    f"Error: Filesize mismatch for {pdf_path} "
                    f"(expected: {pdf_entry.filesize}, actual: {actual_size})"
                )

        # Verify hash
        if pdf_entry.filehash is not None:
            hasher = hashlib.sha256()
            with open(pdf_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            actual_hash = hasher.hexdigest()
            if actual_hash != pdf_entry.filehash:
                errors.append(
                    f"Error: Hash mismatch for {pdf_path} "
                    f"(expected: {pdf_entry.filehash}, actual: {actual_hash})"
                )
    return errors


def verify_data_integrity(data_dir: Path) -> int:
    """
    Verifies the integrity of downloaded LEGO instruction files against their metadata.

    Args:
        data_dir: The root directory containing the downloaded data.

    Returns:
        0 if all files are consistent, 1 if any inconsistencies are found.
    """
    error_found = False
    metadata_files = list(data_dir.rglob("metadata.json"))

    if not metadata_files:
        print("No metadata files found to verify.")
        return 0

    # Use process_map for parallel execution with a progress bar
    # The _verify_single_metadata function will be called for each metadata file.
    # Errors will be collected and reported.
    for errors in process_map(
        _verify_single_metadata,
        metadata_files,
        desc="Verifying sets",
        unit="set",
        max_workers=4,
        chunksize=1,
    ):
        if errors:
            error_found = True
            for error in errors:
                tqdm.write(error)  # Use tqdm.write for consistent error reporting

    if not error_found:
        print("Verification complete. No issues found.")

    return 1 if error_found else 0
