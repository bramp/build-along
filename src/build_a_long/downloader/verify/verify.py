"""Verify the integrity of downloaded data."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm  # Keep tqdm.auto for tqdm.write
from tqdm.contrib.concurrent import process_map

from build_a_long.schemas import InstructionMetadata


class VerificationError(BaseModel):
    type: str
    message: str


def _verify_single_metadata(metadata_path: Path) -> list[VerificationError]:
    """
    Verifies the integrity of a single LEGO instruction file against its metadata.

    Args:
        metadata_path: The path to the metadata.json file.

    Returns:
        A list of verification errors. An empty list means no errors were found.
    """
    errors = []
    declared_pdf_paths = set()  # To store paths of PDFs mentioned in metadata
    set_dir = metadata_path.parent
    try:
        with open(metadata_path) as f:
            data = json.load(f)
        metadata = InstructionMetadata.model_validate(data)
    except (ValidationError, json.JSONDecodeError) as e:
        errors.append(
            VerificationError(
                type="invalid_metadata",
                message=(
                    f"Error: Could not validate or parse metadata in {metadata_path}: {e}"
                ),
            )
        )
        return errors

    for pdf_entry in metadata.pdfs:
        # Check for missing filename
        if not pdf_entry.filename:
            errors.append(
                VerificationError(
                    type="missing_filename",
                    message=(
                        f"Error: Missing filename in metadata for set {metadata.set}, "
                        f"URL: {pdf_entry.url}"
                    ),
                )
            )
            continue  # Can't proceed without a filename

        pdf_path = set_dir / pdf_entry.filename
        declared_pdf_paths.add(pdf_path)  # Add to our declared set

        if not pdf_path.exists():
            errors.append(
                VerificationError(
                    type="missing_file",
                    message=f"Error: Missing file {pdf_path} for set {metadata.set}",
                )
            )
            continue

        # Verify filesize
        if pdf_entry.filesize is not None:
            actual_size = pdf_path.stat().st_size
            if actual_size != pdf_entry.filesize:
                errors.append(
                    VerificationError(
                        type="filesize_mismatch",
                        message=(
                            f"Error: Filesize mismatch for {pdf_path} "
                            f"(expected: {pdf_entry.filesize}, actual: {actual_size})"
                        ),
                    )
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
                    VerificationError(
                        type="hash_mismatch",
                        message=(
                            f"Error: Hash mismatch for {pdf_path} "
                            f"(expected: {pdf_entry.filehash}, actual: {actual_hash})"
                        ),
                    )
                )
    # Check for orphaned PDFs in this set's directory
    all_pdfs_in_set_dir = set(set_dir.glob("*.pdf"))
    orphaned_pdfs = all_pdfs_in_set_dir - declared_pdf_paths

    for pdf in orphaned_pdfs:
        errors.append(
            VerificationError(
                type="orphaned_file",
                message=f"Error: Orphaned PDF file found in {set_dir}: {pdf}",
            )
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
    metadata_files = list(data_dir.rglob("metadata.json"))

    if not metadata_files:
        print("No metadata files found to verify.")
        return 0

    error_counts: Counter[str] = Counter()
    error_found = False

    # Use process_map for parallel execution with a progress bar
    # The _verify_single_metadata function will be called for each metadata file.
    # Errors will be collected and reported.
    for error_list in process_map(
        _verify_single_metadata,
        metadata_files,
        desc="Verifying sets",
        unit="set",
        max_workers=4,
        chunksize=1,
    ):
        if error_list:
            error_found = True
            for error in error_list:
                tqdm.write(
                    error.message
                )  # Use tqdm.write for consistent error reporting
                error_counts[error.type] += 1

    if not error_found:
        print("Verification complete. No issues found.")
    else:
        print("\nVerification Summary:")
        print("-" * 30)
        for error_type, count in error_counts.most_common():
            # Format the label to be more human-readable
            label = error_type.replace("_", " ").title()
            print(f"{label}: {count}")
        print("-" * 30)

    return 1 if error_found else 0
