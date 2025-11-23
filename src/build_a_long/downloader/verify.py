"""Verify the integrity of downloaded data."""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError
from tqdm.auto import tqdm

from build_a_long.schemas import InstructionMetadata


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

    for metadata_path in tqdm(metadata_files, desc="Verifying sets", unit="set"):
        try:
            with open(metadata_path) as f:
                data = json.load(f)
            metadata = InstructionMetadata.model_validate(data)
        except (ValidationError, json.JSONDecodeError) as e:
            tqdm.write(
                f"Error: Could not validate or parse metadata in {metadata_path}: {e}"
            )
            error_found = True
            continue

        set_dir = metadata_path.parent
        for pdf_entry in metadata.pdfs:
            pdf_path = set_dir / pdf_entry.filename

            if not pdf_path.exists():
                tqdm.write(f"Error: Missing file {pdf_path} for set {metadata.set}")
                error_found = True
                continue

            # Verify filesize
            if pdf_entry.filesize is not None:
                actual_size = pdf_path.stat().st_size
                if actual_size != pdf_entry.filesize:
                    tqdm.write(
                        f"Error: Filesize mismatch for {pdf_path} "
                        f"(expected: {pdf_entry.filesize}, actual: {actual_size})"
                    )
                    error_found = True

            # Verify hash
            if pdf_entry.filehash is not None:
                hasher = hashlib.sha256()
                with open(pdf_path, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
                actual_hash = hasher.hexdigest()
                if actual_hash != pdf_entry.filehash:
                    tqdm.write(
                        f"Error: Hash mismatch for {pdf_path} "
                        f"(expected: {pdf_entry.filehash}, actual: {actual_hash})"
                    )
                    error_found = True

    if not error_found:
        print("Verification complete. No issues found.")

    return 1 if error_found else 0
