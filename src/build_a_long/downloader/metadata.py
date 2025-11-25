"""Utilities for reading and writing LEGO instruction metadata files."""

from pathlib import Path

from build_a_long.schemas import InstructionMetadata

__all__ = [
    "read_metadata",
    "write_metadata",
]


def read_metadata(path: Path) -> InstructionMetadata:
    """Read a metadata.json file from disk using Pydantic.

    Args:
        path: Path to the metadata.json file.

    Returns:
        The parsed InstructionMetadata object.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the JSON is invalid or doesn't match the schema.
    """
    text = path.read_text(encoding="utf-8")
    return InstructionMetadata.model_validate_json(text)


def write_metadata(path: Path, data: InstructionMetadata) -> None:
    """Write metadata to disk atomically as UTF-8 JSON.

    This creates parent directories if they do not exist and writes with
    pretty formatting.

    Args:
        path: Destination path for metadata.json
        data: The InstructionMetadata object to write

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data.model_dump_json(indent=2, exclude_unset=True), encoding="utf-8")
    tmp.replace(path)
