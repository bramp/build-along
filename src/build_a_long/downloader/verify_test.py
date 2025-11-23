import hashlib
from pathlib import Path

import pytest
from pydantic import AnyUrl

from build_a_long.downloader.verify import verify_data_integrity
from build_a_long.schemas import InstructionMetadata, PdfEntry


def create_dummy_file(path: Path, content: bytes):
    """Create a dummy file with given content, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Provides a temporary data directory for tests."""
    return tmp_path


def test_verify_data_integrity_happy_path(data_dir: Path, capsys):
    """Test the happy path where all files are correct."""
    set_dir = data_dir / "12345"
    set_dir.mkdir()

    pdf_content = b"dummy pdf content"
    pdf_path = set_dir / "12345-1.pdf"
    create_dummy_file(pdf_path, pdf_content)

    hasher = hashlib.sha256()
    hasher.update(pdf_content)
    pdf_hash = hasher.hexdigest()
    pdf_size = len(pdf_content)

    metadata = InstructionMetadata(
        set="12345",
        locale="en-us",
        pdfs=[
            PdfEntry(
                url=AnyUrl("http://example.com/1.pdf"),
                filename="12345-1.pdf",
                filesize=pdf_size,
                filehash=pdf_hash,
            )
        ],
    )

    metadata_path = set_dir / "metadata.json"
    metadata_path.write_text(metadata.model_dump_json(indent=2))

    assert verify_data_integrity(data_dir) == 0
    captured = capsys.readouterr()
    assert "Verification complete. No issues found." in captured.out


def test_verify_data_integrity_missing_file(data_dir: Path, capsys):
    """Test for a missing PDF file."""
    set_dir = data_dir / "12345"
    set_dir.mkdir()

    metadata = InstructionMetadata(
        set="12345",
        locale="en-us",
        pdfs=[
            PdfEntry(
                url=AnyUrl("http://example.com/1.pdf"),
                filename="12345-1.pdf",
                filesize=123,
                filehash="abc",
            )
        ],
    )

    metadata_path = set_dir / "metadata.json"
    metadata_path.write_text(metadata.model_dump_json(indent=2))

    assert verify_data_integrity(data_dir) == 1
    captured = capsys.readouterr()
    assert "Error: Missing file" in captured.out


def test_verify_data_integrity_filesize_mismatch(data_dir: Path, capsys):
    """Test for a filesize mismatch."""
    set_dir = data_dir / "12345"
    set_dir.mkdir()

    pdf_content = b"dummy pdf content"
    pdf_path = set_dir / "12345-1.pdf"
    create_dummy_file(pdf_path, pdf_content)

    metadata = InstructionMetadata(
        set="12345",
        locale="en-us",
        pdfs=[
            PdfEntry(
                url=AnyUrl("http://example.com/1.pdf"),
                filename="12345-1.pdf",
                filesize=len(pdf_content) + 1,  # Mismatch
                filehash="abc",
            )
        ],
    )

    metadata_path = set_dir / "metadata.json"
    metadata_path.write_text(metadata.model_dump_json(indent=2))

    assert verify_data_integrity(data_dir) == 1
    captured = capsys.readouterr()
    assert "Error: Filesize mismatch" in captured.out


def test_verify_data_integrity_hash_mismatch(data_dir: Path, capsys):
    """Test for a hash mismatch."""
    set_dir = data_dir / "12345"
    set_dir.mkdir()

    pdf_content = b"dummy pdf content"
    pdf_path = set_dir / "12345-1.pdf"
    create_dummy_file(pdf_path, pdf_content)

    hasher = hashlib.sha256()
    hasher.update(pdf_content)
    pdf_hash = hasher.hexdigest()

    metadata = InstructionMetadata(
        set="12345",
        locale="en-us",
        pdfs=[
            PdfEntry(
                url=AnyUrl("http://example.com/1.pdf"),
                filename="12345-1.pdf",
                filesize=len(pdf_content),
                filehash=pdf_hash + "wrong",  # Mismatch
            )
        ],
    )

    metadata_path = set_dir / "metadata.json"
    metadata_path.write_text(metadata.model_dump_json(indent=2))

    assert verify_data_integrity(data_dir) == 1
    captured = capsys.readouterr()
    assert "Error: Hash mismatch" in captured.out


def test_verify_data_integrity_no_metadata(data_dir: Path, capsys):
    """Test when no metadata files are present."""
    assert verify_data_integrity(data_dir) == 0
    captured = capsys.readouterr()
    assert "No metadata files found to verify." in captured.out


def test_verify_data_integrity_invalid_metadata(data_dir: Path, capsys):
    """Test with a malformed metadata.json file."""
    set_dir = data_dir / "12345"
    set_dir.mkdir()

    metadata_path = set_dir / "metadata.json"
    metadata_path.write_text("this is not json")

    assert verify_data_integrity(data_dir) == 1
    captured = capsys.readouterr()
    assert "Could not validate or parse metadata" in captured.out
