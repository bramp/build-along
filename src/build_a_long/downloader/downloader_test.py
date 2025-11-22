"""Tests for downloader.py - LegoInstructionDownloader class (pytest style)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from pydantic import AnyUrl

from build_a_long.downloader.downloader import (
    LegoInstructionDownloader,
    read_metadata,
    write_metadata,
)
from build_a_long.downloader.legocom_test import HTML_WITH_METADATA_AND_PDF
from build_a_long.downloader.models import DownloadedFile
from build_a_long.downloader.util import extract_filename_from_url
from build_a_long.schemas import InstructionMetadata, PdfEntry


def _make_mock_httpx_client(html: str):
    """Helper to create a mock httpx.Client that returns HTML."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_resp
    return mock_client


def _fake_download(self, url: str, dest_dir: Path, **kwargs):
    """Helper for tests to stub out actual file downloads."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = extract_filename_from_url(url)
    assert filename is not None, f"Could not extract filename from {url}"
    p = dest_dir / filename
    content = b"dummy"
    p.write_bytes(content)
    return DownloadedFile(path=p, size=len(content), hash="a" * 64)


def test_download_skips_if_exists(tmp_path: Path):
    url = AnyUrl("https://example.com/file.pdf")
    dest = tmp_path / "file.pdf"
    dest.write_bytes(b"already")

    mock_client = MagicMock()

    downloader = LegoInstructionDownloader(
        client=mock_client, overwrite_download=False, show_progress=False
    )
    out = downloader.download(url, tmp_path)

    mock_client.stream.assert_not_called()

    assert out.path == dest
    assert dest.read_bytes() == b"already"


def test_download_writes_and_shows_progress(tmp_path: Path, capsys):
    # Create a fake streaming response
    def _iter_raw(chunk_size=65536):
        yield b"abc"
        yield b"def"

    mock_resp = SimpleNamespace(
        headers={"Content-Length": str(6)},
        iter_raw=lambda chunk_size=65536: _iter_raw(chunk_size),
        raise_for_status=lambda: None,
    )

    def mock_stream_fn(method, url, **kwargs):
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_resp
        mock_ctx.__exit__ = MagicMock()
        return mock_ctx

    downloader = LegoInstructionDownloader(overwrite_download=True, show_progress=True)
    out = downloader.download(
        AnyUrl("https://example.com/path/file.pdf"),
        tmp_path,
        stream_fn=mock_stream_fn,
    )

    captured = capsys.readouterr()
    assert out.path.exists()
    assert out.path.read_bytes() == b"abcdef"
    # Progress output should include the filename
    assert "file.pdf:" in captured.out


def test_context_manager_closes_client():
    mock_client = MagicMock()
    with LegoInstructionDownloader(client=mock_client) as downloader:
        assert downloader is not None
    # Should not close the client if we provided it
    mock_client.close.assert_not_called()


def test_context_manager_creates_and_closes_client():
    with patch("httpx.Client") as mock_client_class:
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        with LegoInstructionDownloader() as downloader:
            # Client should be created when first accessed
            _ = downloader._get_client()

        # Should close the client we created
        mock_instance.close.assert_called_once()


def test_process_set_writes_metadata_json(tmp_path: Path, monkeypatch):
    mock_client = _make_mock_httpx_client(HTML_WITH_METADATA_AND_PDF)

    monkeypatch.setattr(LegoInstructionDownloader, "download", _fake_download)

    downloader = LegoInstructionDownloader(
        client=mock_client, out_dir=tmp_path, show_progress=False
    )

    exit_code = downloader.process_set("12345")
    assert exit_code == 0

    meta_path = tmp_path / "metadata.json"
    assert meta_path.exists()

    data = json.loads(meta_path.read_text())
    assert data["set"] == "12345"
    assert data.get("name") == "Starfighter"
    assert data.get("age") == "9+"
    assert data.get("pieces") == 1083
    assert data.get("year") == 2024
    assert data.get("set_image_url") == "https://www.lego.com/set_image.png"

    # Ensure PDFs preserved order
    pdfs = data.get("pdfs", [])
    assert len(pdfs) == 2
    assert pdfs[0]["url"] == "https://www.lego.com/6602000.pdf"
    assert pdfs[0]["preview_url"] == "https://www.lego.com/preview1.png"
    assert pdfs[0]["filesize"] == 5
    assert pdfs[0]["filehash"] == "a" * 64
    assert pdfs[1]["url"] == "https://www.lego.com/6602001.pdf"
    assert pdfs[1]["preview_url"] == "https://www.lego.com/preview2.png"
    assert pdfs[1]["filesize"] == 5
    assert pdfs[1]["filehash"] == "a" * 64


def test_process_set_uses_existing_metadata_and_skips_fetch(
    tmp_path: Path, monkeypatch, capsys
):
    """If metadata.json exists and contains PDFs, skip fetching HTML and reuse URLs."""
    # Prepare existing metadata with two PDFs
    meta = {
        "set": "99999",
        "locale": "en-us",
        "name": "Test Set",
        "theme": "LEGO® Theme",
        "age": "10+",
        "pieces": 123,
        "year": 2025,
        "pdfs": [
            {
                "url": "https://www.example.com/7000001.pdf",
                "filename": "7000001.pdf",
                "preview_url": "https://www.example.com/preview1.png",
                "size": 5,
                "hash": "a" * 64,
            },
            {
                "url": "https://www.example.com/7000002.pdf",
                "filename": "7000002.pdf",
                "preview_url": "https://www.example.com/preview2.png",
                "size": 5,
                "hash": "a" * 64,
            },
        ],
    }

    out_dir = tmp_path
    meta_path = out_dir / "metadata.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # Mock client that would raise if network is attempted
    mock_client = _make_mock_httpx_client("<html></html>")

    downloader = LegoInstructionDownloader(
        client=mock_client, out_dir=out_dir, show_progress=False
    )

    monkeypatch.setattr(LegoInstructionDownloader, "download", _fake_download)

    # Run
    exit_code = downloader.process_set("99999")
    assert exit_code == 0

    # Should not have attempted to fetch instructions page again
    # We expect only download calls, the initial client.get used in
    # _make_mock_httpx_client should not be invoked by process_set path that uses
    # existing metadata
    # So ensure we didn't build a new URL call
    # (we can't assert exact call count reliably due to earlier tests, so check
    # printed output)
    out = capsys.readouterr().out
    assert "Processing set: 99999 [cached]" in out
    assert "Found 2 PDF(s) for set 99999" in out
    assert "Test Set" in out

    # Files should be downloaded according to existing metadata
    assert (out_dir / "7000001.pdf").exists()
    assert (out_dir / "7000002.pdf").exists()


def test_read_metadata_handles_invalid_json(tmp_path: Path):
    # Write invalid JSON
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text("{ invalid json }", encoding="utf-8")

    data = read_metadata(meta_path)
    assert data is None


def test_write_and_read_metadata_round_trip(tmp_path: Path):
    meta_path = tmp_path / "metadata.json"
    payload = InstructionMetadata(
        set="12345",
        locale="en-us",
        name="Test",
        theme=None,
        age=None,
        pieces=None,
        year=None,
        set_image_url=None,
        pdfs=[
            PdfEntry(
                url=AnyUrl("https://example.com/x.pdf"),
                filename="x.pdf",
                preview_url=None,
                filesize=None,
                filehash=None,
            )
        ],
    )

    write_metadata(meta_path, payload)
    assert meta_path.exists()

    loaded = read_metadata(meta_path)
    assert loaded == payload


@patch(
    "build_a_long.downloader.downloader.LegoInstructionDownloader.fetch_instructions_page"
)
@patch("build_a_long.downloader.downloader.build_metadata")
def test_process_set_creates_not_found_on_empty_name(
    mock_build_metadata, mock_fetch_instructions_page, tmp_path: Path, capsys
):
    """Test that a .not_found file is created when metadata has an empty name."""
    # Mock build_metadata to return metadata with an empty name
    mock_build_metadata.return_value = InstructionMetadata(
        set="10516",
        locale="en-us",
        name="",
        pdfs=[],
    )
    mock_fetch_instructions_page.return_value = "<html></html>"

    set_number = "10516"
    out_dir = tmp_path / set_number
    not_found_path = out_dir / ".not_found"

    downloader = LegoInstructionDownloader(out_dir=out_dir, show_progress=False)
    exit_code = downloader.process_set(set_number)

    assert exit_code == 0
    assert not_found_path.exists()
    assert "Set 10516 not found or has no data on LEGO.com." in capsys.readouterr().out
    assert not (out_dir / "metadata.json").exists()


def test_process_set_skips_if_not_found_file_exists(tmp_path: Path, capsys):
    """Test that process_set skips a set if a .not_found file exists."""
    set_number = "12345"
    out_dir = tmp_path / set_number
    out_dir.mkdir()
    (out_dir / ".not_found").touch()

    downloader = LegoInstructionDownloader(
        out_dir=out_dir, overwrite_metadata=False, show_progress=False
    )
    exit_code = downloader.process_set(set_number)

    assert exit_code == 0
    assert "Skipping set 12345 (marked as not found)." in capsys.readouterr().out


@patch(
    "build_a_long.downloader.downloader.LegoInstructionDownloader.fetch_instructions_page"
)
def test_process_set_creates_not_found_file_on_404(
    mock_fetch_instructions_page, tmp_path: Path, capsys
):
    """Test that process_set creates a .not_found file on 404."""
    # Configure mock to raise httpx.HTTPStatusError with a 404 response
    mock_request = httpx.Request("GET", "http://test.com")
    mock_response = httpx.Response(404, request=mock_request)
    mock_fetch_instructions_page.side_effect = httpx.HTTPStatusError(
        "Not Found", request=mock_request, response=mock_response
    )

    set_number = "10516"
    out_dir = tmp_path / set_number
    not_found_path = out_dir / ".not_found"

    downloader = LegoInstructionDownloader(out_dir=out_dir, show_progress=False)
    exit_code = downloader.process_set(set_number)

    assert exit_code == 0
    assert not_found_path.exists()
    assert "Set 10516 not found on LEGO.com (404)." in capsys.readouterr().out
    assert not (out_dir / "metadata.json").exists()  # Should not create metadata.json

    # Test that it skips on subsequent runs
    capsys.readouterr()  # Clear previous output
    mock_fetch_instructions_page.reset_mock()
    exit_code_again = downloader.process_set(set_number)
    assert exit_code_again == 0
    assert "Skipping set 10516 (marked as not found)." in capsys.readouterr().out
    mock_fetch_instructions_page.assert_not_called()
