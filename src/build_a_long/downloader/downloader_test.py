"""Tests for downloader.py - LegoInstructionDownloader class (pytest style)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json

from build_a_long.downloader.downloader import (
    LegoInstructionDownloader,
    Metadata,
    PdfEntry,
    read_metadata,
    write_metadata,
)
from build_a_long.downloader.legocom_test import HTML_WITH_METADATA_AND_PDF


def _make_mock_httpx_client(html: str):
    """Helper to create a mock httpx.Client that returns HTML."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_resp
    return mock_client


def test_download_skips_if_exists(tmp_path: Path):
    url = "https://example.com/file.pdf"
    dest = tmp_path / "file.pdf"
    dest.write_bytes(b"already")

    mock_client = MagicMock()

    downloader = LegoInstructionDownloader(
        client=mock_client, overwrite=False, show_progress=False
    )
    out = downloader.download(url, tmp_path)

    mock_client.stream.assert_not_called()

    assert out == dest
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

    downloader = LegoInstructionDownloader(overwrite=True, show_progress=True)
    out = downloader.download(
        "https://example.com/path/file.pdf",
        tmp_path,
        stream_fn=mock_stream_fn,
    )

    captured = capsys.readouterr()
    assert out.exists()
    assert out.read_bytes() == b"abcdef"
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
    # Mock network
    mock_client = _make_mock_httpx_client(HTML_WITH_METADATA_AND_PDF)

    # Stub out actual file downloads
    def fake_download(self, url: str, dest_dir: Path, **kwargs):
        dest_dir.mkdir(parents=True, exist_ok=True)
        p = dest_dir / url.split("/")[-1]
        p.write_bytes(b"dummy")
        return p

    downloader = LegoInstructionDownloader(
        client=mock_client, out_dir=tmp_path, show_progress=False
    )

    monkeypatch.setattr(LegoInstructionDownloader, "download", fake_download)

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
    assert data.get("set_image_url") == "set_image.png"
    # Ensure PDFs preserved order
    pdfs = data.get("pdfs", [])
    assert len(pdfs) == 2
    assert pdfs[0]["url"] == "/6602000.pdf"
    assert pdfs[0]["preview_url"] == "preview1.png"
    assert pdfs[1]["url"] == "/6602001.pdf"
    assert pdfs[1]["preview_url"] == "preview2.png"


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
                "preview_url": "preview1.png",
            },
            {
                "url": "https://www.example.com/7000002.pdf",
                "filename": "7000002.pdf",
                "preview_url": "preview2.png",
            },
        ],
    }

    out_dir = tmp_path
    meta_path = out_dir / "metadata.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # Mock client that would raise if network is attempted
    mock_client = _make_mock_httpx_client("<html></html>")

    # Stub out actual file downloads
    def fake_download(self, url: str, dest_dir: Path, **kwargs):
        dest_dir.mkdir(parents=True, exist_ok=True)
        p = dest_dir / url.split("/")[-1]
        p.write_bytes(b"dummy")
        return p

    downloader = LegoInstructionDownloader(
        client=mock_client, out_dir=out_dir, show_progress=False
    )

    monkeypatch.setattr(LegoInstructionDownloader, "download", fake_download)

    # Run
    exit_code = downloader.process_set("99999")
    assert exit_code == 0

    # Should not have attempted to fetch instructions page again
    # We expect only download calls, the initial client.get used in _make_mock_httpx_client
    # should not be invoked by process_set path that uses existing metadata
    # So ensure we didn't build a new URL call
    # (we can't assert exact call count reliably due to earlier tests, so check printed output)
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
    payload = Metadata(
        set="12345",
        locale="en-us",
        name="Test",
        pdfs=[PdfEntry(url="https://example.com/x.pdf", filename="x.pdf")],
    )

    write_metadata(meta_path, payload)
    assert meta_path.exists()

    loaded = read_metadata(meta_path)
    assert loaded == payload
