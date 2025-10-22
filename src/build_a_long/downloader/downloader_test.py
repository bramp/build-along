"""Tests for downloader.py - LegoInstructionDownloader class (pytest style)."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from build_a_long.downloader.downloader import LegoInstructionDownloader


def _make_mock_httpx_client(html: str):
    """Helper to create a mock httpx.Client that returns HTML."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_resp
    return mock_client


def test_find_instruction_pdfs_parses_and_normalizes():
    html = (
        '<a href="https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf">Download</a>'
        '<a href="/cdn/product-assets/product.bi.core.pdf/6602645.pdf">Download</a>'
        '<a href="/cdn/x/notpdf.txt">Ignore</a>'
    )
    mock_client = _make_mock_httpx_client(html)
    downloader = LegoInstructionDownloader(client=mock_client)

    urls = downloader.find_instruction_pdfs("75419")
    assert urls == [
        "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf",
        "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602645.pdf",
    ]


def test_find_instruction_pdfs_uses_locale():
    html = '<a href="/cdn/product-assets/product.bi.core.pdf/test.pdf">Download</a>'
    mock_client = _make_mock_httpx_client(html)
    downloader = LegoInstructionDownloader(locale="de-de", client=mock_client)

    urls = downloader.find_instruction_pdfs("12345")
    # Verify the URL was built with the correct locale
    mock_client.get.assert_called_once()
    called_url = mock_client.get.call_args[0][0]
    assert "de-de" in called_url
    assert urls == [
        "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/test.pdf"
    ]


def test_download_skips_if_exists(tmp_path: Path):
    url = "https://example.com/file.pdf"
    dest = tmp_path / "file.pdf"
    dest.write_bytes(b"already")

    downloader = LegoInstructionDownloader(
        overwrite=False, show_progress=False)
    out = downloader.download(url, tmp_path)

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


def test_process_set_dry_run(capsys):
    html = '<a href="/cdn/product-assets/product.bi.core.pdf/test.pdf">Download</a>'
    mock_client = _make_mock_httpx_client(html)

    downloader = LegoInstructionDownloader(
        dry_run=True, client=mock_client, show_progress=False
    )
    exit_code = downloader.process_set("12345")

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Found 1 PDF(s)" in output
    assert "Dry run: no files downloaded" in output
