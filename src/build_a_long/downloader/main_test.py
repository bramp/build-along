import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from build_a_long.downloader.main import (
    download,
    find_instruction_pdfs,
)


class DownloaderTests(TestCase):
    def _mock_httpx_client(self, html: str):
        mock_client_ctx = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_ctx.__enter__.return_value = mock_client
        mock_client_ctx.__exit__ = MagicMock()
        return mock_client_ctx

    @patch("httpx.Client")
    def test_find_instruction_pdfs_parses_and_normalizes(self, client_cls):
        html = (
            '<a href="https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf">Download</a>'
            '<a href="/cdn/product-assets/product.bi.core.pdf/6602645.pdf">Download</a>'
            '<a href="/cdn/x/notpdf.txt">Ignore</a>'
        )
        client_cls.return_value = self._mock_httpx_client(html)

        urls = find_instruction_pdfs("75419", "en-us")
        self.assertEqual(
            urls,
            [
                "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf",
                "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602645.pdf",
            ],
        )

    @patch("httpx.Client")
    def test_find_instruction_pdfs_deduplicates(self, client_cls):
        html = (
            '<a href="https://www.lego.com/cdn/product-assets/product.bi.core.pdf/dup.pdf">A</a>'
            '<a href="/cdn/product-assets/product.bi.core.pdf/dup.pdf">B</a>'
        )
        client_cls.return_value = self._mock_httpx_client(html)

        urls = find_instruction_pdfs("1234", "en-us")
        self.assertEqual(
            urls,
            ["https://www.lego.com/cdn/product-assets/product.bi.core.pdf/dup.pdf"],
        )

    @patch("httpx.Client")
    def test_find_instruction_pdfs_filters_non_instructions(self, client_cls):
        html = (
            '<a href="https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf">Instruction</a>'
            '<a href="https://www.lego.com/cdn/cs/aboutus/assets/blt1a02e1065ccb2f31/LEGOGroup_ModernSlaveryTransparencyStatement_2024.pdf">Non-Instruction</a>'
        )
        client_cls.return_value = self._mock_httpx_client(html)

        urls = find_instruction_pdfs("75419", "en-us")
        self.assertEqual(
            urls,
            [
                "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf",
            ],
        )

    @patch("httpx.stream")
    def test_download_skips_if_exists(self, mock_stream):
        # Arrange
        tmp = Path(os.getcwd()) / ".tmp_test_download"
        tmp.mkdir(exist_ok=True)
        try:
            url = "https://example.com/file.pdf"
            dest = tmp / "file.pdf"
            dest.write_bytes(b"already")

            # Act
            out = download(url, tmp, overwrite=False, show_progress=False)

            # Assert
            self.assertEqual(out, dest)
            mock_stream.assert_not_called()
            self.assertEqual(dest.read_bytes(), b"already")
        finally:
            for p in tmp.glob("*"):
                p.unlink()
            tmp.rmdir()

    @patch("httpx.stream")
    def test_download_writes_and_shows_progress(self, mock_stream):
        # Create a fake streaming response
        def _iter_raw(chunk_size=65536):
            yield b"abc"
            yield b"def"

        mock_resp = SimpleNamespace(
            headers={"Content-Length": str(6)},
            iter_raw=lambda chunk_size=65536: _iter_raw(chunk_size),
            raise_for_status=lambda: None,
        )
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_resp
        mock_ctx.__exit__ = MagicMock()
        mock_stream.return_value = mock_ctx

        tmp = Path(os.getcwd()) / ".tmp_test_download2"
        tmp.mkdir(exist_ok=True)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                out = download(
                    "https://example.com/path/file.pdf",
                    tmp,
                    overwrite=True,
                    show_progress=True,
                )
            self.assertTrue(out.exists())
            self.assertEqual(out.read_bytes(), b"abcdef")
            # Progress output should include the filename
            self.assertIn("file.pdf:", buf.getvalue())
        finally:
            for p in tmp.glob("*"):
                p.unlink()
            tmp.rmdir()
