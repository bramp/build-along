from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, List, Optional

import httpx

from build_a_long.downloader.util import (
    LEGO_BASE,
    build_instructions_url,
    parse_instruction_pdf_urls,
)

# TODO Rename to Downloader


class LegoInstructionDownloader:
    """Downloader for LEGO instruction PDFs with shared HTTP client and configuration.

    This class maintains state for locale, output directory, and HTTP client,
    making it easier to test and reducing parameter passing.
    """

    def __init__(
        self,
        locale: str = "en-us",
        out_dir: Optional[Path] = None,
        dry_run: bool = False,
        overwrite: bool = False,
        show_progress: bool = True,
        client: Optional[httpx.Client] = None,
    ):
        """Initialize the downloader.

        Args:
            locale: LEGO locale to use (e.g., "en-us", "en-gb").
            out_dir: Base output directory for downloads.
            dry_run: If True, only list PDFs without downloading.
            overwrite: If True, re-download existing files.
            show_progress: If True, show download progress.
            client: Optional httpx.Client to use (if None, creates one internally).
        """
        self.locale = locale
        self.out_dir = out_dir
        self.dry_run = dry_run
        self.overwrite = overwrite
        self.show_progress = show_progress
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(follow_redirects=True, timeout=30)
        return self._client

    def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "LegoInstructionDownloader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def fetch_url_text(self, url: str) -> str:
        """Fetch a URL and return the response text.

        Args:
            url: The URL to fetch.

        Returns:
            The response body as text.
        """
        client = self._get_client()
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text

    def find_instruction_pdfs(self, set_number: str) -> List[str]:
        """Find all instruction PDF URLs for a LEGO set.

        Args:
            set_number: The LEGO set number (e.g., "75419").

        Returns:
            List of absolute PDF URLs.
        """
        url = build_instructions_url(set_number, self.locale)
        html = self.fetch_url_text(url)
        return parse_instruction_pdf_urls(html, base=LEGO_BASE)

    def download(
        self,
        url: str,
        dest_dir: Path,
        *,
        stream_fn: Optional[Callable[..., ContextManager[Any]]] = None,
        chunk_iter: Optional[Callable[[Any, int], Iterable[bytes]]] = None,
    ) -> Path:
        """Download a URL to a directory.

        Args:
            url: The file URL.
            dest_dir: Destination directory (created if missing).
            stream_fn: Injectable streaming function (for testing).
            chunk_iter: Optional injector to iterate raw chunks (for testing).

        Returns:
            Path to the downloaded file.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1]
        dest = dest_dir / filename

        if dest.exists() and not self.overwrite:
            print(f"Skip (exists): {dest}")
            return dest

        # Use injected stream_fn for testing, otherwise use client.stream
        if stream_fn is not None:
            stream_ctx = stream_fn("GET", url, follow_redirects=True, timeout=None)
        else:
            client = self._get_client()
            stream_ctx = client.stream("GET", url, follow_redirects=True, timeout=None)

        with stream_ctx as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0"))
            downloaded = 0
            last_pct = -1
            with open(dest, "wb") as f:
                raw_iter = (
                    chunk_iter(r, 64 * 1024)
                    if chunk_iter
                    else r.iter_raw(chunk_size=64 * 1024)
                )
                for chunk in raw_iter:
                    if not chunk:
                        continue
                    f.write(chunk)
                    if self.show_progress:
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int(downloaded * 100 / total)
                            if pct != last_pct:
                                print(f"  {filename}: {pct}%", end="\r", flush=True)
                                last_pct = pct
            if self.show_progress:
                # Clear the progress line
                print(" " * 60, end="\r")
        return dest

    def process_set(self, set_number: str) -> int:
        """Process and download instruction PDFs for a single LEGO set.

        Args:
            set_number: The LEGO set number to process.

        Returns:
            Exit code: 0 for success, non-zero for errors.
        """
        out_dir = self.out_dir if self.out_dir else Path("data") / set_number

        pdfs = self.find_instruction_pdfs(set_number)
        if not pdfs:
            print(f"No PDFs found for set {set_number} (locale={self.locale}).")
            return 0

        print(f"Found {len(pdfs)} PDF(s) for set {set_number}:")
        for u in pdfs:
            print(f" - {u}")

        if self.dry_run:
            print("Dry run: no files downloaded.")
            return 0

        print(f"Downloading to: {out_dir}")
        for u in pdfs:
            dest = self.download(u, out_dir)
            size = dest.stat().st_size
            print(f"Downloaded {dest} ({size / 1_000_000:.2f} MB)")

        return 0

    def process_sets(self, set_numbers: List[str]) -> int:
        """Process multiple LEGO sets.

        Args:
            set_numbers: List of LEGO set numbers to process.

        Returns:
            Exit code: 0 for success, non-zero if any set failed.
        """
        overall_exit_code = 0
        for set_number in set_numbers:
            print(f"Processing set: {set_number}")
            exit_code = self.process_set(set_number)
            if exit_code != 0:
                overall_exit_code = exit_code
        return overall_exit_code
