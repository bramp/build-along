from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, List, Optional

import httpx

from build_a_long.downloader.legocom import (
    LEGO_BASE,
    build_instructions_url,
    build_metadata,
    Metadata,
    PdfEntry,
    parse_instruction_pdf_urls,
)


def read_metadata(path: Path) -> Optional[Metadata]:
    """Read a metadata.json file from disk.

    Args:
        path: Path to the metadata.json file.

    Returns:
        The parsed Metadata object if successful; otherwise None.
    """
    import json

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            print(f"Warning: Metadata at {path} is not a JSON object; ignoring")
            return None

        # Build PdfEntry list
        pdf_entries: List[PdfEntry] = []
        for item in raw.get("pdfs", []) or []:
            if isinstance(item, dict) and item.get("url") and item.get("filename"):
                pdf_entries.append(PdfEntry(url=item["url"], filename=item["filename"]))

        meta = Metadata(
            set=str(raw.get("set", "")),
            locale=str(raw.get("locale", "en-us")),
            name=raw.get("name"),
            theme=raw.get("theme"),
            age=raw.get("age"),
            pieces=raw.get("pieces"),
            year=raw.get("year"),
            pdfs=pdf_entries,
        )
        return meta
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not read existing metadata ({e}); ignoring")
    return None


def write_metadata(path: Path, data: Metadata) -> None:
    """Write metadata to disk atomically as UTF-8 JSON.

    This creates parent directories if they do not exist and writes with
    pretty formatting. In case of failure, it emits a warning and does
    not raise to keep downloads non-fatal.

    Args:
        path: Destination path for metadata.json
        data: The Metadata object to write
    """
    import json
    from dataclasses import asdict

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(data), f, indent=2, ensure_ascii=False)
        tmp.replace(path)
        print(f"Wrote metadata: {path}")
    except Exception as e:  # pragma: no cover - non-fatal write error
        print(f"Warning: Failed to write metadata.json: {e}")


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
        overwrite: bool = False,
        show_progress: bool = True,
        client: Optional[httpx.Client] = None,
    ):
        """Initialize the downloader.

        Args:
            locale: LEGO locale to use (e.g., "en-us", "en-gb").
            out_dir: Base output directory for downloads.
            overwrite: If True, re-download existing files.
            show_progress: If True, show download progress.
            client: Optional httpx.Client to use (if None, creates one internally).
        """
        self.locale = locale
        self.out_dir = out_dir
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

    def fetch_instructions_page(self, set_number: str) -> str:
        """Fetch the HTML for the instructions page of a set."""
        url = build_instructions_url(set_number, self.locale)
        return self.fetch_url_text(url)

    def download(
        self,
        url: str,
        dest_dir: Path,
        *,
        progress_prefix: str = "",
        stream_fn: Optional[Callable[..., ContextManager[Any]]] = None,
        chunk_iter: Optional[Callable[[Any, int], Iterable[bytes]]] = None,
    ) -> Path:
        """Download a URL to a directory.

        Args:
            url: The file URL.
            dest_dir: Destination directory (created if missing).
            progress_prefix: Optional prefix for progress line (e.g., " - url").
            stream_fn: Injectable streaming function (for testing).
            chunk_iter: Optional injector to iterate raw chunks (for testing).

        Returns:
            Path to the downloaded file.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1]
        dest = dest_dir / filename

        if dest.exists() and not self.overwrite:
            if progress_prefix:
                print(f"{progress_prefix} [cached]")
            else:
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
                                if progress_prefix:
                                    print(
                                        f"{progress_prefix} {pct}%",
                                        end="\r",
                                        flush=True,
                                    )
                                else:
                                    print(f"  {filename}: {pct}%", end="\r", flush=True)
                                last_pct = pct
            if self.show_progress:
                if progress_prefix:
                    # Show final size on same line
                    size = dest.stat().st_size
                    print(f"{progress_prefix} [{size / 1_000_000:.2f} MB]")
                else:
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
        meta_path = out_dir / "metadata.json"

        # Try to load existing metadata first (if allowed)
        existing_meta: Optional[Metadata] = None
        if meta_path.exists() and not self.overwrite:
            existing_meta = read_metadata(meta_path)

        if existing_meta:
            print(f"Processing set: {set_number} [cached]")
        else:
            print(f"Processing set: {set_number}")

        if existing_meta and existing_meta.pdfs:
            # Use existing metadata to avoid re-fetching the index page
            meta_dc = None  # We'll skip writing metadata later
            pdfs = [p.url for p in existing_meta.pdfs]
            metadata = existing_meta
        else:
            # Build fresh metadata (includes ordered PDFs)
            meta_dc = self._build_metadata_for_set(set_number)
            pdfs = [p.url for p in meta_dc.pdfs]
            metadata = meta_dc

        if not pdfs:
            print(f"No PDFs found for set {set_number} (locale={self.locale}).")
            return 0

        # Print metadata info
        self._print_metadata_info(set_number, metadata, len(pdfs))

        # Write metadata.json alongside downloaded PDFs (only if we fetched it)
        if meta_dc is not None:
            write_metadata(meta_path, meta_dc)

        # Download each PDF with inline progress
        for i, u in enumerate(pdfs, 1):
            progress_prefix = f" - {u}"
            self.download(u, out_dir, progress_prefix=progress_prefix)

        return 0

    def _build_metadata_for_set(self, set_number: str):
        """Fetch the instructions page and build metadata dataclass for a set."""
        html = self.fetch_instructions_page(set_number)
        return build_metadata(html, set_number, self.locale, base=LEGO_BASE)

    def _print_metadata_info(
        self, set_number: str, metadata: Metadata, pdf_count: int
    ) -> None:
        """Print metadata information on a single line.

        Args:
            set_number: The LEGO set number.
            metadata: Metadata object.
            pdf_count: Number of PDFs found.
        """
        parts = [f"Found {pdf_count} PDF(s) for set {set_number}"]

        if metadata.name:
            parts.append(f"{metadata.name}")
        if metadata.theme:
            parts.append(f"({metadata.theme})")
        if metadata.pieces is not None:
            parts.append(f"{metadata.pieces} pieces")
        if metadata.age:
            parts.append(f"ages {metadata.age}")
        if metadata.year is not None:
            parts.append(f"released {metadata.year}")

        print(" - ".join(parts) + ":")

    def process_sets(self, set_numbers: List[str]) -> int:
        """Process multiple LEGO sets.

        Args:
            set_numbers: List of LEGO set numbers to process.

        Returns:
            Exit code: 0 for success, non-zero if any set failed.
        """
        overall_exit_code = 0
        for set_number in set_numbers:
            exit_code = self.process_set(set_number)
            if exit_code != 0:
                overall_exit_code = exit_code
        return overall_exit_code
