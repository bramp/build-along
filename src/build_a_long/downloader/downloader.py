from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import httpx
from pydantic import AnyUrl

from build_a_long.downloader.legocom import (
    LEGO_BASE,
    build_instructions_url,
    build_metadata,
)
from build_a_long.downloader.models import DownloadedFile
from build_a_long.schemas import (
    InstructionMetadata,
)

__all__ = [
    "LegoInstructionDownloader",
]


def read_metadata(path: Path) -> InstructionMetadata | None:
    """Read a metadata.json file from disk using Pydantic.

    Args:
        path: Path to the metadata.json file.

    Returns:
        The parsed InstructionMetadata object if successful; otherwise None.
    """
    try:
        text = path.read_text(encoding="utf-8")
        return InstructionMetadata.model_validate_json(text)
    except (
        OSError,
        ValueError,
    ) as e:
        print(f"Warning: Could not read existing metadata ({e}); ignoring")
    return None


def write_metadata(path: Path, data: InstructionMetadata) -> None:
    """Write metadata to disk atomically as UTF-8 JSON.

    This creates parent directories if they do not exist and writes with
    pretty formatting. In case of failure, it emits a warning and does
    not raise to keep downloads non-fatal.

    Args:
        path: Destination path for metadata.json
        data: The InstructionMetadata object to write
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            data.model_dump_json(indent=2, exclude_unset=True), encoding="utf-8"
        )
        tmp.replace(path)
        print(f"Wrote metadata: {path}")
    except Exception as e:  # pragma: no cover - non-fatal write error
        print(f"Warning: Failed to write metadata.json: {e}")


class LegoInstructionDownloader:
    """Downloader for LEGO instruction PDFs with shared HTTP client and configuration.

    This class maintains state for locale, output directory, and HTTP client,
    making it easier to test and reducing parameter passing.
    """

    # Suffix for files that mark a resource as not found.
    NOT_FOUND_SUFFIX = ".not_found"

    def __init__(
        self,
        locale: str = "en-us",
        out_dir: Path | None = None,
        overwrite_metadata: bool = False,
        overwrite_download: bool = False,
        show_progress: bool = True,
        client: httpx.Client | None = None,
        debug: bool = False,
    ):
        """Initialize the downloader.

        Args:
            locale: LEGO locale to use (e.g., "en-us", "en-gb").
            out_dir: Base output directory for downloads.
            overwrite_metadata: If True, re-download existing metadata.
            overwrite_download: If True, re-download existing files.
            show_progress: If True, show download progress.
            client: Optional httpx.Client to use (if None, creates one internally).
            debug: If True, enable debug output.
        """
        self.locale = locale
        self.out_dir = out_dir
        self.overwrite_metadata = overwrite_metadata
        self.overwrite_download = overwrite_download
        self.show_progress = show_progress
        self._client = client
        self._owns_client = client is None
        self.debug = debug

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

    def __enter__(self) -> LegoInstructionDownloader:
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

    def fetch_instructions_page(self, set_number: str) -> str:
        """Fetch the HTML for the instructions page of a set."""
        url = build_instructions_url(set_number, self.locale)
        return self.fetch_url_text(url)

    def download(
        self,
        url: AnyUrl,
        dest_path: Path,
        *,
        progress_prefix: str = "",
        stream_fn: Callable[..., AbstractContextManager[Any]] | None = None,
        chunk_iter: Callable[[Any, int], Iterable[bytes]] | None = None,
    ) -> DownloadedFile:
        """Download a URL to a specific path.

        Args:
            url: The file URL.
            dest_path: Destination path for the downloaded file (parent dir created if missing).
            progress_prefix: Optional prefix for progress line (e.g., " - url").
            stream_fn: Injectable streaming function (for testing).
            chunk_iter: Optional injector to iterate raw chunks (for testing).

        Returns:
            Path to the downloaded file, its size, and its SHA256 hash.
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        filename = dest_path.name

        # Use injected stream_fn for testing, otherwise use client.stream
        if stream_fn is None:
            client = self._get_client()
            stream_fn = client.stream

        file_hash_obj = hashlib.sha256()
        with stream_fn("GET", str(url), follow_redirects=True, timeout=None) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0"))
            downloaded = 0
            last_pct = -1
            with open(dest_path, "wb") as f:
                raw_iter = (
                    chunk_iter(r, 64 * 1024)
                    if chunk_iter
                    else r.iter_raw(chunk_size=64 * 1024)
                )
                for chunk in raw_iter:
                    if not chunk:
                        continue
                    f.write(chunk)
                    file_hash_obj.update(chunk)
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
                                    print(
                                        f"  {filename}: {pct}%",
                                        end="\r",
                                        flush=True,
                                    )
                                last_pct = pct
            if self.show_progress:
                if progress_prefix:
                    # Show final size on same line
                    size = dest_path.stat().st_size
                    print(f"{progress_prefix} [{size / 1_000_000:.2f} MB]")
                else:
                    # Clear the progress line
                    print(" " * 60, end="\r")
        file_size = dest_path.stat().st_size
        file_hash = file_hash_obj.hexdigest()
        return DownloadedFile(path=dest_path, size=file_size, hash=file_hash)

    def _process_set_metadata(
        self, set_number: str, out_dir: Path
    ) -> tuple[InstructionMetadata, bool] | None:
        """Fetch and cache metadata for a single LEGO set.

        This method handles the logic for checking for existing metadata,
        fetching it from the LEGO website if necessary, and storing it
        in a `metadata.json` file. It also creates a `.not_found` file
        if the set is not found on the website.

        Args:
            set_number: The LEGO set number.
            out_dir: The output directory for the set.

        Returns:
            A tuple of the `InstructionMetadata` and a boolean indicating
            if the metadata was loaded from cache, or `None` if the set
            was not found.
        """
        meta_path = out_dir / "metadata.json"
        not_found_path = out_dir / self.NOT_FOUND_SUFFIX

        # If a .not_found file exists, and we're not forcing a metadata
        # update, skip this set.
        if not_found_path.exists() and not self.overwrite_metadata:
            print(f"Skipping set {set_number} (marked as not found).")
            return None

        # If metadata.json exists and we're not forcing an update,
        # try to load it. If it contains PDFs, we can use it.
        if meta_path.exists() and not self.overwrite_metadata:
            existing_meta = read_metadata(meta_path)
            if existing_meta and existing_meta.pdfs:
                print(f"Processing set: {set_number} [cached]")
                return existing_meta, True

        # If we're here, we need to fetch the metadata from the website.
        print(f"Processing set: {set_number}")
        try:
            html = self.fetch_instructions_page(set_number)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(f"Set {set_number} not found on LEGO.com (404).")
                out_dir.mkdir(parents=True, exist_ok=True)
                not_found_path.touch()
                return None
            raise

        metadata = build_metadata(
            html, set_number, self.locale, base=LEGO_BASE, debug=self.debug
        )

        # If no metadata is found, mark it as not found and return.
        if not metadata.name:
            print(f"Set {set_number} not found or has no data on LEGO.com.")
            out_dir.mkdir(parents=True, exist_ok=True)
            not_found_path.touch()
            return None

        # Write the new metadata to disk.
        write_metadata(meta_path, metadata)
        return metadata, False

    def _process_set_pdfs(self, metadata: InstructionMetadata, out_dir: Path) -> bool:
        """Download PDFs for a single LEGO set.

        This method iterates through the PDFs in the metadata, and for each
        one, it checks if it already exists or is marked as not found. If
        not, it downloads the PDF and updates the metadata with the file
        size and hash.

        Args:
            metadata: The `InstructionMetadata` for the set.
            out_dir: The output directory for the set.

        Returns:
            `True` if all PDFs were processed successfully, `False` otherwise.
        """
        if not metadata.pdfs:
            print(f"No PDFs found for set {metadata.set} (locale={metadata.locale}).")
            return False

        for entry in metadata.pdfs:
            dest_path = out_dir / entry.filename
            not_found_path = dest_path.with_suffix(
                dest_path.suffix + self.NOT_FOUND_SUFFIX
            )
            progress_prefix = f" - {entry.url}"

            # If a .not_found file exists for this PDF and we're not forcing
            # a re-download, skip it.
            if not_found_path.exists() and not self.overwrite_download:
                print(f"{progress_prefix} [cached - not found]")
                continue

            # If the PDF file exists and we're not forcing a re-download,
            # skip it, but update the filesize from the existing file.
            if dest_path.exists() and not self.overwrite_download:
                print(f"{progress_prefix} [cached]")
                entry.filesize = dest_path.stat().st_size
                continue

            # Try to download the PDF.
            try:
                downloaded_file = self.download(
                    entry.url, dest_path, progress_prefix=progress_prefix
                )
                entry.filesize = downloaded_file.size
                entry.filehash = downloaded_file.hash
            except httpx.HTTPStatusError as e:
                # If the download fails with a 404, create a .not_found file
                # so we don't try again next time.
                if e.response.status_code == 404:
                    print(f"Warning: PDF not found: {entry.url} (404). Skipping.")
                    not_found_path.touch()
                else:
                    # For other HTTP errors, we re-raise the exception.
                    raise
        return True

    def process_set(self, set_number: str) -> int:
        """Process and download instruction PDFs for a single LEGO set.

        This method coordinates the processing of a single set, by first
        fetching the metadata and then downloading the associated PDFs.

        Args:
            set_number: The LEGO set number to process.

        Returns:
            Exit code: 0 for success, non-zero for errors.
        """
        out_dir = self.out_dir if self.out_dir else Path("data") / set_number

        # Process the metadata for the set.
        result = self._process_set_metadata(set_number, out_dir)
        if not result:
            return 0  # Metadata processing handled the output

        metadata, use_cached = result
        self._print_metadata_info(set_number, metadata)

        # Process the PDFs for the set.
        if self._process_set_pdfs(metadata, out_dir) and not use_cached:
            # If the metadata was not loaded from cache (i.e. it's new or
            # updated), write the updated metadata back to disk.
            write_metadata(out_dir / "metadata.json", metadata)

        return 0

    def _print_metadata_info(
        self,
        set_number: str,
        metadata: InstructionMetadata,
    ) -> None:
        """Print metadata information on a single line.

        Args:
            set_number: The LEGO set number.
            metadata: InstructionMetadata object.
        """
        parts = [f"Found {len(metadata.pdfs)} PDF(s) for set {set_number}"]

        if metadata.name:
            parts.append(f"{metadata.name}")
        if metadata.theme:
            parts.append(f"({metadata.theme})")
        if metadata.pieces is not None:
            parts.append(f"({metadata.pieces} pieces)")
        if metadata.age:
            parts.append(f"ages {metadata.age}")
        if metadata.year is not None:
            parts.append(f"released {metadata.year}")

        print(" - ".join(parts) + ":")

    def process_sets(self, set_numbers: list[str]) -> int:
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
