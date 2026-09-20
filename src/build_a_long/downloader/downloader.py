from __future__ import annotations

import datetime
import hashlib
import json
import os
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import httpx
from pydantic import AnyUrl

from build_a_long.downloader.legocom import (
    LEGO_BASE,
    build_instructions_url,
    fetch_metadata,
)
from build_a_long.downloader.metadata import read_metadata, write_metadata
from build_a_long.downloader.models import DownloadedFile, DownloaderStats
from build_a_long.downloader.set_sources import (
    fetch_lego_sitemap_sets,
    fetch_rebrickable_sets,
)
from build_a_long.downloader.transport import RateLimitedTransport
from build_a_long.downloader.util import extract_filename_from_url
from build_a_long.schemas import (
    LegoSetMetadata,
)


def get_data_dir() -> Path | None:
    """Get the LEGO data directory from the LEGO_DATA_DIR environment variable."""
    if env_dir := os.environ.get("LEGO_DATA_DIR"):
        return Path(env_dir)
    return None


__all__ = [
    "LegoInstructionDownloader",
    "get_data_dir",
]


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
        data_dir: Path | None = None,
        overwrite_metadata_if_older_than: datetime.timedelta | None = None,
        overwrite_download: bool = False,
        show_progress: bool = True,
        client: httpx.Client | None = None,
        debug: bool = False,
        max_calls: int = 60,
        period: int = 60,
        skip_pdfs: bool = False,
        released_within_years: int | None = None,
    ):
        """Initialize the downloader.

        Args:
            locale: LEGO locale to use (e.g., "en-us", "en-gb").
            data_dir: Base directory containing downloaded LEGO set data.
            overwrite_metadata_if_older_than: Overwrite metadata if older than this
                timedelta.
            overwrite_download: If True, re-download existing files.
            show_progress: If True, show download progress.
            client: Optional httpx.Client to use (if None, creates one internally).
            debug: If True, enable debug output.
            max_calls: Maximum number of calls to allow in a period (defaults to 60).
            period: The time period in seconds (defaults to 60).
            skip_pdfs: If True, only download metadata, skip PDF downloads.
            released_within_years: If set, only overwrite metadata for sets released
                within the last N years.
        """
        self.locale = locale
        self.data_dir = data_dir
        self.overwrite_metadata_if_older_than = overwrite_metadata_if_older_than
        self.overwrite_download = overwrite_download
        self.show_progress = show_progress
        self._client = client
        self._owns_client = client is None
        self.debug = debug
        self.max_calls = max_calls
        self.period = period
        self.skip_pdfs = skip_pdfs
        self.released_within_years = released_within_years

        # Statistics
        self.stats = DownloaderStats()

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            transport = RateLimitedTransport(
                max_calls=self.max_calls, period=self.period
            )
            self._client = httpx.Client(
                transport=transport, follow_redirects=True, timeout=30
            )
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

    def fetch_set_metadata(self, set_number: str) -> LegoSetMetadata | None:
        """Fetch complete set metadata using GraphQL first, falling back to HTML."""
        client = self._get_client()
        return fetch_metadata(
            client=client,
            set_number=set_number,
            locale=self.locale,
            base=LEGO_BASE,
            debug=self.debug,
        )

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
            dest_path: Destination path for the downloaded file (parent dir created if
                missing).
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
            assert stream_fn is not None

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

    def _load_existing_metadata(self, meta_path: Path) -> LegoSetMetadata | None:
        """Load metadata from disk if present, handling errors gracefully."""
        if meta_path.exists():
            try:
                return read_metadata(meta_path)
            except (OSError, ValueError) as e:
                print(f"Warning: Could not read {meta_path}: {e}")
        return None

    def _should_overwrite_metadata(
        self,
        meta_path: Path,
        existing_meta: LegoSetMetadata | None,
        set_number: str,
    ) -> bool:
        """Check whether existing metadata should be overwritten based on age and release year."""
        if self.overwrite_metadata_if_older_than is None or not meta_path.exists():
            return False

        if (
            self.released_within_years is not None
            and existing_meta
            and existing_meta.year is not None
        ):
            current_year = datetime.datetime.now(datetime.timezone.utc).year
            min_year = current_year - self.released_within_years
            if existing_meta.year < min_year:
                if self.debug:
                    print(
                        f"Set {set_number} released in {existing_meta.year} (< {min_year}). Skipping overwrite."
                    )
                return False

        if existing_meta and existing_meta.last_updated:
            file_mtime = existing_meta.last_updated
        else:
            file_mtime = datetime.datetime.fromtimestamp(
                meta_path.stat().st_mtime, tz=datetime.timezone.utc
            )

        if file_mtime.tzinfo is None:
            file_mtime = file_mtime.replace(tzinfo=datetime.timezone.utc)

        now = datetime.datetime.now(datetime.timezone.utc)
        if (now - file_mtime) > self.overwrite_metadata_if_older_than:
            print(
                f"Metadata for set {set_number} is older than specified duration. Overwriting."
            )
            return True
        return False

    def _mark_set_not_found(self, out_dir: Path, message: str) -> None:
        """Mark a set as not found by creating a .not_found file and updating stats."""
        print(message)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / self.NOT_FOUND_SUFFIX).touch()
        self.stats.sets_not_found += 1

    def _merge_existing_pdf_info(
        self,
        metadata: LegoSetMetadata,
        existing_meta: LegoSetMetadata | None,
    ) -> None:
        """Carry over filename, filesize, and filehash from matching existing PDFs."""
        if not existing_meta:
            return
        existing_pdfs_by_url = {str(p.url): p for p in existing_meta.pdfs}
        for pdf in metadata.pdfs:
            if existing_pdf := existing_pdfs_by_url.get(str(pdf.url)):
                if not pdf.filename and existing_pdf.filename:
                    pdf.filename = existing_pdf.filename
                if not pdf.filesize:
                    pdf.filesize = existing_pdf.filesize
                if not pdf.filehash:
                    pdf.filehash = existing_pdf.filehash

    def _process_set_metadata(
        self,
        set_number: str,
        out_dir: Path,
    ) -> tuple[LegoSetMetadata, bool] | None:
        """Fetch and cache metadata for a single LEGO set.

        This method handles the logic for checking for existing metadata,
        fetching it from the LEGO website if necessary, and storing it
        in a `metadata.json` file. It also creates a `.not_found` file
        if the set is not found on the website.

        Args:
            set_number: The LEGO set number.
            out_dir: The output directory for the set.

        Returns:
            A tuple of the `LegoSetMetadata` and a boolean indicating
            if the metadata was loaded from cache, or `None` if the set
            was not found.
        """
        meta_path = out_dir / "metadata.json"
        not_found_path = out_dir / self.NOT_FOUND_SUFFIX

        existing_meta = self._load_existing_metadata(meta_path)
        should_overwrite = self._should_overwrite_metadata(
            meta_path, existing_meta, set_number
        )

        # If a .not_found file exists, and we're not forcing an update, skip this set.
        if not_found_path.exists() and not should_overwrite:
            print(f"Skipping set {set_number} (marked as not found).")
            self.stats.sets_not_found += 1
            return None

        # If metadata.json exists and we're not forcing an update, use the cached metadata.
        if existing_meta and not should_overwrite:
            print(f"Processing set: {set_number} [cached]")
            self.stats.sets_found += 1
            return existing_meta, True

        # Fetch fresh metadata from the website
        print(f"Processing set: {set_number}")
        try:
            metadata = self.fetch_set_metadata(set_number)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self._mark_set_not_found(
                    out_dir, f"Set {set_number} not found on LEGO.com (404)."
                )
                return None
            raise

        if not metadata or not metadata.name:
            self._mark_set_not_found(
                out_dir, f"Set {set_number} not found or has no data on LEGO.com."
            )
            return None

        # Carry over existing downloaded PDF details
        self._merge_existing_pdf_info(metadata, existing_meta)

        # Write the new metadata to disk.
        try:
            write_metadata(meta_path, metadata)
            print(f"Wrote metadata: {meta_path}")
        except OSError as e:
            print(f"Warning: Failed to write {meta_path}: {e}")
        self.stats.sets_found += 1
        return metadata, False

    def _process_set_pdfs(self, metadata: LegoSetMetadata, out_dir: Path) -> bool:
        """Download PDFs for a single LEGO set.

        This method iterates through the PDFs in the metadata, and for each
        one, it checks if it already exists or is marked as not found. If
        not, it downloads the PDF and updates the metadata with the file
        size and hash.

        Args:
            metadata: The `LegoSetMetadata` for the set.
            out_dir: The output directory for the set.

        Returns:
            `True` if all PDFs were processed successfully, `False` otherwise.
        """
        if not metadata.pdfs:
            print(f"No PDFs found for set {metadata.set} (locale={metadata.locale}).")
            return False

        self.stats.pdfs_found += len(metadata.pdfs)

        for entry in metadata.pdfs:
            # Determine destination filename:
            # 1. Use existing entry.filename if present
            # 2. Extract from URL
            # 3. Skip if undetermined (should likely warn)
            filename = entry.filename or extract_filename_from_url(entry.url)
            if not filename:
                print(
                    f"Warning: Could not determine filename for {entry.url}. Skipping."
                )
                continue

            dest_path = out_dir / filename
            not_found_path = dest_path.with_suffix(
                dest_path.suffix + self.NOT_FOUND_SUFFIX
            )
            progress_prefix = f" - {entry.url}"

            if self.debug:
                print(f"DEBUG: Checking {dest_path} (Exists: {dest_path.exists()})")

            # If a .not_found file exists for this PDF and we're not forcing
            # a re-download, skip it.
            if not_found_path.exists() and not self.overwrite_download:
                print(f"{progress_prefix} [cached - not found]")
                self.stats.pdfs_skipped += 1
                continue

            # If the PDF file exists and we're not forcing a re-download,
            # skip it, but update the filesize from the existing file.
            if dest_path.exists() and not self.overwrite_download:
                print(f"{progress_prefix} [cached]")
                entry.filesize = dest_path.stat().st_size
                # Ensure filename is set if it was missing
                entry.filename = filename
                self.stats.pdfs_skipped += 1
                continue

            # Try to download the PDF.
            try:
                downloaded_file = self.download(
                    entry.url, dest_path, progress_prefix=progress_prefix
                )
                entry.filesize = downloaded_file.size
                entry.filehash = downloaded_file.hash
                # Update filename in entry to match downloaded file
                entry.filename = downloaded_file.path.name
                self.stats.pdfs_downloaded += 1
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
        self.stats.sets_processed += 1
        base_dir = self.data_dir if self.data_dir else get_data_dir()
        if not base_dir:
            raise ValueError(
                "No data directory specified. Provide data_dir or set the LEGO_DATA_DIR environment variable."
            )
        out_dir = base_dir / set_number

        # Process the metadata for the set.
        result = self._process_set_metadata(set_number, out_dir)
        if not result:
            return 0  # Metadata processing handled the output

        metadata, use_cached = result
        self._print_metadata_info(set_number, metadata)

        # Skip PDF downloads if skip_pdfs is True
        if self.skip_pdfs:
            self.stats.pdfs_found += len(metadata.pdfs)
            return 0

        # Process the PDFs for the set.
        # We always call this to ensure stats are updated and missing files are
        # downloaded, even if metadata was cached.
        pdfs_processed = self._process_set_pdfs(metadata, out_dir)

        if pdfs_processed and not use_cached:
            # If the metadata was not loaded from cache (i.e. it's new or
            # updated), write the updated metadata back to disk.
            try:
                write_metadata(out_dir / "metadata.json", metadata)
                print(f"Wrote metadata: {out_dir / 'metadata.json'}")
            except OSError as e:
                print(f"Warning: Failed to write {out_dir / 'metadata.json'}: {e}")

        return 0

    def _print_metadata_info(
        self,
        set_number: str,
        metadata: LegoSetMetadata,
    ) -> None:
        """Print metadata information on a single line.

        Args:
            set_number: The LEGO set number.
            metadata: LegoSetMetadata object.
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

    def process_sets(self, set_numbers: list[str]) -> DownloaderStats:
        """Process multiple LEGO sets.

        Args:
            set_numbers: List of LEGO set numbers to process.

        Returns:
            A DownloaderStats object containing statistics of the operation.
        """
        for set_number in set_numbers:
            self.process_set(set_number)
        return self.stats

    def _get_cache_dir(self) -> Path:
        """Resolve the cache directory for downloaded artifacts and set lists."""
        if self.data_dir:
            cache_dir = self.data_dir / ".cache"
        else:
            xdg_cache = os.environ.get("XDG_CACHE_HOME")
            base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
            cache_dir = base / "build-along"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def get_set_list(
        self,
        source: str = "lego",
        *,
        use_cache: bool = True,
        cache_ttl: datetime.timedelta | None = datetime.timedelta(days=1),
        min_year: int | None = None,
    ) -> list[str]:
        """Fetch list of set numbers from LEGO.com sitemap or Rebrickable.

        Results can be cached to disk to avoid repeated network calls.

        Args:
            source: Source to query ('lego' or 'rebrickable'). Defaults to 'lego'.
            use_cache: Whether to use cached set list if still within cache_ttl.
            cache_ttl: How long cached results remain valid. Set to None to keep indefinitely.
            min_year: Optional minimum release year to filter by (supported for 'rebrickable').

        Returns:
            Sorted list of unique set number strings.
        """
        source_key = source.lower().strip()
        if source_key in ("lego", "lego.com"):
            cache_filename = f"set_list_lego_{self.locale}.json"
        elif source_key == "rebrickable":
            year_part = f"_min{min_year}" if min_year is not None else ""
            cache_filename = f"set_list_rebrickable{year_part}.json"
        else:
            raise ValueError(
                f"Unknown set list source: '{source}'. Supported: 'lego', 'rebrickable'"
            )

        cache_path = self._get_cache_dir() / cache_filename

        if use_cache and cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_at_str = data.get("_timestamp")
                if cached_at_str and cache_ttl is not None:
                    cached_at = datetime.datetime.fromisoformat(cached_at_str)
                    if cached_at.tzinfo is None:
                        cached_at = cached_at.replace(tzinfo=datetime.timezone.utc)
                    now = datetime.datetime.now(datetime.timezone.utc)
                    if (now - cached_at) <= cache_ttl:
                        if self.debug:
                            print(
                                f"Loaded {len(data['sets'])} sets from cache: {cache_path}"
                            )
                        return list(data["sets"])
                elif cache_ttl is None:
                    return list(data["sets"])
            except Exception as e:
                if self.debug:
                    print(f"Warning: Failed to read cache {cache_path}: {e}")

        # Fetch fresh list
        client = self._get_client()
        if source_key in ("lego", "lego.com"):
            sets = fetch_lego_sitemap_sets(client, locale=self.locale, base=LEGO_BASE)
        else:
            sets = fetch_rebrickable_sets(client, min_year=min_year)

        # Cache to disk
        try:
            cache_record = {
                "_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source": source_key,
                "sets": sets,
            }
            cache_path.write_text(json.dumps(cache_record, indent=2), encoding="utf-8")
            if self.debug:
                print(f"Cached {len(sets)} sets to {cache_path}")
        except Exception as e:
            if self.debug:
                print(f"Warning: Failed to write cache {cache_path}: {e}")

        return sets
