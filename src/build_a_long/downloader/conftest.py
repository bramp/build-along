"""Pytest configuration for downloader tests.

This configures VCR.py (via pytest-recording) to record and replay HTTP interactions
for integration tests, making them fast and avoiding repeated requests to LEGO.com.

Enhancements:
- Store cassettes in a persistent user cache directory (not in the repo)
- Time-based refresh: delete cassettes older than N days before running (statically
  configured)
"""

from __future__ import annotations

import os
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest


def _default_cassette_dir() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return base / "build-along" / "cassettes" / "downloader"


def _resolve_cassette_dir() -> Path:
    override = os.environ.get("CASSETTE_DIR")
    return Path(override).expanduser() if override else _default_cassette_dir()


CASSETTE_MAX_AGE_DAYS = 7  # Statically configured retention period


@pytest.fixture(scope="function")
def vcr_config(vcr_cassette_name: str):
    """Configure VCR for recording HTTP interactions.

    Cassettes (recorded interactions) are stored in a persistent cache dir, not the
    repo.
    They will be automatically created on first run and reused thereafter.

        Time-based refresh:
        - If a cassette file exists and is older than CASSETTE_MAX_AGE_DAYS, it is
          deleted
            before the test runs so it will be re-recorded.

    To force-refresh all cassettes manually:
    - Delete the cassette directory, or
    - Run with: pytest --record-mode=rewrite (or via Pants:
      pants test --pytest-args="--record-mode=rewrite")
    """
    cassette_dir = _resolve_cassette_dir()
    cassette_dir.mkdir(parents=True, exist_ok=True)

    # Time-based refresh of a single cassette (statically configured)
    cassette_path = cassette_dir / f"{vcr_cassette_name}.yaml"
    if cassette_path.exists():
        mtime = datetime.fromtimestamp(cassette_path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=CASSETTE_MAX_AGE_DAYS):
            with suppress(OSError):
                cassette_path.unlink()

    return {
        # Store cassettes in a dedicated, persistent directory
        "cassette_library_dir": str(cassette_dir),
        # Match requests by method and URI (not body, as we don't send POST data)
        "match_on": ["method", "uri"],
        # Record once, then replay
        "record_mode": "once",
        # Filter sensitive data (none for LEGO.com, but good practice)
        "filter_headers": [],
        # Decode compressed responses for better diffs
        "decode_compressed_response": True,
        # Ignore localhost/internal requests (if any)
        "ignore_localhost": True,
    }


@pytest.fixture
def vcr_cassette_name(request):
    """Generate cassette names based on test module and function name.

    This creates organized cassette files like:
    - cassettes/legocom_integration_test/test_fetch_real_set_30708.yaml
    """
    # Get the test module name (without .py extension)
    module_name = request.module.__name__.split(".")[-1]

    # Get the test function name
    test_name = request.node.name

    # Return a path that groups cassettes by test module
    return f"{module_name}/{test_name}"


@pytest.fixture
def http_client(vcr):
    """Provide an HTTP client that uses VCR for recording/replaying.

    The vcr fixture is provided by pytest-recording and will automatically
    record HTTP interactions to cassettes and replay them on subsequent runs.
    """
    # When using VCR, we don't need a real client context manager
    # VCR intercepts at the socket level, so we can use a simple client
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        yield client
