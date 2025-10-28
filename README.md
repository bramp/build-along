# Build-along Lego Instructions

This project is a collection of tools to download, analyze, and view Lego Instruction manuals.

## Components

1. **PDF Downloader**: A script to download Lego instruction PDFs based on a set number.
2. **Bounding Box Extractor**: A tool to analyze the PDFs and extract bounding boxes for instruction numbers, parts lists, and build steps.
3. **Instruction Viewer**: A Flutter mobile app to view the instructions with pan-and-zoom functionality.

## Getting Started

TODO Simple setup for users (not developers)

## Developer Setup

### Install dependencies

Pants requires a `pants.lock` file to manage Python dependencies. To generate it for the first time or after updating `requirements.txt`, run:

```bash
pants generate-lockfiles
```

This will create a `pants.lock` file in the root of the project.

## Code Style and Linting

This project uses `pants` to run `ruff` for linting and formatting, configured
in the root `pyproject.toml`. `pre-commit` hooks also enforce code style by
running `pants` commands.

```shell
pants fix ::
pants check ::
```

### VSCode Configuration

To enable VSCode to find the Python interpreter and libraries managed by Pants, generate a `.env` file:

```bash
ROOTS=$(pants roots)
python3 -c "print('PYTHONPATH=./' + ':./'.join('''${ROOTS}'''.replace(' ', '\\\\ ').split('\n')) + ':\$PYTHONPATH')" > .env
# TODO also add dist/codegen to the path.

# Setup a symlinked immutable virtualenv (so the IDE can find 3rd party dependencies)
# See https://www.pantsbuild.org/dev/docs/using-pants/setting-up-an-ide#python-third-party-dependencies-and-tools
pants export \
    --py-resolve-format=symlinked_immutable_virtualenv \
    --resolve=python-default
```

### Pre-commit Hooks

To install the pre-commit hooks, run:

```shell
pip install pre-commit
pre-commit install
```

And to run them

```shell
pre-commit run --all-files
```

### Downloader CLI

Download LEGO building instruction PDFs for a given set number by scraping the official instructions page.

Run with Pants (recommended):

```bash
# Fetch metadata only (no downloads) for set 75419
pants run src/build_a_long/downloader:main -- 75419 --metadata

# Download PDFs to data/75419
pants run src/build_a_long/downloader:main -- 75419

# Pipe a list of set numbers into the downloader
echo -e "75419\n75159\n" | pants run src/build_a_long/downloader:main -- --stdin

# Fetch metadata for multiple sets from stdin
cat sets.txt | pants run src/build_a_long/downloader:main -- --stdin --metadata
```

Options:

- `--locale` (default `en-us`): Locale segment used by lego.com.
- `--out-dir`: Output directory (defaults to `data/<set>`).
- `--metadata`: Only fetch and print metadata as JSON (no downloads).
- `--force`: Re-download PDFs even if the file already exists.

Notes:

- Large PDFs may take time to download; files are saved under the `data/` folder (ignored by git).
- Sources are scraped from pages like `https://www.lego.com/en-us/service/building-instructions/<set>`.
- Existing files are skipped by default; use `--force` to overwrite. A simple progress indicator is shown per file.
- Metadata includes: set number, title, age, pieces, year, and ordered list of instruction PDF URLs.

### Bounding Box Extractor CLI

Analyze a LEGO instruction PDF and extract bounding boxes with hierarchical structure for debugging and classifier development.

Run with Pants:

```bash
# Process all pages (outputs to same directory as PDF)
pants run src/build_a_long/pdf_extract:main -- path/to/manual.pdf

# Process a single page
pants run src/build_a_long/pdf_extract:main -- \
  path/to/manual.pdf \
  --pages 1

# Process a range of pages (e.g., pages 5-10)
pants run src/build_a_long/pdf_extract:main -- \
  path/to/manual.pdf \
  --pages 5-10

# Process from page 10 to end
pants run src/build_a_long/pdf_extract:main -- \
  path/to/manual.pdf \
  --pages 10-

# Process from start to page 5
pants run src/build_a_long/pdf_extract:main -- \
  path/to/manual.pdf \
  --pages -5

# Specify custom output directory
pants run src/build_a_long/pdf_extract:main -- \
  path/to/manual.pdf \
  --pages 5-10 \
  --output-dir output/debug
```

Options:

- `--output-dir`: Directory to save extracted images and JSON files. Defaults to same directory as the PDF.
- `--pages`: Page range to process (1-indexed). Supports:
  - Single page: `"5"`
  - Explicit range: `"5-10"`
  - Open-ended from page: `"10-"` (from page 10 to end)
  - Open-ended to page: `"-5"` (from start to page 5)
  - Defaults to all pages if not specified

Output:

- **Images**: One PNG per page (`page_001.png`, `page_002.png`, etc.) with bounding boxes drawn in red (step numbers) or blue (other elements).
- **JSON**: Single file named `page_001-003.json` (for pages 1-3) containing:
  - Flat list of typed page elements (StepNumber, Drawing, Unknown)
  - Hierarchical tree based on bounding box containment
  - Full element metadata serialized via dataclasses

Notes:

- Requires `PyMuPDF` and `Pillow`.
- Page numbers are 1-indexed to match PDF viewers.
- JSON schema includes `__type__` discriminators for each element to enable deserialization.
- Example JSON structure:
  
    ```json
    {
        "pages": [
            {
                "page_number": 1,
                "elements": [
                    {
                        "bbox": {"x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 40.0},
                        "value": 1,
                        "__type__": "StepNumber"
                    },
                    {
                        "bbox": {"x0": 50.0, "y0": 60.0, "x1": 150.0, "y1": 200.0},
                        "image_id": "image_0",
                        "__type__": "Drawing"
                    }
                ],
                "hierarchy": [
                    {
                        "element": {...},
                        "children": [...]
                    }
                ]
            }
        ]
    }
    ```

### Running Tests

Run all tests with Pants:

```bash
pants test ::
```

or print detailed output:

```bash
pants test --output=all src/build_a_long/...
```

or with debug logging:

```bash
pants test --output=all src/build_a_long/... --log-cli-level=DEBUG
```

#### Integration Tests and HTTP Caching

Integration tests make real HTTP requests to LEGO.com, but are skipped by default.
They run in CI and can be enabled locally by setting an environment variable:

```bash
ENABLE_INTEGRATION_TESTS=true pants test src/build_a_long/downloader::
```

These tests use VCR.py (via pytest-recording) to record and replay HTTP
interactions, so we avoid hammering LEGO.com and the tests run fast. Cassettes are
stored outside the repo in your user cache directory (by default):

- macOS/Linux: "$XDG_CACHE_HOME/build-along/cassettes/downloader" or "~/.cache/build-along/cassettes/downloader"

You can customize behavior with environment variables:

- `CASSETTE_MAX_AGE_DAYS` (default: 14): If a cassette is older than this, it will be
  deleted before the test runs so it can be re-recorded.
- `CASSETTE_DIR`: Override the cassette directory used for recording/replay.

To force refresh all cassettes in one go:

```bash
pants test --pytest-args="--record-mode=rewrite" src/build_a_long/downloader::
```

In CI, you can cache the cassette directory between runs to avoid re-recording. For
GitHub Actions, cache the path printed above (e.g., `~/.cache/build-along/cassettes`).

#### Integration Tests

Some tests make real HTTP requests to external services (e.g., LEGO.com) and are skipped by default to avoid hammering their servers during local development. These integration tests run automatically in CI.

To run integration tests locally:

```bash
ENABLE_INTEGRATION_TESTS=true pants test ::
```

Or for a specific test file:

```bash
ENABLE_INTEGRATION_TESTS=true pants test src/build_a_long/downloader:legocom_integration_test
```

HTTP caching (VCR):

- Integration tests record HTTP interactions once and then replay them using VCR.py (via pytest-recording).
- Cassettes are stored in a persistent user cache directory by default:
  - macOS/Linux: `~/.cache/build-along/cassettes/downloader`
  - Or if `XDG_CACHE_HOME` is set: `$XDG_CACHE_HOME/build-along/cassettes/downloader`
- Cassettes older than 14 days are automatically deleted before the test runs and will be re-recorded.
- To force-refresh all cassettes:
  - Delete the cache directory, or
  - Run: `pants test --pytest-args="--record-mode=rewrite" ::`

Note: We do not commit cassettes to the repo. In CI, you can add a cache step to persist the cassette directory between runs for faster tests and fewer network calls.

## Contributing

Contributions are welcome! Please refer to the `TODO.md` file for a list of
tasks and features to work on.

## License

```text
Copyright 2025 Andrew Brampton (bramp.net)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
