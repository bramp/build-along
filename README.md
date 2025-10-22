# Build-along Lego Instructions

This project is a collection of tools to download, analyze, and view Lego Instruction manuals.

## Components

1. **PDF Downloader**: A script to download Lego instruction PDFs based on a set number.
2. **Bounding Box Extractor**: A tool to analyze the PDFs and extract bounding boxes for instruction numbers, parts lists, and build steps.
3. **Instruction Viewer**: A Flutter mobile app to view the instructions with pan-and-zoom functionality.

## Getting Started

TODO Simple setup for users (not developers)

## Developer Setup

### Pants Setup

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
# List targets to confirm the binary exists
pants list src/build_a_long/downloader:

# Dry run: only list PDFs for set 75419
pants run src/build_a_long/downloader:downloader -- 75419 --dry-run

# Download PDFs to data/75419
pants run src/build_a_long/downloader:downloader -- 75419
```

Options:

- `--locale` (default `en-us`): Locale segment used by lego.com.
- `--out-dir`: Output directory (defaults to `data/<set>`).
- `--dry-run`: Only list found PDFs without downloading.
- `--force`: Re-download PDFs even if the file already exists.

Notes:

- Large PDFs may take time to download; files are saved under the `data/` folder (ignored by git).
- Sources are scraped from pages like `https://www.lego.com/en-us/service/building-instructions/<set>`.
- Existing files are skipped by default; use `--force` to overwrite. A simple progress indicator is shown per file.

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
