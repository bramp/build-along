# PDF files for testing.
# These are symlinked into the sandbox (not copied) for efficiency.
# PDFs are downloaded via the downloader tool and NOT committed to source control.
#
# The .placeholder/placeholder.pdf is a minimal valid PDF that's always committed.
# This ensures Pants doesn't fail when no real PDFs are present (fresh checkouts).
# Tests skip gracefully if specific PDFs aren't found at runtime.

files(
    name="test_pdfs",
    sources=[
        # Placeholder PDF - always exists, ensures target is valid even without real PDFs
        ".placeholder/placeholder.pdf",
        # PDFs referenced by fixtures/index.json5 for raw fixture tests
        # Using globs so we don't need to list each file explicitly
        "75375/*.pdf",  # Star Wars Millennium Falcon
        "40573/*.pdf",  # Ideas Christmas Tree
        "10237/*.pdf",  # Tower of Orthanc (3 manuals)
        "75406/*.pdf",  # Kylo Ren's Command Shuttle
    ],
)
