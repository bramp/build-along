python_test_utils(
    name="fixture_utils",
    sources=["fixture_utils.py"],
    dependencies=[
        "//src/build_a_long/pdf_extract/fixtures:data",
    ],
)

python_tests(
    name="tests",
    sources=["*_test.py", "!extractor_raw_fixture_test.py"],
)

python_tests(
    name="raw_fixture_tests",
    sources=["extractor_raw_fixture_test.py"],
    dependencies=[
        ":fixture_utils",
        "//src/build_a_long/pdf_extract/fixtures:data",
        "//data:test_pdfs",
    ],
)
