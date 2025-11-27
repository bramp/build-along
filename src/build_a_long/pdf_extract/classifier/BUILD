python_sources()

python_test_utils(
    name="conftest",
    sources=["conftest.py"],
)

python_tests(
    name="tests",
    sources=["*_test.py"],
    dependencies=[
        "//src/build_a_long/pdf_extract:fixtures",
    ],
)
