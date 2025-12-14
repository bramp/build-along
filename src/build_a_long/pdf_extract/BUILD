python_sources(
    sources=["*.py", "!fixtures.py"],
)

python_test_utils(
    name="test_utils",
    sources=["fixtures.py"],
)

python_tests(
    name="tests",
    sources=["*_test.py"],
)

pex_binary(
    name="main",
    entry_point="build_a_long.pdf_extract.main:main",
)
