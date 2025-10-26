python_sources()

python_tests(
    name="tests",
    sources=["*_test.py"],
)

pex_binary(
    name="main",
    entry_point="build_a_long.pdf_extract.main:main",
)

pex_binary(
    name="analyze_classifier",
    entry_point="build_a_long.pdf_extract.analyze_classifier:main",
)
