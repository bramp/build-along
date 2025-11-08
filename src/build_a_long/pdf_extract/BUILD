python_sources()

python_tests(
    name="tests",
    sources=["*_test.py"],
)

files(
    name="fixtures",
    sources=["fixtures/*.json", "fixtures/*.json.bz2"],
)

pex_binary(
    name="main",
    entry_point="build_a_long.pdf_extract.main:main",
)
