python_sources()

python_test_utils(
    name="test-helpers",
    sources=["test_helpers.py"],
)

python_tests(
    name="tests",
    sources=["*_test.py"],
    dependencies=[":fixtures"],
)

files(
    name="fixtures",
    sources=["fixtures/*.json"],
)
