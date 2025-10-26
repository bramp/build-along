python_sources()

python_tests(
    name="tests",
    sources=["*_test.py"],
    dependencies=[":fixture_real_example"],
)

file(
    name="fixture_real_example",
    source="fixtures/real_example_parts_list_and_deletions.json",
)
