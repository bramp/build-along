python_sources()

python_test_utils(
    name="test-helpers",
    sources=["test_helpers.py"],
)

python_tests(
    name="tests",
    sources=["*_test.py"],
    dependencies=[":fixtures", ":test-helpers"],
)

python_source(
    name="generate-golden-files",
    source="generate_golden_files.py",
    # Disable sandboxing so the script can write directly to the repository
    run_goal_use_sandbox=False,
)

files(
    name="fixtures",
    sources=["fixtures/*.json"],
)
