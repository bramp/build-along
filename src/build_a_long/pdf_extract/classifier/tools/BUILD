python_sources(
    overrides={
        # Disable sandboxing so the scripts can read/write directly in the repository.
        # Use: pants run src/build_a_long/pdf_extract/classifier/tools/generate_golden_files.py
        "generate_golden_files.py": {"run_goal_use_sandbox": False},
        "generate_golden_hints.py": {"run_goal_use_sandbox": False},
    },
)

shell_sources(
    name="shell-scripts",
)

python_tests(
    name="tests",
)
