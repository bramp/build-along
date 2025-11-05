python_sources(
    overrides={
        # Disable sandboxing so the script can write directly to the repository
        "generate_golden_files.py": {"run_goal_use_sandbox": False},
    },
)

python_tests(
    name="tests",
)
