python_sources(
    overrides={
        # Disable sandboxing so the script can write directly to the repository
        "generate_golden_files.py": {"run_goal_use_sandbox": False},
        "generate_font_hints_golden.py": {"run_goal_use_sandbox": False},
    },
)

python_tests(
    name="tests",
)

pex_binary(
    name="generate-golden-files",
    entry_point="build_a_long.pdf_extract.classifier.tools.generate_golden_files:main",
)

pex_binary(
    name="generate-font-hints-golden",
    entry_point="build_a_long.pdf_extract.classifier.tools.generate_font_hints_golden:main",
)
