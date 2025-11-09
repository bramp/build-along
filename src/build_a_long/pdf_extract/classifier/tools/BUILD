python_sources(
    overrides={
        # Disable sandboxing so the script can write directly to the repository
        "generate_golden_files.py": {"run_goal_use_sandbox": False},
        "generate_golden_font_hints.py": {"run_goal_use_sandbox": False},
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
    name="generate-golden-font-hints",
    entry_point="build_a_long.pdf_extract.classifier.tools.generate_golden_font_hints:main",
)
