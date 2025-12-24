python_sources(
    sources=["*.py", "!analyze_classifier_scores.py"],
    overrides={
        # Disable sandboxing so the scripts can read/write directly in the repository.
        # Use: pants run src/build_a_long/pdf_extract/classifier/tools/<script>.py
        "generate_golden_files.py": {"run_goal_use_sandbox": False},
        "generate_golden_hints.py": {"run_goal_use_sandbox": False},
        "generate_ocr_fixtures.py": {"run_goal_use_sandbox": False},
        "regenerate_fixtures.py": {"run_goal_use_sandbox": False},
        "tune_config.py": {"run_goal_use_sandbox": False},
    },
)

# Separate target for analyze_classifier_scores.py which needs fixture data
python_source(
    name="analyze_classifier_scores",
    source="analyze_classifier_scores.py",
    dependencies=[
        "//src/build_a_long/pdf_extract/fixtures:data",
    ],
)


python_tests(
    name="tests",
)
