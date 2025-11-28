# python_sources()

shell_sources(
    name="shell-scripts",
)

run_shell_command(
    name="process-all-examples",
    command="scripts/process_all_examples.sh",
    runnable_dependencies=[
        "src/build_a_long/pdf_extract:main",
    ],
    workdir="/",
)
