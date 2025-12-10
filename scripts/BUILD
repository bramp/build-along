# No python_sources in this directory as it's not part of the pants build path.
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
