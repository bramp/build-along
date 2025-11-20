pex_binary(
    name="datamodel_codegen",
    entry_point="datamodel_codegen:main",
    dependencies=[
        "//3rdparty:dev#datamodel-code-generator",
    ],
    description="Generate Pydantic models from OpenAPI specs.",
)

# Shell script to generate Pydantic models from OpenAPI schema
shell_sources(name="scripts", sources=["generate_models.sh"])

# Generate Pydantic models from OpenAPI schema
# Usage: pants run src/build_a_long/schemas:generate_models
run_shell_command(
    name="generate_models",
    command="./generate_models.sh",
    execution_dependencies=[
        ":scripts",
        "//schemas:openapi",
        ":datamodel_codegen",
    ],
    runnable_dependencies=[":datamodel_codegen"],
)

# TODO add a adhoc_tool to run generate_models directly.

# Python source files
python_sources()
