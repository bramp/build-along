# Shell script to generate Pydantic models from OpenAPI schema
shell_sources(name="scripts", sources=["generate_models.sh"])

pex_binary(
    name="codegen",
    entry_point="datamodel_codegen:main",
    dependencies=[
        "//3rdparty:dev#datamodel-code-generator",
    ],
    description="Generate Pydantic models from OpenAPI specs.",
)


# Generate Pydantic models from OpenAPI schema
# Usage: pants run src/build_a_long/schemas:codegen
run_shell_command(
    name="generate_models",
    command="./generate_models.sh",
    execution_dependencies=[
        ":scripts",
        "//schemas:openapi",
        "//tools/datamodel_codegen:codegen",
    ],
    runnable_dependencies=["//tools/datamodel_codegen:codegen"],
)

# TODO add a adhoc_tool to run generate_models directly.

# Python source files
python_sources()
