# Python source files
python_sources()

# Generate JSON Schema from Pydantic models
# Usage:
#   pants run src/build_a_long/schemas:generate_schema -- manual schemas/lego_manual.schema.yaml
#   pants run src/build_a_long/schemas:generate_schema -- metadata schemas/lego_metadata.schema.yaml
pex_binary(
    name="generate_schema",
    entry_point="generate_schema.py",
)
