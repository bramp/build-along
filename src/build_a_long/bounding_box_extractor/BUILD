python_sources()

pex_binary(
    name="main",
    entry_point="build_a_long.bounding_box_extractor.main:main",
    dependencies=[
        "src/build_a_long/bounding_box_extractor",
    ],
)

python_tests(
    name="tests",
    sources=["*_test.py"],
)
