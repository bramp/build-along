python_sources()

python_tests(
    name="tests",
    sources=["*_test.py"],
)

pex_binary(
    name="main",
    entry_point="build_a_long.bounding_box_extractor.main:main",
)

pex_binary(
    name="analyze_classifier",
    entry_point="build_a_long.bounding_box_extractor.analyze_classifier:main",
)
