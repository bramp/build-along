python_sources()

pex_binary(
    name="main",
    entry_point="build_a_long.downloader.main:main",
)

python_tests(
    name="tests",
    sources=["*_test.py", "!*_integration_test.py"],
)

python_tests(
    name="integration_tests",
    sources=["*_integration_test.py"],
    extra_env_vars=["ENABLE_INTEGRATION_TESTS"],
    dependencies=[
        "3rdparty:dev#pytest-recording",
        "3rdparty:dev#vcrpy",
    ],
)

python_test_utils(
    name="test_utils",
    sources=["conftest.py"],
)
