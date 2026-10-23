"""CUDA tests require explicit opt-in and fail if prerequisites are absent."""

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("nanoep")
    group.addoption("--run-cuda", action="store_true", help="Run native CUDA tests")
    group.addoption("--run-distributed", action="store_true", help="Run torchrun integration tests")
    group.addoption("--run-acceptance", action="store_true", help="Run the larger example matrix")
    group.addoption("--nproc", type=int, choices=(1, 2, 4, 8), default=2, help="GPU process count")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("distributed") and not config.getoption("--run-distributed"):
            item.add_marker(pytest.mark.skip(reason="enable with --run-distributed --nproc N"))
        elif item.get_closest_marker("cuda") and not (config.getoption("--run-cuda") or config.getoption("--run-distributed")):
            item.add_marker(pytest.mark.skip(reason="enable with --run-cuda"))
