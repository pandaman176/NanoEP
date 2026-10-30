"""Check specialization keys, caching and build inputs without a CUDA device."""

from types import SimpleNamespace

import pytest
import torch.utils.cpp_extension

from nano_ep.backends._build import prepare_combine_epilogue, prepare_combine_main



@pytest.fixture
def builds(monkeypatch):
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name=kwargs["name"])

    prepare_combine_epilogue.cache_clear()
    prepare_combine_main.cache_clear()
    monkeypatch.setattr(torch.utils.cpp_extension, "load", load)
    yield calls
    prepare_combine_epilogue.cache_clear()
    prepare_combine_main.cache_clear()


@pytest.mark.parametrize(
    "factory,source",
    [
        (prepare_combine_epilogue, "combine_epilogue.cu"),
        (prepare_combine_main, "combine_main.cu"),
    ],
)
def test_every_geometry_dimension_selects_a_specialization(builds, factory, source):
    shape = (256, 3, 2, 3, 5)
    first = factory(*shape)
    assert factory(*shape) is first
    for position, value in enumerate((512, 8, 4, 8, 17)):
        changed = list(shape)
        changed[position] = value
        assert factory(*changed) is not first
    assert len(builds) == 6
    assert len({entry["name"] for entry in builds}) == 6
    assert builds[0]["sources"][0].endswith(source)
    assert "-DNANO_EP_HIDDEN=256" in builds[0]["extra_cuda_cflags"]
    assert "-DNANO_EP_MAX_TOKENS=5" in builds[0]["extra_cuda_cflags"]
    assert not any(
        "PARTITIONS" in arg or "ROW_BYTES" in arg
        for arg in builds[0]["extra_cuda_cflags"]
    )


@pytest.mark.parametrize("factory", [prepare_combine_epilogue, prepare_combine_main])
@pytest.mark.parametrize("hidden", [0, 1, 128, 255, 257])
def test_invalid_hidden_never_starts_compilation(builds, hidden, factory):
    with pytest.raises(RuntimeError, match="multiple of 256"):
        factory(hidden, 3, 2, 3, 5)
    assert not builds


def test_main_and_epilogue_use_distinct_build_modules(builds):
    geometry = (256, 3, 2, 3, 5)
    main = prepare_combine_main(*geometry)
    epilogue = prepare_combine_epilogue(*geometry)
    assert main.name != epilogue.name
    assert builds[0]["extra_include_paths"][0].endswith("include")
