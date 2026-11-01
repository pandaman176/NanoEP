"""Check specialization keys, caching and build inputs without a CUDA device."""

from types import SimpleNamespace

import pytest
import torch.utils.cpp_extension

from nano_ep.backends._build import (
    prepare_combine_epilogue,
    prepare_combine_main,
    prepare_dispatch_epilogue,
)



@pytest.fixture
def builds(monkeypatch):
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name=kwargs["name"])

    prepare_combine_epilogue.cache_clear()
    prepare_combine_main.cache_clear()
    prepare_dispatch_epilogue.cache_clear()
    monkeypatch.setattr(torch.utils.cpp_extension, "load", load)
    yield calls
    prepare_combine_epilogue.cache_clear()
    prepare_combine_main.cache_clear()
    prepare_dispatch_epilogue.cache_clear()


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


@pytest.mark.parametrize("cached_mode", [False, True])
def test_dispatch_geometry_specializes_only_consumed_dimensions(builds, cached_mode):
    geometry = (256, 3, 5)
    first = prepare_dispatch_epilogue(*geometry, cached_mode=cached_mode)
    assert prepare_dispatch_epilogue(*geometry, cached_mode=cached_mode) is first
    for position, value in enumerate((512, 8, 40)):
        changed = list(geometry)
        changed[position] = value
        assert prepare_dispatch_epilogue(*changed, cached_mode=cached_mode) is not first
    assert len(builds) == 4
    assert len({entry["name"] for entry in builds}) == 4
    assert builds[0]["sources"][0].endswith("dispatch_epilogue.cu")
    assert builds[0]["extra_include_paths"][0].endswith("include")
    assert [arg for arg in builds[0]["extra_cuda_cflags"] if arg.startswith("-D")] == [
        "-DNANO_EP_HIDDEN=256",
        "-DNANO_EP_TOPK=3",
        "-DNANO_EP_EXPERTS_PER_RANK=5",
        f"-DNANO_EP_CACHED_MODE={int(cached_mode)}",
    ]
    assert first.name not in {
        prepare_combine_main(256, 3, 2, 5, 7).name,
        prepare_combine_epilogue(256, 3, 2, 5, 7).name,
    }


@pytest.mark.parametrize(
    "geometry",
    [
        (0, 3, 5),
        (128, 3, 5),
        (257, 3, 5),
        (2**31, 3, 5),
        (256, 0, 5),
        (256, 33, 5),
        (256, 3, 0),
    ],
)
def test_invalid_dispatch_geometry_never_compiles(builds, geometry):
    with pytest.raises(RuntimeError, match="invalid dispatch epilogue geometry"):
        prepare_dispatch_epilogue(*geometry, cached_mode=False)
    assert not builds


def test_dispatch_cache_modes_build_distinct_modules(builds):
    fresh = prepare_dispatch_epilogue(256, 3, 5, cached_mode=False)
    cached = prepare_dispatch_epilogue(256, 3, 5, cached_mode=True)
    assert fresh is not cached and fresh.name != cached.name
    assert prepare_dispatch_epilogue(256, 3, 5, cached_mode=False) is fresh
    assert prepare_dispatch_epilogue(256, 3, 5, cached_mode=True) is cached
    assert len(builds) == 2
    assert "-DNANO_EP_CACHED_MODE=0" in builds[0]["extra_cuda_cflags"]
    assert "-DNANO_EP_CACHED_MODE=1" in builds[1]["extra_cuda_cflags"]
