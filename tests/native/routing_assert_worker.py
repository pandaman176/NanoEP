"""Exercise one device assertion per process, then discard its CUDA context."""

import importlib.util
import os
import sys
from pathlib import Path

import torch

from nano_ep.backends._build import (
    load_extension,
    prepare_dispatch_epilogue,
    prepare_dispatch_main,
)
from nano_ep.testing.native_dispatch import prepare_dispatch_metadata


def run(stage, kind):
    torch.cuda.set_device(0)
    native = load_extension()
    if stage == "dispatch_epilogue":
        for cached_mode in (False, True):
            prepare_dispatch_epilogue(256, 3, 3, cached_mode=cached_mode)
    if stage == "dispatch_payload":
        binary = Path(os.environ["NANO_EP_PEER_PROBE_PATH"])
        spec = importlib.util.spec_from_file_location(binary.name.split(".")[0], binary)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
    x = torch.ones((2, 256), dtype=torch.bfloat16, device="cuda")
    valid = torch.tensor([[0, 1, -1], [2, -1, 0]], device="cuda")
    layout = native.window_layout(1, 2, 256, 3, 3)
    runtime = native.PeerRuntime(0, 1, layout[-1])
    window = runtime.local_window()
    window[layout[1] : layout[1] + 32].view(torch.int64).fill_(-1)
    torch.cuda.synchronize()
    runtime.import_peer_handles([runtime.export_local_handle()])

    def launch(indices):
        if stage == "dispatch_payload":
            window[layout[1] + 32 : layout[1] + 40].view(torch.int64).zero_()
            slots = torch.full((2, 1), -1, dtype=torch.int64, device="cuda")
            probe.run_dispatch_payload(
                runtime.peer_addresses(), x, indices, None, slots, 0, 3, 2, False
            )
        else:
            state = native.dispatch_main(
                runtime,
                x,
                indices,
                None,
                3,
                2,
                4,
                [],
                0,
                allow_multiple_reduction=False,
                launcher=prepare_dispatch_main(
                    x.shape[1], indices.shape[1], 1, 3, 2, 4
                ).launcher,
            )
            if stage == "dispatch_epilogue":
                prepare_dispatch_metadata(state)
        torch.cuda.synchronize()

    # Build/import/storage failures must not be mistaken for a routing assertion.
    launch(valid)
    invalid = valid.clone()
    invalid[0, 1] = {"negative": -2, "overflow": 3, "duplicate": 0}[kind]
    try:
        launch(invalid)
    except RuntimeError as error:
        message = str(error).lower()
        if "cuda" not in message or not any(
            text in message
            for text in (
                "illegal instruction",
                "device-side assert",
                "unspecified launch failure",
                "launch failed",
            )
        ):
            raise
        print(f"NANO_EP_ROUTING_ASSERT_OK={stage}:{kind}", flush=True)
        # Device traps invalidate the context; process teardown releases its allocations.
        os._exit(0)
    raise AssertionError(f"{stage} accepted invalid routing: {kind}")


if __name__ == "__main__":
    run(*sys.argv[1:])
