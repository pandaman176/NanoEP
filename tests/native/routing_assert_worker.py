"""Exercise one device assertion per process, then discard its CUDA context."""

import os
import sys

import torch

from nano_ep.backends._build import load_extension
from nano_ep.testing.native_dispatch import prepare_dispatch_metadata


def run(stage, kind):
    torch.cuda.set_device(0)
    native = load_extension()
    x = torch.ones((2, 4), device="cuda")
    valid = torch.tensor([[0, 1, -1], [2, -1, 0]], device="cuda")
    if stage != "route_counts":
        layout = native.window_layout(1, 2, 4, 3, 3)
        runtime = native.PeerRuntime(0, 1, layout[-1])
        window = runtime.local_window()
        window[layout[1] : layout[1] + 32].view(torch.int64).fill_(-1)
        torch.cuda.synchronize()
        runtime.import_peer_handles([runtime.export_local_handle()])

    def launch(indices):
        if stage == "route_counts":
            native.route_counts(indices, 3, 1)
        elif stage == "reserve_and_put":
            window[layout[1] + 32 : layout[1] + 40].view(torch.int64).zero_()
            native.reserve_and_put(runtime, x, indices, None, 3, 2)
        else:
            state = native.dispatch_main(runtime, x, indices, None, 3, 2, 4, [])
            if stage == "dispatch_epilogue":
                prepare_dispatch_metadata(native, state)
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
