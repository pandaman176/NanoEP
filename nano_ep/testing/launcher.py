"""Bound a torchrun process tree and retain its output on success or failure."""

import os
import signal
import subprocess


def run_logged(command, *, log, cwd, env=None, timeout=300):
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Workers share the new process group, so a timeout releases all GPUs.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
