# SPDX-License-Identifier: MIT
"""A simulated run's meta build puts nothing on a device, a named device included.

`NonAllocatingRunner._build_and_load_model` builds the model class on the meta
device. A module may name a device for a tensor it makes, as ATOM's attention
names this rank's GPU for its KV scale; that tensor is built on meta too.

The server test runs the slice of `test_vertical_slice` beside the first GPU in
`HIP_VISIBLE_DEVICES` and reads that GPU's used VRAM with `rocm-smi`, so this
process never touches the device itself. It skips as the slice does.
"""

import json
import os
import subprocess
import threading

import torch
from test_runner_non_allocating import Runner
from test_vertical_slice import NEEDS_A_RUN, run_slice

GPU = (os.environ.get("HIP_VISIBLE_DEVICES") or "0").split(",")[0]
SLACK_BYTES = 8 * 2**20


class NamedDevices(torch.nn.Module):
    """A model that names a device for each tensor it makes."""

    def __init__(self, config):
        super().__init__()
        self.kv_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda:0")
        self.weight = torch.nn.Parameter(
            torch.empty(4, 4, device=torch.device("cuda", 0))
        )
        self.register_buffer("rope", torch.ones(4, device="cpu"))


def test_a_tensor_made_on_a_named_device_is_built_on_meta():
    runner = Runner()
    runner._build_and_load_model(NamedDevices)
    tree = runner.model.tree
    assert [t.device.type for t in (tree.kv_scale, tree.weight, tree.rope)] == [
        "meta"
    ] * 3


def _used_vram() -> int:
    out = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "-d", GPU, "--json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (card,) = json.loads(out).values()
    return int(card["VRAM Total Used Memory (B)"])


@NEEDS_A_RUN
def test_a_simulated_server_leaves_its_gpu_at_idle(tmp_path):
    idle = _used_vram()
    used, done = [], threading.Event()

    def sample():
        while not done.is_set():
            used.append(_used_vram())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        run_slice(tmp_path / "run", "1")
    finally:
        done.set()
        sampler.join()
    print(f"\nGPU {GPU}: idle {idle} B, peak {max(used)} B over {len(used)} reads")
    assert max(used) - idle <= SLACK_BYTES
