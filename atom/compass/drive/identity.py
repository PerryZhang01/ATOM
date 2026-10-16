# SPDX-License-Identifier: MIT
"""run.json's identity: what a command records about the run it made, so that a
consumer can tell two runs' source, host, device and stack apart and refuse a
mix of them.

It holds no workload: no command starts the client, and which requests a server
served is read from its step record.
"""

import json
import os
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import torch

from atom.compass.artifacts.provenance import (
    git_described_root,
    git_tree_root,
    module_root,
)
from atom.compass.drive.gpu import GpuRefusal, smi
from atom.compass.run import host_key_of
from atom.compass.run.step_grid import BOUNDS

FORMAT_VERSION = 1
#: What names a GPU's model, from rocm-smi's product listing.
CARD_FIELDS = {"series": "Card Series", "sku": "Card SKU", "gfx": "GFX Version"}


def identity(
    engine_file: Path, model: str, argv: Sequence[str], gpus: tuple[int, ...] = ()
) -> dict:
    """The identity entries of run.json for a command line `argv` serving the
    model `model` names, on `gpus`, which is empty for a command that uses no
    GPU. `engine_file` is the `dp<r>.engine.json` the run's engine wrote, so a
    command calls this once its server stops."""
    engine = json.loads(Path(engine_file).read_text())
    # The file adds the scheduler bounds and the block count to `keyed`.
    keyed = {
        k: v for k, v in engine.items() if k not in (*BOUNDS, "num_kvcache_blocks")
    }
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / ".info" / "version"
    return {
        "format_version": FORMAT_VERSION,
        "argv": list(argv),
        "source": git_tree_root("atom", module_root("atom")).as_json(),
        "host_key": host_key_of(model, keyed),
        "device": _device(gpus),
        "stack": {
            "rocm": rocm.read_text().strip(),
            "torch": torch.__version__,
            "aiter": git_described_root("aiter", module_root("aiter")).revision,
        },
        "profiled": False,  # a command that profiles ops sets it
    }


def _device(gpus: tuple[int, ...]) -> list[dict]:
    """Each GPU's index and model as rocm-smi names it; a GPU it does not list
    refuses."""
    if not gpus:
        return []
    cards = json.loads(smi("--showproductname", "--json"))
    missing = [gpu for gpu in gpus if f"card{gpu}" not in cards]
    if missing:
        raise GpuRefusal(f"rocm-smi lists no GPU {missing}; it lists {list(cards)}")
    return [
        {"index": gpu, **{k: cards[f"card{gpu}"][f] for k, f in CARD_FIELDS.items()}}
        for gpu in gpus
    ]


@contextmanager
def phase(phases: dict, name: str):
    """Record the start and, once it ends, the end of the phase `name` in
    `phases`, each in wall and monotonic seconds."""
    edges = phases[name] = {"start": _now()}
    yield
    edges["end"] = _now()


def _now() -> dict:
    return {"wall": time.time(), "monotonic": time.monotonic()}
