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
from atom.compass.run import host_key

FORMAT_VERSION = 1
#: The phases whose boundaries a run records; ops are profiled only in "profiling".
PHASES = ("warm-up", "profiling")
#: What names a GPU's model, from rocm-smi's product listing.
CARD_FIELDS = {"series": "Card Series", "sku": "Card SKU", "gfx": "GFX Version"}


def identity(config, argv: Sequence[str], gpus: tuple[int, ...] = ()) -> dict:
    """The identity entries of run.json for a command line `argv` run under
    ATOM's resolved `config` on `gpus`, which is empty for a command that uses
    no GPU."""
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / ".info" / "version"
    return {
        "format_version": FORMAT_VERSION,
        "argv": list(argv),
        "source": git_tree_root("atom", module_root("atom")).as_json(),
        "host_key": host_key(config),
        "device": _device(gpus),
        "stack": {
            "rocm": rocm.read_text().strip(),
            "torch": torch.__version__,
            "aiter": git_described_root("aiter", module_root("aiter")).revision,
        },
        "profiled": False,
        "phases": {},
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
def phase(run: dict, name: str):
    """Record the start and the end of the phase `name` in `run`, each in wall
    and monotonic seconds. A run that enters its profiling phase is profiled."""
    if name not in PHASES:
        raise ValueError(f"{name!r} is not a phase; the phases are {PHASES}")
    run["profiled"] = run["profiled"] or name == "profiling"
    edges = run["phases"][name] = {"start": _now()}
    yield
    edges["end"] = _now()


def _now() -> dict:
    return {"wall": time.time(), "monotonic": time.monotonic()}
