# SPDX-License-Identifier: MIT
"""run.json's identity: what a command records about the run it made, so that a
consumer can tell two runs' source, host, device and stack apart and refuse a
mix of them.

The device and the stack are recorded as a machine spec states them: each GPU's
name and arch, and ROCm's version, aiter's commit, RCCL's version and torch's.

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

from atom.compass.artifacts.provenance import _run, git_tree_root, module_root
from atom.compass.drive.gpu import GpuRefusal, smi
from atom.compass.run import host_key_of
from atom.compass.run.step_grid import keyed_of

FORMAT_VERSION = 2
#: rocm-smi's series is this brand and then the model a machine spec names.
BRAND = "AMD Instinct "
#: The model a machine spec names for each PCI device id (rocm-smi's model),
#: for a card whose series is not the brand and a model.
MODELS = {"0x75a3": "MI355X"}
#: The hex digits of aiter's commit a machine spec pins; git's own default
#: follows the checkout's object count, and a shallow clone reads fewer.
AITER_ABBREV = 9


def identity(
    engine_file: Path, model: str, argv: Sequence[str], gpus: tuple[int, ...] = ()
) -> dict:
    """The identity entries of run.json for a command line `argv` serving the
    model `model` names, on `gpus`, which is empty for a command that uses no
    GPU. `engine_file` is the `dp<r>.engine.json` the run's engine wrote, so a
    command calls this once its server stops."""
    engine = json.loads(Path(engine_file).read_text())
    keyed = keyed_of(engine)
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / ".info" / "version"
    return {
        "format_version": FORMAT_VERSION,
        "argv": list(argv),
        "source": git_tree_root("atom", module_root("atom")).as_json(),
        "host_key": host_key_of(model, keyed),
        "device": _device(gpus),
        "stack": {
            "rocm": rocm.read_text().strip(),
            "aiter": _commit(module_root("aiter")),
            "rccl": ".".join(map(str, torch.cuda.nccl.version())),
            "torch": torch.__version__,
        },
        "profiled": False,  # a command that profiles ops sets it
    }


def _device(gpus: tuple[int, ...]) -> list[dict]:
    """Each GPU's index, and its name and arch as a machine spec states them,
    from rocm-smi's series and gfx version, or `MODELS` when the series is not
    the brand and a model; a GPU it does not list or a model `MODELS` lacks
    refuses."""
    if not gpus:
        return []
    cards = json.loads(smi("--showproductname", "--json"))
    missing = [gpu for gpu in gpus if f"card{gpu}" not in cards]
    if missing:
        raise GpuRefusal(f"rocm-smi lists no GPU {missing}; it lists {list(cards)}")
    return [
        {
            "index": gpu,
            "name": _model(gpu, cards[f"card{gpu}"]),
            "arch": cards[f"card{gpu}"]["GFX Version"],
        }
        for gpu in gpus
    ]


def _model(gpu: int, card: dict) -> str:
    """The model a machine spec names GPU `gpu` by, from its rocm-smi `card`."""
    series, model = card["Card Series"], card.get("Card Model")
    if series.startswith(BRAND):
        return series.removeprefix(BRAND)
    if model in MODELS:
        return MODELS[model]
    raise GpuRefusal(
        f"GPU {gpu} is {series} with model {model}, which no machine spec names"
    )


def _commit(root: str) -> str:
    """The commit of the checkout at `root` in `AITER_ABBREV` hex digits, and
    `-dirty` when a tracked file differs: with every tag excluded, `git
    describe` names the commit itself. A root git cannot name refuses."""
    argv = ["git", "-C", root, "describe", "--always", "--dirty", "--exclude=*"]
    code, commit, error = _run([*argv, f"--abbrev={AITER_ABBREV}"])
    if code or not commit:
        raise ValueError(f"git names no commit at {root}: {error or 'no output'}")
    return commit


@contextmanager
def phase(phases: dict, name: str):
    """Record the start and, once it ends, the end of the phase `name` in
    `phases`, each in wall and monotonic seconds."""
    edges = phases[name] = {"start": _now()}
    yield
    edges["end"] = _now()


def _now() -> dict:
    return {"wall": time.time(), "monotonic": time.monotonic()}
