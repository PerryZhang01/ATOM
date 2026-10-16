# SPDX-License-Identifier: MIT
"""`compass measure memory`: one engine start's memory readings, published as
its configuration's `memory_readings` entry.

    compass measure memory --gpu G --machine M --store S --out D -- <ATOM args>

ATOM's engine starts from ATOM's arguments (`EngineArgs`) on the GPUs `--gpu`
names, through `MemoryReadingModelRunner`, and stops once it has started: its
weights loaded, its warmup run a second time on a warm compile cache, its KV
cache sized and its graphs captured. `gpu.quiet` checks those GPUs before the
start and after it. Each rank writes its readings and the key it read off its
resolved config, and the ranks must agree on the key. The entry's key is that
one with the digest of the machine spec `--machine` names; its conditions are
that spec's device and the stack the start ran, ROCm, AITER, RCCL and torch
as run.json's identity states them, so a run whose spec pins another stack
refuses it. `non_torch` refuses a
spread across the ranks above `SPREAD_LIMIT`. One member per rank holds
`memory.ENTRY_FIELDS`, its `record` the digest of the engine's log.

``--out`` gets the engine's log, run.json's identity from the rank-0 engine
file, and ``memory_readings.json``, the entry's readable copy. ATOM's
``--load_dummy`` refuses until its readings are shown to match a real load's.
"""

import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from atom.compass import parity
from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Provenance,
    machine_spec_digest,
    machine_spec_key,
    read_back,
    roots_for,
)
from atom.compass.drive import serve
from atom.compass.drive.gpu import add_gpu_flag, quiet
from atom.compass.drive.identity import identity
from atom.compass.run import CLOCK_ENV, ENV, _write_whole, memory
from atom.compass.spec import MachineSpec, across_ranks
from atom.compass.spec.fields import PINNED

PROG = "compass measure memory"
#: Where each rank writes its readings: a directory this command names.
OUT_ENV = "ATOM_COMPASS_MEMORY_OUT"
RUNNER = "atom.compass.drive.memory_runner.MemoryReadingModelRunner"
#: ATOM's dummy weights flag; its parser reads `_` in a flag as `-`.
LOAD_DUMMY = "--load-dummy"
START_TIMEOUT_S = 3600
LOG, READINGS = "engine.log", "readings"


def start(argv: list[str]) -> None:
    """Start ATOM's engine from ATOM's arguments `argv` through the reading
    runner, and stop it once it has started."""
    from atom.model_engine.arg_utils import EngineArgs
    from atom.model_engine.llm_engine import LLMEngine
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = EngineArgs.from_cli_args(parser.parse_args(argv))
    kwargs = args._get_engine_kwargs()
    LLMEngine(args.model, runner_qualname=RUNNER, **kwargs).close()


def main(argv: list[str]) -> None:
    cut = argv.index("--") if "--" in argv else len(argv)
    args = _parser().parse_args(argv[:cut])
    atom_args = argv[cut + 1 :]
    for arg in atom_args:
        flag = arg.split("=", 1)[0].replace("_", "-")
        if len(flag) > 2 and flag.startswith("--") and LOAD_DUMMY.startswith(flag):
            raise ValueError(
                f"{arg} is among ATOM's arguments; dummy weights are not yet "
                "shown to read as a real load's do"
            )
    model = serve._model(atom_args)
    machine = MachineSpec.from_mapping(json.loads(Path(args.machine).read_text()))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} holds an earlier run's files; one run per directory")
    (out / READINGS).mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in (ENV, CLOCK_ENV, parity.ENV)}
    env[OUT_ENV] = str(out / READINGS)
    env["HIP_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))
    code = f"import sys; from {__name__} import start; start(sys.argv[1:])"
    with quiet(args.gpu, out), (out / LOG).open("w") as log:
        engine = subprocess.Popen(
            [sys.executable, "-c", code, *atom_args],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            exited = engine.wait(START_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            exited = f"no exit in {START_TIMEOUT_S} s"
        finally:
            # The engine's own processes are in its session; none outlives it.
            try:
                os.killpg(engine.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            engine.wait()
    if exited != 0:
        raise ValueError(f"the engine did not start ({exited}); see {out / LOG}")
    first = json.loads(min((out / READINGS).iterdir()).read_text())
    (out / "engine.json").write_text(json.dumps(first["engine"]))
    identified = identity(out / "engine.json", model, sys.argv, args.gpu)
    _write_whole(out / "run.json", json.dumps(identified))
    publish(out, machine, args.store, " ".join([PROG, *argv]), identified["stack"])


def publish(out: Path, machine, store: str, command: str, stack: dict) -> None:
    """Publish the readings the ranks wrote in `out` at their key on `machine`,
    under `stack`, the stack the start ran, as run.json's identity states it;
    and write the entry's readable copy."""
    ranks = {p.name: json.loads(p.read_text()) for p in (out / READINGS).iterdir()}
    record = "sha256:" + hashlib.sha256((out / LOG).read_bytes()).hexdigest()
    for held in ranks.values():
        held["readings"]["record"] = record
    keys = {json.dumps(r["key"], sort_keys=True) for r in ranks.values()}
    if len(keys) != 1:
        raise ValueError(
            f"the ranks read {len(keys)} keys off their configs: {sorted(keys)}"
        )
    first = ranks[min(ranks)]
    fields = first["key"]
    reserve = [
        r["readings"]["driver_and_collective_reserve_bytes"] for r in ranks.values()
    ]
    across_ranks("non_torch", len(reserve), reserve)
    key = Key.of(
        Kind.MEMORY_READINGS, machine_spec=machine_spec_digest(machine), **fields
    )
    members = {name: json.dumps(r["readings"]).encode() for name, r in ranks.items()}
    ArtifactStore(store).publish(
        key,
        provenance=Provenance(
            command,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            roots_for({"atom": "atom", "aiter": "aiter"}),
        ),
        topology=read_back(min(ranks))[1].topology,
        conditions=memory.conditions(
            fields,
            device=machine_spec_key(machine).values["device"],
            software_stack=" / ".join(
                f"{c} {stack[c]}" for c in PINNED if c != "torch"
            ),
            torch=stack["torch"],
        ),
        gates=GateState.of(),
        members=members,
    )
    copy = {"key": key.values, "token": key.token()}
    copy["members"] = {name: r["readings"] for name, r in sorted(ranks.items())}
    (out / "memory_readings.json").write_text(json.dumps(copy, indent=1) + "\n")
    print(f"published {key} as {key.token()}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    add_gpu_flag(p)
    p.add_argument("--machine", required=True, help="this platform's machine spec")
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="an empty directory for the logs")
    return p
