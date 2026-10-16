# SPDX-License-Identifier: MIT
"""`compass record`: one instance's real server, with its step record, serving
whatever client runs reach it until it is stopped.

    compass record --gpu G --out DIR -- <ATOM args>

`serve` starts the server on the GPUs `--gpu` names with the step record on,
and at its stop writes run.json's identity, unprofiled and with no workload.
Then each DP rank's record, ``steps/dp<r>.jsonl``, gets its cost lines beside
it (``dp<r>.cost.jsonl``, `step_labels.write_table`; a rank that ran only DP's
dummy forwards gets the version line alone), and run.json its
``engine`` entry: the engine's bounds as the record's engine files state them,
the fields `TARGET` names. DP ranks whose engine files state different bounds
refuse the entry, naming each. One record spans the server's lifetime, every
client run it served included; `parity.served` splits it per client run.

`--profile-ops` and `--profile-steps A:B`, the op-level half's profiled pass,
refuse; so does ATOM's ``--load_dummy``: a record times real weights, and MoE
routing follows them.
"""

import argparse
import json
from pathlib import Path

from atom.compass.backends.step_grid_plan import TARGET
from atom.compass.backends.step_labels import label, write_table
from atom.compass.cli import OPS
from atom.compass.drive import serve
from atom.compass.drive.gpu import add_gpu_flag
from atom.compass.run import _write_whole

PROG = "compass record"
#: ATOM's dummy weights flag; its parser reads `_` in a flag as `-`.
LOAD_DUMMY = "--load-dummy"


def main(argv: list[str]) -> None:
    cut = argv.index("--") if "--" in argv else len(argv)
    args = _parser().parse_args(argv[:cut])
    atom_args = argv[cut + 1 :]
    if args.profile_ops or args.profile_steps:
        raise ValueError(f"--profile-ops and --profile-steps profile ops, and {OPS}")
    for arg in atom_args:
        flag = arg.split("=", 1)[0].replace("_", "-")
        if len(flag) > 2 and flag.startswith("--") and LOAD_DUMMY.startswith(flag):
            raise ValueError(
                f"{arg} is among ATOM's arguments; a record times real weights, "
                "and MoE routing follows them"
            )
    serve.serve(atom_args, args.out, args.gpu)
    write_engine_and_costs(args.out)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    add_gpu_flag(p)
    p.add_argument(
        "--out", required=True, help="an empty directory for the record and logs"
    )
    p.add_argument("--profile-ops", action="store_true", help="refused: ops half")
    p.add_argument("--profile-steps", metavar="A:B", help="refused: ops half")
    return p


def write_engine_and_costs(out) -> None:
    """Write run.json's ``engine`` entry and each DP rank's cost lines into
    `out`, a stopped server's directory."""
    out = Path(out)
    records = sorted(
        (out / serve.STEPS).glob("dp*[0-9].jsonl"), key=lambda p: int(p.stem[2:])
    )
    engines = {}
    for record in records:
        engine = json.loads(record.with_suffix(".engine.json").read_text())
        engines[record.stem] = {k: engine[k] for k in TARGET}
    differ = [
        f"{k}: " + "; ".join(f"{json.dumps(e[k])} in {r}" for r, e in engines.items())
        for k in TARGET
        if len({json.dumps(e[k]) for e in engines.values()}) > 1
    ]
    if differ:
        raise ValueError(
            "the DP ranks resolved different engines, so run.json states none:\n"
            + "\n".join(differ)
        )
    run = json.loads((out / serve.RUN_JSON).read_text())
    run["engine"] = engines[records[0].stem]
    _write_whole(out / serve.RUN_JSON, json.dumps(run))
    for record in records:
        # A DP rank no request reached ran only the dummy forwards DP needs.
        lines = [json.loads(line) for line in record.read_text().splitlines()]
        idle = all(line["is_dummy_run"] for line in lines)
        write_table(record, [] if idle else label(record)[1])
