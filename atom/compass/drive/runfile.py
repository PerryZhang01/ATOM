# SPDX-License-Identifier: MIT
"""The run file a simulated run hands ATOM, built from its command's flags.

`add_arguments` adds the flags `trace steps` and `run` share, and `build`
resolves them, store keys included, into the run file's mapping; the command
then sets ``out_dir`` and ``clock_endpoint`` and hands it to the server.

- The backend, exactly one, each flag named by its `run` run-file key:
  ``--step-stub FILE``, the stated `Coefficients`; ``--step-linear KEY``, a
  `step_linear_law` entry's params with its key's ``engine_config``;
  ``--step-grid KEY``, the store's root and a `step_grid_bundle` entry's key.
  A KEY is the token `Key.token` prints, resolved under ``--store``.
  ``backend`` holds the backend's ``digest`` and, for a store entry, its
  ``provenance``, with its report's as ``inputs``; a stub's is None.
- ``--host-laws KEY``, a `host_laws` entry's params, as ``host``; without it
  the run charges no host time.
- ``--machine``, the spec; ``--store``'s root as ``store``; ``--on-refusal``;
  ``--step-roofline``, which reads ``parameter_count`` from the model
  directory's safetensors index and refuses when it states none.
- Not flags: `FIXED`, and ``data_parallel_size``, read from ATOM's arguments.
  The server serves client runs until it is stopped.
"""

import hashlib
import json
from pathlib import Path

from atom.compass import run
from atom.compass.artifacts import ArtifactStore, Kind, member_name
from atom.compass.fit.publish import RANK
from atom.compass.spec import MachineSpec

#: Each backend flag's store kind; the step stub's coefficients are a file.
KINDS = {run.STEP_LINEAR: Kind.STEP_LINEAR_LAW, run.STEP_GRID: Kind.STEP_GRID_BUNDLE}
#: The API server's admission path, the lookaheads no measurement backs
#: (`single_engine_table`), and a simulated-time bound past any run's.
FIXED = {
    "admission_path": "serving",
    "ipc_s": 2.0**-14,
    "stream_s": 2.0**-12,
    "bound_s": 1e7,
}
INDEX = "model.safetensors.index.json"


def add_arguments(parser) -> None:
    """Add the run file's flags to `parser`."""
    backend = parser.add_mutually_exclusive_group(required=True)
    for key in (run.STEP_STUB, *KINDS):
        flag, stub = "--" + key.replace("_", "-"), key == run.STEP_STUB
        backend.add_argument(flag, dest=key, metavar="FILE" if stub else "KEY")
    parser.add_argument("--host-laws", metavar="KEY", help="a host_laws token")
    parser.add_argument("--machine", required=True, help="the machine spec")
    parser.add_argument("--store", required=True, help="the artifact store's root")
    parser.add_argument("--on-refusal", choices=("mark", "abort"), default="mark")
    parser.add_argument("--step-roofline", action="store_true")


def build(args, atom_args) -> dict:
    """The run file of `args`, parsed with `add_arguments`, for a server
    started with ATOM's arguments `atom_args`."""
    store = ArtifactStore(Path(args.store).resolve())
    engine = _engine_args(atom_args)
    # Under DP-attention every TP rank is a DP rank, as `run.frontend` counts.
    ranks = engine.tensor_parallel_size if engine.enable_dp_attention else 1
    machine = json.loads(Path(args.machine).read_text())
    # Read as `compass fit` and the runner read it, so a spec they would
    # refuse refuses before a server starts.
    MachineSpec.from_mapping(run._width_keys(machine))
    out = {
        **FIXED,
        "client_runs": None,
        "data_parallel_size": engine.data_parallel_size * ranks,
        "machine": machine,
        "store": str(store.root),
        "on_refusal": args.on_refusal,
        "step_roofline": args.step_roofline,
    }
    if stub := getattr(args, run.STEP_STUB):
        out[run.STEP_STUB] = json.loads(Path(stub).read_text())
        stated = json.dumps(out[run.STEP_STUB], sort_keys=True).encode()
        out["backend"] = {
            "digest": "sha256:" + hashlib.sha256(stated).hexdigest(),
            "provenance": None,
        }
    else:
        name = next(k for k in KINDS if getattr(args, k))
        key = store.key_from_token(KINDS[name], getattr(args, name))
        entry = store.read(key)
        if name == run.STEP_GRID:
            out[name] = {"store": str(store.root), **key.values}
        else:
            engine_config = json.loads(key.values["engine_config"])
            out[name] = {**_member(entry, "params"), "engine_config": engine_config}
        reported = member_name("report", RANK, "json") in entry.members
        report = _member(entry, "report") if reported else {}
        out["backend"] = {
            "digest": entry.digest,
            "provenance": {
                **entry.provenance.as_json(),
                "inputs": report.get("provenance"),
            },
        }
    if args.host_laws:
        host = store.read(store.key_from_token(Kind.HOST_LAWS, args.host_laws))
        out["host"] = _member(host, "params")
    if args.step_roofline:
        out["parameter_count"] = _parameter_count(engine.model)
    return out


def _member(entry, stem: str):
    """`entry`'s `stem` member, as `fit` publishes one, read as JSON."""
    return json.loads(entry.read_member(stem, RANK, "json"))


def _engine_args(atom_args):
    """ATOM's arguments as its own parser reads them."""
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = FlexibleArgumentParser(add_help=False)
    EngineArgs.add_cli_args(parser)
    return parser.parse_known_args(atom_args)[0]


def _parameter_count(model: str) -> int:
    """The ``total_parameters`` `model`'s safetensors index states, resolved
    as loading resolves it."""
    from transformers.utils import cached_file

    try:
        path = cached_file(model, INDEX, local_files_only=True)
        index = json.loads(Path(path).read_text())
    except OSError:
        index = {}
    count = index.get("metadata", {}).get("total_parameters")
    if type(count) is not int:
        raise ValueError(
            f"the step roofline reads the model's parameter count, and {model} "
            f"states none: its {INDEX} has no metadata.total_parameters"
        )
    return count
