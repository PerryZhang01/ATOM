# SPDX-License-Identifier: MIT
"""A record's steps per client run, with the traces and request parameters each sent.

A server serves any number of client runs, one client invocation each, and its
step record holds them all. A client run is stated by its export, a JSON
document ``{"requests": {NAME: {"trace": TRACE, "sampling": {...}}}}`` naming
each request it sent by its response ``id``, with the trace it came from and
its request parameters; or by that ``requests`` mapping itself. A request the
record names with no response id (`read`) is named by its key instead.

`served` splits a record by them: a step belongs to the client run whose
requests it holds, a step holding none to no client run, and a step holding two
client runs' requests refuses the split, and so does a client run whose
requests no step holds.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from atom.compass.parity import read


def requests(client_run) -> dict[str, dict]:
    """A client run's requests by name: from its export at a path, or the mapping given."""
    if isinstance(client_run, Mapping):
        return dict(client_run)
    return json.loads(Path(client_run).read_text())["requests"]


def served(directory, client_runs: Sequence = ()) -> dict[int, list[dict]]:
    """The record at `directory` as `read` gives it, each step with `client_run`,
    the index in `client_runs` of the client run whose requests it holds, or
    None; `traces`, the traces those requests came from; and `sampling`, their
    request parameters, each set once, as sorted JSON."""
    runs = [requests(c) for c in client_runs]
    owner: dict[str, int] = {}
    for k, run in enumerate(runs):
        for name in run:
            if owner.setdefault(name, k) != k:
                raise ValueError(
                    f"request {name} is in client runs {owner[name]} and {k}; "
                    "a request is sent by one client run"
                )
    record = read(directory)
    for rank, steps in record.items():
        ids = {k: v for s in steps for k, v in s.get("response_ids", {}).items()}
        for step in steps:
            names = [ids.get(r[0], r[0]) for r in step["batch"]]
            names = [n for n in names if n in owner]
            ks = sorted({owner[n] for n in names})
            if len(ks) > 1:
                raise ValueError(
                    f"DP rank {rank} step {step['step']} holds requests of client "
                    f"runs {ks}; a step belongs to one client run"
                )
            sent = [runs[owner[n]][n] for n in names]
            step["client_run"] = ks[0] if ks else None
            step["traces"] = sorted({r["trace"] for r in sent})
            step["sampling"] = sorted(
                {json.dumps(r["sampling"], sort_keys=True) for r in sent}
            )
    owning = {s["client_run"] for steps in record.values() for s in steps}
    idle = [k for k in range(len(runs)) if k not in owning]
    if idle:
        raise ValueError(f"client runs {idle} name no request this record's steps hold")
    return record
