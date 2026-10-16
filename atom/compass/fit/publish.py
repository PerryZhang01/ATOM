# SPDX-License-Identifier: MIT
"""What a fit command shares: its record inputs, and publishing what it fitted.

A fit reads record directories, each with the client runs its server served,
by their exports (`--client-run RECORD EXPORT`, in the order they ran), and
loads their samples under one configuration (`samples.load`). A record no
`--client-run` names refuses: its steps would name no trace they served.
`--machine M` names the spec of the platform the records ran on: records
whose device, software stack or torch is not the one it states refuse, naming
each and both values. The configuration gains its `machine_spec_digest`, which
a fit's key and params state.

`publish` hands off one store entry whose members are `params`, what a run
file carries, and `report`, the fit's report, and writes both, with the
entry's key and token, as `<kind>.json` in `--out`. The entry's readings are
the ones its inputs were recorded under, read from their configuration; its
provenance names the command and the ATOM and aiter trees it ran from.
"""

import argparse
import datetime
import json
from pathlib import Path

from atom.compass.artifacts import (
    ArtifactStore,
    Conditions,
    GateState,
    Provenance,
    Reading,
    Topology,
    machine_spec_digest,
    machine_spec_key,
    member_name,
    roots_for,
)
from atom.compass.artifacts.keys import platform_key
from atom.compass.fit.samples import load
from atom.compass.run import _width_keys
from atom.compass.spec import MachineSpec

#: The rank every member is written at: a fit has one writer.
RANK = Topology().ranks()[0]


def parser(prog: str, description: str) -> argparse.ArgumentParser:
    """A fit command's parser: its records, their client runs, the store and `--out`."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("records", nargs="+", metavar="RECORD", help="record directories")
    p.add_argument(
        "--client-run",
        nargs=2,
        action="append",
        default=[],
        metavar=("RECORD", "EXPORT"),
        help="a client run RECORD's server served, by its export; every RECORD has one",
    )
    p.add_argument("--machine", required=True, help="the records' machine spec")
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="where the readable copy goes")
    return p


def inputs(args, sources) -> tuple[list, dict]:
    """The samples of `args`' records, each with its client runs, and their
    configuration; a fit taking only `sources` refuses any other input."""
    runs = {Path(d): [] for d in args.records}
    for record, export in args.client_run:
        if Path(record) not in runs:
            raise ValueError(f"--client-run names {record}, which is not a RECORD")
        runs[Path(record)].append(export)
    other = [str(d) for d in runs if not (d / "run.json").is_file()]
    if other:
        raise ValueError(
            f"{', '.join(other)} is not a record directory (no run.json), and "
            f"this fit takes {' and '.join(sources)} samples only"
        )
    unnamed = [str(d) for d, exports in runs.items() if not exports]
    if unnamed:
        raise ValueError(
            f"{', '.join(unnamed)} is named by no --client-run; name the client "
            "runs its server served, so the fit records the traces they sent"
        )
    samples, config = load(runs)
    spec = MachineSpec.from_mapping(
        _width_keys(json.loads(Path(args.machine).read_text()))
    )
    torch = spec.value("device.software_pinned_to.torch")
    ran, stated = platform(config), {**machine_spec_key(spec).values, "torch": torch}
    differ = [
        f"{axis} {ran[axis]!r} in the records and {stated[axis]!r} in {args.machine}"
        for axis in stated
        if ran[axis] != stated[axis]
    ]
    if differ:
        raise ValueError(
            "the records ran on another platform than --machine states: "
            + "; ".join(differ)
        )
    return samples, {**config, "machine_spec": machine_spec_digest(spec)}


def runs(samples) -> list[dict]:
    """Each run's provenance, in sample order: its first sample's, less the
    requests that step served, with the traces every sample of the run served."""
    held: dict[str, dict] = {}
    for s in samples:
        first = {k: v for k, v in s.provenance.items() if k != "requests"}
        run = held.setdefault(s.provenance["run"], {**first, "traces": set()})
        run["traces"] |= set(s.provenance["traces"])
    return [{**run, "traces": sorted(run["traces"])} for run in held.values()]


def platform(config) -> dict:
    """The device, software-stack and torch readings `config`, the inputs'
    configuration, states, the first two joined as a machine spec's key."""
    stack = config["stack"]
    return {**platform_key(config["device"], stack).values, "torch": stack["torch"]}


def conditions(config) -> Conditions:
    """The readings `config`, the inputs' configuration, states for every axis."""
    return Conditions.of(
        **platform(config),
        atom_src=Reading(*config["source_root"].split("-", 1)),
        model=config["model"],
        host_cpu=config["host_key"]["host_cpu"],
        engine_config=json.dumps(config["engine_config"], sort_keys=True),
    )


def publish(args, command: str, key, config, params, report):
    """Hand `params` and `report` off at `key` in `args.store`, and write their
    readable copy in `args.out`."""
    members = {"params": params, "report": report}
    ArtifactStore(args.store).publish(
        key,
        provenance=Provenance(
            command,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            roots_for({"atom": "atom", "aiter": "aiter"}),
        ),
        topology=Topology(),
        conditions=conditions(config),
        gates=GateState.of(),
        members={
            member_name(stem, RANK, "json"): json.dumps(body).encode()
            for stem, body in members.items()
        },
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    copy = {"key": key.values, "token": key.token(), **members}
    (out / f"{key.kind}.json").write_text(json.dumps(copy, indent=1) + "\n")
    print(f"published {key} as {key.token()}")
