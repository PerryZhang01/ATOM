# SPDX-License-Identifier: MIT
"""What a fit command shares: its inputs, and publishing what it fitted.

A fit reads record directories, each with the client runs its server served,
by their exports (`--client-run RECORD EXPORT`, in the order they ran), and a
fit taking replayed batches reads step measurements too, each named by the
token `Key.token` prints; it loads their samples under one configuration
(`samples.load`). A record no `--client-run` names refuses: its steps would
name no trace they served. `--machine M` names the spec of the platform the
inputs ran on; the configuration gains its `machine_spec_digest`, which a
fit's key and params state, and `readings`, those a run on the spec takes with
the inputs' source root, model and engine, under which each measurement is
loaded.

`publish` hands off one store entry whose members are `params`, what a run
file carries, and `report`, the fit's report, under the readings it is
given, and writes both, with the entry's key and token, as `<kind>.json` in
`--out`; its provenance names the command and the ATOM and aiter trees it ran
from.
"""

import argparse
import datetime
import json
from pathlib import Path

from atom.compass.artifacts import (
    ArtifactStore,
    Conditions,
    GateState,
    Kind,
    Provenance,
    Reading,
    Topology,
    machine_spec_digest,
    member_name,
    roots_for,
)
from atom.compass.fit.samples import load
from atom.compass.run import _width_keys
from atom.compass.run.step_grid import readings as run_readings
from atom.compass.spec import MachineSpec

#: The rank every member is written at: a fit has one writer.
RANK = Topology().ranks()[0]


def parser(prog: str, description: str) -> argparse.ArgumentParser:
    """A fit command's parser: its inputs, their client runs, the store and `--out`."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="record directories, and step_measurement tokens where the fit takes them",
    )
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
    """The samples of `args`' inputs, records each with its client runs and,
    where `sources` holds `replay`, step measurements, and their configuration;
    a fit taking only `sources` refuses any other input."""
    runs = {Path(i): [] for i in args.inputs if (Path(i) / "run.json").is_file()}
    tokens = [i for i in args.inputs if Path(i) not in runs]
    if tokens and "replay" not in sources:
        raise ValueError(
            f"{', '.join(tokens)} is not a record directory (no run.json), and "
            f"this fit takes {' and '.join(sources)} samples only"
        )
    for record, export in args.client_run:
        if Path(record) not in runs:
            raise ValueError(f"--client-run names {record}, which is not a RECORD")
        runs[Path(record)].append(export)
    unnamed = [str(d) for d, exports in runs.items() if not exports]
    if unnamed:
        raise ValueError(
            f"{', '.join(unnamed)} is named by no --client-run; name the client "
            "runs its server served, so the fit records the traces they sent"
        )
    store = ArtifactStore(args.store)
    measurements = [store.key_from_token(Kind.STEP_MEASUREMENT, t) for t in tokens]
    samples, config = load(runs, measurements, args.store)
    spec = MachineSpec.from_mapping(
        _width_keys(json.loads(Path(args.machine).read_text()))
    )
    here = spec_readings(spec, config)
    for key in measurements:
        store.load(key, conditions=here, gates=GateState.of())
    return samples, {
        **config,
        "machine_spec": machine_spec_digest(spec),
        "readings": here,
    }


def runs(samples) -> list[dict]:
    """Each run's provenance, in sample order: its first sample's, less the
    requests that step served, with the traces every sample of the run served."""
    held: dict[str, dict] = {}
    for s in samples:
        first = {k: v for k, v in s.provenance.items() if k != "requests"}
        run = held.setdefault(s.provenance["run"], {**first, "traces": set()})
        run["traces"] |= set(s.provenance["traces"])
    return [{**run, "traces": sorted(run["traces"])} for run in held.values()]


def conditions(config) -> Conditions:
    """The readings `config`, the inputs' configuration, states for every axis."""
    stack = dict(config["stack"])
    torch = stack.pop("torch")
    return Conditions.of(
        software_stack=" / ".join(f"{c} {v}" for c, v in stack.items()),
        torch=torch,
        atom_src=Reading(*config["source_root"].split("-", 1)),
        model=config["model"],
        device=config["device"],
        host_cpu=config["host_key"]["host_cpu"],
        engine_config=json.dumps(config["engine_config"], sort_keys=True),
    )


def spec_readings(spec, config) -> Conditions:
    """The readings a run on `spec` takes, with the source root, model and
    engine `config`, the inputs' configuration, states."""
    return run_readings(
        spec,
        atom_src=Reading(*config["source_root"].split("-", 1)),
        model=config["model"],
        engine_config=json.dumps(config["engine_config"], sort_keys=True),
    )


def publish(args, command: str, key, readings: Conditions, params, report):
    """Hand `params` and `report` off at `key` in `args.store` under `readings`,
    and write their readable copy in `args.out`."""
    members = {"params": params, "report": report}
    ArtifactStore(args.store).publish(
        key,
        provenance=Provenance(
            command,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            roots_for({"atom": "atom", "aiter": "aiter"}),
        ),
        topology=Topology(),
        conditions=readings,
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
