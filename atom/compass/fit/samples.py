# SPDX-License-Identifier: MIT
"""Labelled samples from `record` directories and step measurements, under one
configuration.

A record directory is `record`'s output: `run.json`, holding the run's
identity in format `FORMAT_VERSION`, and the step record in `steps/`, with the
client runs its server served (`parity.served`). Its steps are labelled by
`step_labels`, and a sample's provenance gains the `traces` its step served.
A DP rank no request reached writes a line per dummy forward it runs, and gives
no sample; its file still counts in the run's digest. An empty
rank file lost its lines, and refuses.

A step measurement is a `step_measurement` entry `measure steps` published,
read by key; its member `MEMBER` holds the batch view as a step-record line
does (`rows`, `rung`, `produces_output`), its device seconds (`seconds`), its
`origin` (`replay` or `plan`) and the `traces` the batch came from. Its sample's
provenance holds its `source`, the origin, its `run`, the entry's digest, and
those `traces`.

Every input states one configuration. A record reads it from its `run.json`,
its `dp0.engine.json` and the requests it served: `model`, the served
config.json's digest; `device`, each GPU's `<name> <arch>`, as a spec states
it; `engine_config`, `keyed`'s fields, the parallel layout among them;
`source_root`, `<revision_kind>-<revision>`; `sampling`, the request
parameters, null for a record serving no client run's request; its host key
less its model and engine; and its `stack`, the software versions run.json
names. A measurement reads the same fields but the host key and stack from its
key, whose `engine_config` and `sampling` are JSON objects as text. A field
that differs between inputs refuses, naming each value and the inputs holding
it; so do a profiled record and a record whose requests carry two sets of
request parameters.
"""

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from atom.compass.artifacts import ArtifactStore, Topology
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_labels import Sample, digest, label
from atom.compass.parity import ARRIVALS
from atom.compass.parity.served import served
from atom.compass.run.step_grid import keyed_of

#: The run.json format this reads.
FORMAT_VERSION = 2
#: A step measurement's one member: the bench replies for the whole TP group.
MEMBER = ("measurement", Topology().ranks()[0], "json")
#: What names a GPU's model in run.json's `device`.
CARD = ("name", "arch")


def load(
    records: Mapping, measurements: Sequence = (), store=None
) -> tuple[list[Sample], dict]:
    """The samples of `records`, {record directory: the client runs its server
    served}, and of the `measurements` keys in the store at `store`, in run
    digest, DP rank and step order; and the configuration they state."""
    samples, configs = [], {}
    for directory, client_runs in records.items():
        got, configs[str(directory)] = _record(Path(directory), client_runs)
        samples += got
    for key in measurements:
        got, configs[str(key)] = _measurement(ArtifactStore(store), key)
        samples.append(got)
    fields = sorted({f for c in configs.values() for f in c})
    differ = [
        line
        for f in fields
        for line in _differ(f, {i: c[f] for i, c in configs.items() if f in c})
    ]
    if differ:
        raise ValueError("the inputs are not one configuration:\n" + "\n".join(differ))
    one = {k: v for c in configs.values() for k, v in c.items()}
    samples.sort(key=lambda s: (s.provenance["run"], s.rank, s.step))
    return samples, one


def _record(directory: Path, client_runs) -> tuple[list[Sample], dict]:
    run = json.loads((directory / "run.json").read_text())
    if run["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"{directory}/run.json is format {run['format_version']}, and this "
            f"reads format {FORMAT_VERSION}"
        )
    if run["profiled"]:
        raise ValueError(
            f"{directory} was profiled, and profiling splits the steps it times, "
            "so no fit takes them"
        )
    steps = directory / "steps"
    if (steps / ARRIVALS).exists():
        # `parity.read` then drops the engine's start and renumbers the rest.
        raise ValueError(
            f"{steps} holds {ARRIVALS}, so its steps are not numbered as its lines"
        )
    record = served(steps, client_runs)
    sampling = sorted({p for s in _steps(record) for p in s["sampling"]})
    if len(sampling) > 1:
        raise ValueError(
            f"{directory} served requests under {len(sampling)} sets of request "
            f"parameters: {' and '.join(sampling)}"
        )
    traces = {(s["dp_rank"], s["step"]): s["traces"] for s in _steps(record)}
    files = sorted(steps.glob("dp*[0-9].jsonl"), key=lambda p: int(p.stem[2:]))
    name, samples = ", ".join(map(digest, files)), []
    for path in files:
        lines = record[int(path.stem[2:])]
        if lines and all(s["is_dummy_run"] for s in lines):
            continue
        for s in label(path, name)[0]:
            traced = {**s.provenance, "traces": traces[s.rank, s.step]}
            samples.append(dataclasses.replace(s, provenance=traced))
    engine = json.loads((steps / "dp0.engine.json").read_text())
    host = run["host_key"]
    return samples, {
        **run_fields(run),
        "engine_config": keyed_of(engine),
        "stack": run["stack"],
        "sampling": json.loads(sampling[0]) if sampling else None,
        "host_key": {k: v for k, v in host.items() if k not in ("model", "engine")},
    }


def run_fields(run: dict) -> dict:
    """The `model`, `device` and `source_root` a run.json's identity states."""
    return {
        "model": run["host_key"]["model"],
        "device": ", ".join(
            sorted({" ".join(str(gpu[f]) for f in CARD) for gpu in run["device"]})
        ),
        "source_root": f"{run['source']['revision_kind']}-{run['source']['revision']}",
    }


def _steps(record: dict) -> list[dict]:
    return [s for steps in record.values() for s in steps]


def _measurement(store: ArtifactStore, key) -> tuple[Sample, dict]:
    entry = store.read(key)
    try:
        m = json.loads(entry.read_member(*MEMBER))
        view = BatchView(
            tuple(RequestShape(q, c, bool(d)) for q, c, d in m["rows"]),
            capture_rung=m["rung"],
            produces_output=m["produces_output"],
        )
        provenance = {
            "source": m["origin"],
            "run": entry.digest,
            "requests": [],
            "config": None,
            "traces": sorted(m["traces"]),
        }
        values = key.values
        config = {
            **{f: values[f] for f in ("model", "device", "source_root")},
            **{f: json.loads(values[f]) for f in ("engine_config", "sampling")},
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{key} does not load: {type(error).__name__}: {error}"
        ) from error
    return Sample(0, 0, view, m["seconds"], None, {}, False, provenance), config


def _differ(field: str, held: dict) -> list[str]:
    """Where `held`, {input: value}, disagrees on `field`; a mapping field by field."""
    if all(isinstance(v, Mapping) for v in held.values()):
        names = sorted({k for v in held.values() for k in v})
        return [
            line
            for k in names
            for line in _differ(f"{field} {k}", {i: v.get(k) for i, v in held.items()})
        ]
    by: dict[str, list[str]] = {}
    for i, v in held.items():
        by.setdefault(json.dumps(v, sort_keys=True), []).append(i)
    if len(by) < 2:
        return []
    return [f"{field}: " + "; ".join(f"{v} in {', '.join(i)}" for v, i in by.items())]
