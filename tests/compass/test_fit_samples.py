# SPDX-License-Identifier: MIT
"""Labelled samples from record directories and step measurements, under one configuration.

A record directory holds `test_parity_served.py`'s record and a run.json
written by hand with the identity entries `record` writes there; its server
served two client runs. Measurements are published through `ArtifactStore`.
"""

import copy
import dataclasses
import json
import re

import pytest
from test_artifact_store import CONDITIONS, stanza
from test_memory_readings import STACK
from test_parity_served import SAMPLING, client_run, recorded

from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Topology,
    member_name,
)
from atom.compass.backends.step_labels import digest, label
from atom.compass.fit.samples import MEMBER, load
from atom.compass.parity import ARRIVALS
from atom.compass.parity.served import served
from atom.compass.run.step_grid import BOUNDS

HOST = {"host_cpu": "planted cpu", "atom": "0" * 64, "model": "sha256:" + "1" * 64}
DEVICE = {"index": 7, "name": "MI355X", "arch": "gfx950"}
SOURCE = {
    "name": "atom",
    "root": "/workspace/ATOM/atom",
    "revision": "67c176f0",
    "revision_kind": "tree",
    "method": "git rev-parse HEAD^{tree}, the tree git archive ships",
    "dirty": 0,
}
RUNS = [client_run([0, 1]), client_run([2, 3])]
#: The batch a measurement holds, as a step-record line holds it.
LINE = {"rows": [[1, 300, True], [1, 210, True]], "rung": 2, "produces_output": True}


def record(tmp_path, monkeypatch, name="a", groups=((0, 1), (2, 3)), **run):
    """`tmp_path/name`, a record of `groups` and its run.json, `run` set over it."""
    directory = tmp_path / name
    steps = recorded(directory, monkeypatch, groups, HOST)
    engine = json.loads((steps / "dp0.engine.json").read_text())
    keyed = {
        k: v for k, v in engine.items() if k not in (*BOUNDS, "num_kvcache_blocks")
    }
    identity = {
        "format_version": 2,
        "argv": ["compass", "record", "--out", str(directory)],
        "source": SOURCE,
        "host_key": {**HOST, "engine": keyed},
        "device": [DEVICE],
        "stack": STACK,
        "profiled": False,
    }
    (directory / "run.json").write_text(json.dumps({**identity, **run}))
    return directory


def idle_rank(steps):
    """`steps/dp1.jsonl`, written as a DP rank no request reached: at each of
    rank 0's steps, the dummy forward it runs to keep step with its group."""
    dummy = {"dp_rank": 1, "is_dummy_run": True, "batch": [], "named": []}
    rank0 = (steps / "dp0.jsonl").read_text().splitlines()
    path = steps / "dp1.jsonl"
    path.write_text(
        "".join(json.dumps({**json.loads(x), **dummy}) + "\n" for x in rank0)
    )
    return path


def edit(path, change):
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def measured(store, config, sampling=SAMPLING, origin="replay"):
    """A step measurement keyed by `config`, as `measure steps` publishes one."""
    key = Key.of(
        Kind.STEP_MEASUREMENT,
        **{f: config[f] for f in ("model", "device", "parallel", "source_root")},
        engine_config=json.dumps(config["engine_config"], sort_keys=True),
        sampling=json.dumps(sampling, sort_keys=True),
        step_kind="decode",
        view="sha256:5e1f",
    )
    member = {**LINE, "seconds": 1.5e-4, "origin": origin, "traces": ["trace-9"]}
    ArtifactStore(store).publish(
        key,
        provenance=stanza("compass measure steps"),
        topology=Topology(),
        conditions=CONDITIONS,
        gates=GateState.of(),
        members={member_name(*MEMBER): json.dumps(member).encode()},
    )
    return key


def test_two_records_differing_only_in_kv_cache_dtype_refuse_naming_both_values(
    tmp_path, monkeypatch
):
    a, b = record(tmp_path, monkeypatch, "a"), record(tmp_path, monkeypatch, "b")
    edit(b / "steps/dp0.engine.json", lambda e: e.update(kv_cache_dtype="fp8"))

    with pytest.raises(ValueError) as refused:
        load({a: RUNS, b: RUNS})
    assert str(refused.value) == (
        "the inputs are not one configuration:\n"
        f'engine_config kv_cache_dtype: "bf16" in {a}; "fp8" in {b}'
    )


@pytest.mark.parametrize(
    "field, file, change",
    [
        ("model", "run.json", lambda r: r["host_key"].update(model="sha256:2")),
        ("device", "run.json", lambda r: r["device"][0].update(name="MI300X")),
        (
            "parallel",
            "steps/dp0.engine.json",
            lambda e: e.update(tensor_parallel_size=2),
        ),
        ("source_root", "run.json", lambda r: r["source"].update(revision="0e724ed5")),
        ("host_key host_cpu", "run.json", lambda r: r["host_key"].update(host_cpu="x")),
        ("sampling temperature", None, None),
    ],
)
def test_each_field_read_from_the_input_refuses_when_it_differs(
    tmp_path, monkeypatch, field, file, change
):
    a, b = record(tmp_path, monkeypatch, "a"), record(tmp_path, monkeypatch, "b")
    runs = [client_run([0, 1, 2, 3], {**SAMPLING, "temperature": 0.7})]
    if file:
        edit(b / file, change)
        runs = RUNS

    named = rf"(?m)^{re.escape(field)}: .+ in {re.escape(str(a))}; .+ in {re.escape(str(b))}$"
    with pytest.raises(ValueError, match=named):
        load({a: RUNS, b: runs})


@pytest.mark.parametrize(
    "named, change",
    [
        (
            "was profiled",
            lambda d: edit(d / "run.json", lambda r: r.update(profiled=True)),
        ),
        (
            "is format 1",
            lambda d: edit(d / "run.json", lambda r: r.update(format_version=1)),
        ),
        (f"holds {ARRIVALS}", lambda d: (d / "steps" / ARRIVALS).write_text("{}")),
    ],
)
def test_a_record_not_read_as_an_unprofiled_run_s_lines_refuses(
    tmp_path, monkeypatch, named, change
):
    a = record(tmp_path, monkeypatch)
    change(a)

    with pytest.raises(ValueError, match=re.escape(named)):
        load({a: RUNS})


def test_a_record_serving_two_sets_of_request_parameters_refuses(tmp_path, monkeypatch):
    a = record(tmp_path, monkeypatch)
    runs = [RUNS[0], client_run([2, 3], {**SAMPLING, "temperature": 0.7})]

    with pytest.raises(ValueError, match="under 2 sets of request parameters"):
        load({a: runs})


def test_each_sample_copies_its_input_s_provenance_and_the_traces_it_served(
    tmp_path, monkeypatch
):
    a = record(tmp_path, monkeypatch)
    samples, config = load({a: RUNS})
    path = a / "steps/dp0.jsonl"
    split = {s["step"]: s["traces"] for s in served(a / "steps", RUNS)[0]}

    labelled = label(path)[0]
    assert [s.step for s in samples] == [s.step for s in labelled]
    for s, plain in zip(samples, labelled):
        assert s.provenance == {**plain.provenance, "traces": split[s.step]}
        assert dataclasses.replace(s, provenance=plain.provenance) == plain
    assert {t for s in samples for t in s.provenance["traces"]} == {
        f"trace-{i}" for i in range(4)
    }
    assert samples[0].provenance["run"] == digest(path)
    assert config == {
        "model": HOST["model"],
        "device": "MI355X gfx950",
        "parallel": "tp1-dp1",
        "engine_config": json.loads(a.joinpath("run.json").read_text())["host_key"][
            "engine"
        ],
        "source_root": "tree-67c176f0",
        "stack": STACK,
        "sampling": SAMPLING,
        "host_key": {"host_cpu": "planted cpu", "atom": "0" * 64},
    }


def test_a_measurement_loads_beside_records_of_its_configuration_only(
    tmp_path, monkeypatch
):
    a, store = record(tmp_path, monkeypatch), tmp_path / "store"
    config = load({a: RUNS})[1]
    key = measured(store, config, origin="plan")

    samples, both = load({a: RUNS}, [key], store)
    [sample] = [s for s in samples if s.provenance["source"] == "plan"]
    assert both == config
    assert sample.device_s == 1.5e-4 and sample.view.capture_rung == 2
    assert sample.provenance == {
        "source": "plan",
        "run": ArtifactStore(store).read(key).digest,
        "requests": [],
        "config": None,
        "traces": ["trace-9"],
    }


@pytest.mark.parametrize(
    "named, field, inner, value",
    [
        ("model", "model", None, "sha256:2"),
        ("device", "device", None, "AMD Instinct MI355X M3550 gfx950"),
        ("parallel", "parallel", None, "tp2-dp1"),
        ("source_root", "source_root", None, "tree-0e724ed5"),
        ("engine_config kv_cache_dtype", "engine_config", "kv_cache_dtype", "fp8"),
        ("sampling top_p", "sampling", "top_p", 0.9),
    ],
)
def test_a_measurement_differing_from_the_records_in_one_key_field_refuses(
    tmp_path, monkeypatch, named, field, inner, value
):
    a, store = record(tmp_path, monkeypatch), tmp_path / "store"
    config = load({a: RUNS})[1]
    there = copy.deepcopy(config)
    held = there[field] if inner else there
    old, held[inner or field] = held[inner or field], value
    key = measured(store, there, there["sampling"])

    with pytest.raises(ValueError) as refused:
        load({a: RUNS}, [key], store)
    assert str(refused.value) == (
        "the inputs are not one configuration:\n"
        f"{named}: {json.dumps(old)} in {a}; {json.dumps(value)} in {key}"
    )


def test_samples_come_in_one_order_whatever_order_the_inputs_came_in(
    tmp_path, monkeypatch
):
    a = record(tmp_path, monkeypatch, "a")
    b = record(tmp_path, monkeypatch, "b", groups=((2, 3), (0, 1)))
    store = tmp_path / "store"
    key = measured(store, load({a: RUNS})[1])

    forward = load({a: RUNS, b: RUNS}, [key], store)
    backward = load({b: RUNS[::-1], a: RUNS}, [key], store)
    assert forward == backward
    runs = [s.provenance["run"] for s in forward[0]]
    assert runs == sorted(runs) and len(set(runs)) == 3


def test_a_dp_rank_no_request_reached_adds_no_sample(tmp_path, monkeypatch):
    a = record(tmp_path, monkeypatch)
    alone, config = load({a: RUNS})
    rank1 = idle_rank(a / "steps")

    samples, both = load({a: RUNS})
    run = f"{digest(a / 'steps/dp0.jsonl')}, {digest(rank1)}"
    assert samples == [
        dataclasses.replace(s, provenance={**s.provenance, "run": run}) for s in alone
    ]
    assert both == {**config, "parallel": "tp1-dp2"}


def test_an_empty_rank_file_refuses_as_a_rank_whose_lines_never_reached_disk(
    tmp_path, monkeypatch
):
    a = record(tmp_path, monkeypatch)
    (a / "steps/dp1.jsonl").write_text("")

    with pytest.raises(ValueError, match="dp1.jsonl has no timed steps"):
        load({a: RUNS})
