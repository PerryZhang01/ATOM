# SPDX-License-Identifier: MIT
"""Labelled samples from record directories, under one configuration.

A record directory holds `test_parity_served.py`'s record and a run.json
written by hand with the identity entries `record` writes there; its server
served two client runs.
"""

import dataclasses
import json
import re

import pytest
from test_parity_served import SAMPLING, client_run, recorded

from atom.compass.backends.step_labels import digest, label
from atom.compass.fit.samples import load
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
        "stack": {"rocm": "7.2.4", "torch": "2.10.0", "aiter": "v0.1.21"},
        "profiled": False,
    }
    (directory / "run.json").write_text(json.dumps({**identity, **run}))
    return directory


def edit(path, change):
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


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
        "stack": {"rocm": "7.2.4", "torch": "2.10.0", "aiter": "v0.1.21"},
        "sampling": SAMPLING,
        "host_key": {"host_cpu": "planted cpu", "atom": "0" * 64},
    }


def test_samples_come_in_one_order_whatever_order_the_inputs_came_in(
    tmp_path, monkeypatch
):
    a = record(tmp_path, monkeypatch, "a")
    b = record(tmp_path, monkeypatch, "b", groups=((2, 3), (0, 1)))

    forward = load({a: RUNS, b: RUNS})
    backward = load({b: RUNS[::-1], a: RUNS})
    assert forward == backward
    runs = [s.provenance["run"] for s in forward[0]]
    assert runs == sorted(runs) and len(set(runs)) == 2
