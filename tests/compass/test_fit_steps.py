# SPDX-License-Identifier: MIT
"""`compass fit steps`, run through `cli.main`.

Each record directory holds one of `test_backend_step_linear_fit.py`'s planted
timed records as `steps/dp0.jsonl`, its host key and engine file beside it,
and a run.json written by hand as `test_fit_samples.py` writes one. The store
is a fresh directory; the entry's provenance names planted trees, since git
cannot name a read-only checkout in every container.
"""

import json
import re
from pathlib import Path

import pytest
from test_artifact_store import AITER_ROOT, ATOM_ROOT
from test_backend_step_linear_fit import _steps, _timed_record
from test_fit_samples import DEVICE, SOURCE, idle_rank
from test_memory_readings import DOCUMENT, STACK
from test_parity_served import SAMPLING

from atom.compass import cli
from atom.compass.artifacts import (
    ArtifactStore,
    Axis,
    Key,
    Kind,
    Reading,
    Row,
    machine_spec_digest,
)
from atom.compass.backends import StepLinearLaw, step_linear_fit
from atom.compass.backends.step_labels import digest
from atom.compass.fit import publish
from atom.compass.run.step_grid import BOUNDS
from atom.compass.spec import MachineSpec

ENGINE = {
    "ladder": [],
    "kv_cache_block_size": 16,
    "kv_cache_dtype": "bf16",
    "quantization": "unquantized-bf16",
    "tensor_parallel_size": 1,
    "data_parallel_size": 1,
}
HOST = {
    "host_cpu": "planted cpu",
    "atom": "0" * 64,
    "model": "sha256:" + "1" * 64,
    "engine": ENGINE,
}
LINEAR = ("--backend", "step-linear")
#: The digest of the run files' machine spec, the one every fit here names.
DIGEST = machine_spec_digest(MachineSpec.from_mapping(DOCUMENT))


@pytest.fixture(autouse=True)
def planted_roots(monkeypatch):
    monkeypatch.setattr(publish, "roots_for", lambda modules: (ATOM_ROOT, AITER_ROOT))


def record(directory, seed, key=HOST, keyed=True, stack=STACK, trace="trace"):
    """`directory`, a record of `_steps(seed)`; without `keyed`, no host key
    beside it. Every request is named, and `directory/client.json` is the
    export of one client run sending them all, each from trace `<trace>.<k>`,
    `k` its first step's index over 100."""
    steps = directory / "steps"
    steps.mkdir(parents=True)
    path = steps / "dp0.jsonl"
    _timed_record(path, _steps(seed), key=key if keyed else None)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    sent = {}
    for i, line in enumerate(lines):
        line["named"] = [[r[0], f"r{r[0]}"] for r in line["batch"]]
        for r in line["batch"]:
            at = {"trace": f"{trace}.{i // 100}", "sampling": SAMPLING}
            sent.setdefault(f"r{r[0]}", at)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    (directory / "client.json").write_text(json.dumps({"requests": sent}))
    engine = {**key["engine"], **dict.fromkeys(BOUNDS, 4096), "num_kvcache_blocks": 900}
    (steps / "dp0.engine.json").write_text(json.dumps(engine))
    run = {
        "format_version": 2,
        "argv": ["compass", "record", "--out", str(directory)],
        "source": SOURCE,
        "host_key": key,
        "device": [DEVICE],
        "stack": stack,
        "ptl": "disabled",
        "profiled": False,
    }
    (directory / "run.json").write_text(json.dumps(run))
    return directory


def fit(tmp_path, command, *args):
    """`compass <command> <args>` on the run files' machine spec, publishing to
    `tmp_path/store`."""
    machine, store, out = (str(tmp_path / f) for f in ("spec.json", "store", "out"))
    Path(machine).write_text(json.dumps(DOCUMENT))
    tail = ["--machine", machine, "--store", store, "--out", out]
    cli.main([*command.split(), *map(str, args), *tail])


def law_key(form="history") -> Key:
    return Key.of(
        Kind.STEP_LINEAR_LAW,
        model=HOST["model"],
        engine_config=json.dumps(ENGINE, sort_keys=True),
        source_root="tree-67c176f0",
        form=form,
        machine_spec=DIGEST,
    )


def member(store, key, stem):
    return json.loads(
        ArtifactStore(store).read(key).read_member(stem, publish.RANK, "json")
    )


@pytest.mark.parametrize("form", step_linear_fit.FORMS)
def test_the_law_published_is_the_one_step_linear_fits_command_writes_less_its_host_laws(
    tmp_path, form
):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    fit(tmp_path, "fit steps", *LINEAR, "--form", form, a, b, *clients(a, b))

    law = member(tmp_path / "store", law_key(form), "params")
    # The command names runs in argument order, the samples in digest order.
    paths = sorted((d / "steps/dp0.jsonl" for d in (a, b)), key=digest)
    law_file = str(tmp_path / "law.json")
    step_linear_fit.main(
        [*map(str, paths), "--form", form, "--machine-spec", DIGEST, "--out", law_file]
    )
    written = json.loads((tmp_path / "law.json").read_text())
    assert law == {k: v for k, v in written.items() if k not in ("host", "host_fit")}
    StepLinearLaw(law)
    copy = json.loads((tmp_path / "out/step_linear_law.json").read_text())
    report = member(tmp_path / "store", law_key(form), "report")
    assert copy == {
        "key": law_key(form).values,
        "token": law_key(form).token(),
        "params": law,
        "report": report,
    }


def test_step_linear_fits_the_history_form_unless_told_another(tmp_path):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    for name, *form in (("default",), ("base", "--form", "base")):
        (tmp_path / name).mkdir()
        fit(tmp_path / name, "fit steps", *LINEAR, *form, a, b, *clients(a, b))

    law = member(tmp_path / "default/store", law_key("history"), "params")
    assert law["form"] == "history"
    assert list(law["prefill"]["coefficients"]) == list(step_linear_fit.HISTORY)
    base = member(tmp_path / "base/store", law_key("base"), "params")
    assert list(base["prefill"]["coefficients"]) == list(step_linear_fit.PREFILL)


def test_fit_steps_twice_on_the_same_inputs_refuses_the_second_as_the_same_members(
    tmp_path,
):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    fit(tmp_path, "fit steps", *LINEAR, "--seed", 7, a, b, *clients(a, b))

    with pytest.raises(SystemExit, match="this publish holds the same members"):
        fit(tmp_path, "fit steps", *LINEAR, "--seed", 7, b, a, *clients(b, a))


def clients(*records):
    """`--client-run` with each record's export."""
    return [x for d in records for x in ("--client-run", d, d / "client.json")]


def served(key, traced) -> list[dict]:
    """The report's provenance of the records `traced` names, each with its trace."""
    named = {digest(d / "steps/dp0.jsonl"): t for d, t in traced.items()}
    return [
        {
            "source": "record",
            "run": run,
            "config": key,
            "traces": [f"{named[run]}.{k}" for k in range(3)],
        }
        for run in sorted(named)
    ]


def test_the_report_names_each_run_with_every_trace_its_steps_served(tmp_path):
    a = record(tmp_path / "a", 19, trace="trace-a")
    b = record(tmp_path / "b", 20, trace="trace-b")
    fit(tmp_path, "fit steps", *LINEAR, a, b, *clients(a, b))

    report = member(tmp_path / "store", law_key(), "report")
    assert report["provenance"] == served(HOST, {a: "trace-a", b: "trace-b"})


def test_the_entry_s_readings_are_the_ones_its_records_state(tmp_path):
    a = record(tmp_path / "a", 19)
    fit(tmp_path, "fit steps", *LINEAR, a, *clients(a))

    entry = ArtifactStore(tmp_path / "store").read(law_key())
    assert dict(entry.fingerprints[Row.STEP_LINEAR_LAW].cells) == {
        Axis.SOFTWARE_STACK: Reading(
            "stated", "rocm 7.2.4 / aiter f4e7c7509 / rccl 2.22.3"
        ),
        Axis.TORCH: Reading("stated", "2.10.0"),
        Axis.ATOM_SRC: Reading("tree", "67c176f0"),
        Axis.MODEL: Reading("stated", HOST["model"]),
        Axis.DEVICE: Reading("stated", "MI355X gfx950"),
        Axis.PTL: Reading("stated", "disabled"),
        Axis.ENGINE_CONFIG: Reading("stated", json.dumps(ENGINE, sort_keys=True)),
    }
    assert entry.provenance.source_roots == (ATOM_ROOT, AITER_ROOT)


def test_records_under_two_stacks_refuse_naming_the_component_and_both_values(
    tmp_path,
):
    a = record(tmp_path / "a", 19)
    b = record(tmp_path / "b", 20, stack={**STACK, "torch": "2.9.0"})

    named = f'stack torch: "2.10.0" in {a}; "2.9.0" in {b}'
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit steps", *LINEAR, a, b, *clients(a, b))


@pytest.mark.parametrize(
    "args, named",
    [
        (["--backend", "step-stub"], "stated, not fitted"),
        (
            ["--backend", "step-op"],
            "no backend step-op is fitted; step-linear and step-grid are",
        ),
        (
            ["--backend", "step-linear", "--grid-definition", "g.json"],
            "--grid-definition is read by step-grid, not step-linear",
        ),
        (
            ["--backend", "step-linear", "0f1e2d3c4b5a"],
            "0f1e2d3c4b5a is not a record directory",
        ),
        (
            ["--backend", "step-linear", "--client-run", "c", "x.json"],
            "names c, which is not a RECORD",
        ),
    ],
)
def test_what_step_linear_does_not_fit_refuses_by_name(tmp_path, args, named):
    a = record(tmp_path / "a", 19)

    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit steps", *args, a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_a_record_no_client_run_names_refuses_naming_it(tmp_path):
    a = record(tmp_path / "a", 19)

    with pytest.raises(SystemExit, match=re.escape(f"{a} is named by no --client-run")):
        fit(tmp_path, "fit steps", *LINEAR, a)
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize(
    "backend, named",
    [
        ("step-linear", "--sessions is read by step-grid, not step-linear"),
        ("step-grid", "{b} is named by no --client-run"),
    ],
)
def test_sessions_refuse_off_step_grid_and_an_unnamed_grid_input_refuses_under_them(
    tmp_path, backend, named
):
    a, b = record(tmp_path / "a", 19), tmp_path / "b"
    grid = ["--grid", b, "--grid-definition", "g"] if backend == "step-grid" else []
    sessions = ["--sessions", "0:4", *clients(a)]

    with pytest.raises(SystemExit, match=re.escape(named.format(b=b))):
        fit(tmp_path, "fit steps", "--backend", backend, a, *grid, *sessions)
    assert not (tmp_path / "store").exists()


def test_the_join_width_is_the_engine_configs_not_the_record_files(tmp_path):
    a = record(
        tmp_path / "a", 19, key={**HOST, "engine": {**ENGINE, "data_parallel_size": 2}}
    )

    with pytest.raises(SystemExit, match="are not runs of 2 ranks"):
        fit(tmp_path, "fit steps", *LINEAR, a, *clients(a))


def test_a_dp_record_a_rank_of_which_gave_no_sample_refuses_at_its_width(tmp_path):
    a = record(
        tmp_path / "a", 19, key={**HOST, "engine": {**ENGINE, "data_parallel_size": 2}}
    )
    idle_rank(a / "steps")

    with pytest.raises(SystemExit, match="are not runs of 2 ranks"):
        fit(tmp_path, "fit steps", *LINEAR, a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_records_predating_the_data_parallel_width_refuse_naming_it(tmp_path):
    engine = {k: v for k, v in ENGINE.items() if k != "data_parallel_size"}
    a = record(tmp_path / "a", 19, key={**HOST, "engine": engine})
    with pytest.raises(SystemExit, match="states no data_parallel_size"):
        fit(tmp_path, "fit steps", *LINEAR, a, *clients(a))
