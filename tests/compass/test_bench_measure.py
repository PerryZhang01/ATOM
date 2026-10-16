# SPDX-License-Identifier: MIT
"""`compass measure steps`, run through `cli.main`.

Inputs are `test_fit_samples.py`'s record, whose step record ATOM's
`Scheduler` drove through `test_parity_record.py`'s runner and which served
two client runs, and that record's lines published as a `shape_population`.
The bench is a fake in `Bench`'s place, on the record's engine, whose result
is the batch's token count in microseconds; run.json's identity is the
record's, so a measurement states the record's configuration; the GPU checks
pass and the provenance names planted trees.
"""

import json
import random
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_artifact_store import AITER_ROOT, ATOM_ROOT, CONDITIONS, stanza
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture
from test_fit_samples import RUNS, record
from test_parity_record import _config
from test_parity_served import SAMPLING, client_run
from test_step_grid_plan import DEFINITION

from atom.compass import cli
from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Row,
    Topology,
    member_name,
)
from atom.compass.artifacts import store as store_module
from atom.compass.backends.step_grid_plan import plan
from atom.compass.bench import measure, window
from atom.compass.fit.samples import load
from atom.compass.parity import read
from atom.compass.parity.served import served


class Bench:
    """The bench, measuring each batch at its token count in microseconds."""

    started, measured = [], []
    refuse = None

    def __init__(self, config):
        Bench.started.append(config)

    def measure(self, batch):
        Bench.measured.append(batch)
        if Bench.refuse and batch.total_tokens_num == Bench.refuse:
            return {"protocol": window.PROTOCOL, "refused": "a planted refusal"}
        ns = 1_000 * batch.total_tokens_num
        return {
            "protocol": window.PROTOCOL,
            "median_ns": ns,
            "repeats": [[ns, 0, ns]] * 3,
            "discarded": 0,
            "preload_ns": 5,
        }

    def exit(self):
        Bench.started.append("exit")


@pytest.fixture
def run(tmp_path, monkeypatch, model_dir):  # noqa: F811
    """`run(*args)`: `compass measure steps args --gpu 7 --store S --out D`,
    with `--sampling` SAMPLING unless given; a client run export per record
    client run is at `exports`."""
    rec = record(tmp_path, monkeypatch)
    config = _config()
    config.model, config.trust_remote_code = str(model_dir), False
    config.hf_config = SimpleNamespace(model_type="llama")
    monkeypatch.setattr(measure, "atom_config", lambda atom_args: config)
    identity = json.loads((rec / "run.json").read_text())
    monkeypatch.setattr(measure, "identity", lambda *_: dict(identity))
    monkeypatch.setattr(measure, "quiet", lambda gpus, out: nullcontext())
    monkeypatch.setattr(measure, "roots_for", lambda _: (ATOM_ROOT, AITER_ROOT))
    monkeypatch.setattr(measure, "start_bench", Bench)
    monkeypatch.setattr(Bench, "started", [])
    monkeypatch.setattr(Bench, "measured", [])
    exports = []
    for i, requests in enumerate(RUNS):
        exports.append(tmp_path / f"export{i}.json")
        exports[-1].write_text(json.dumps({"requests": requests}))

    def run(*args, out="out"):
        args = list(args)
        if "--sampling" not in args:
            args += ["--sampling", json.dumps(SAMPLING)]
        cli.main(
            ["measure", "steps", *args, "--gpu", "7"]
            + ["--store", str(tmp_path / "store"), "--out", str(tmp_path / out)]
            + ["--", "--model", "m"]
        )
        return ArtifactStore(tmp_path / "store")

    run.record, run.exports, run.tmp = rec, exports, tmp_path
    run.client_runs = [a for e in exports for a in ("--client-run", str(rec), str(e))]
    return run


def views(lines) -> set:
    return {measure._digest({k: s[k] for k in measure.VIEW}) for s in lines}


def entries(store) -> list:
    """Each step measurement in `store`, as (key, member)."""
    held = []
    for place in sorted((store.root / "step_measurement").iterdir()):
        document = json.loads((place / "entry.json").read_text())
        key = Key.of(Kind.STEP_MEASUREMENT, **document["key"])
        held.append((key, json.loads(store.read(key).read_member(*measure.MEMBER))))
    return held


def by_view(store) -> dict:
    return {key.values["view"]: (key, member) for key, member in entries(store)}


def test_measuring_one_record_twice_publishes_each_view_once_and_then_nothing(run):
    store = run(str(run.record), "--sample", "all", *run.client_runs)
    lines, split = read(run.record / "steps")[0], served(run.record / "steps", RUNS)[0]
    held = by_view(store)
    assert set(held) == views(lines) and len(Bench.measured) == len(held)
    assert Bench.started[-1] == "exit"
    for view, (key, member) in held.items():
        assert measure._digest({k: member[k] for k in measure.VIEW}) == view
        decode, output = member["rows"][0][2], member["produces_output"]
        assert key.values["step_kind"] == (
            f"decode rung {member['rung']}"
            if decode
            else f"prefill {'with' if output else 'without'} output"
        )
        assert member["seconds"] == 1e-6 * sum(q for q, *_ in member["rows"])
        assert member["repeats"] and member["statistic"] == "median"
        assert member["protocol"] == window.PROTOCOL
        assert member["replay_rule"] == {"name": "fresh-contiguous", "version": 1}
        assert member["origin"] == "replay"
        assert member["sources"] == [
            measure.file_digest(run.record / "steps/dp0.jsonl")
        ]
        assert member["traces"] == sorted(
            {t for s in split if views([s]) == {view} for t in s["traces"]}
        )
    # Each key states the record's configuration as `fit` reads it.
    config = load({run.record: RUNS})[1]
    for key, _ in held.values():
        stated = {f: json.loads(key.values[f]) for f in ("engine_config", "sampling")}
        for f in ("model", "device", "parallel", "source_root"):
            stated[f] = key.values[f]
        assert stated == {f: config[f] for f in stated}
    phases = json.loads((run.tmp / "out/run.json").read_text())["phases"]
    assert list(phases) == ["warm-up", "measuring"]
    assert all("end" in edges for edges in phases.values())

    Bench.started.clear()
    run(str(run.record), "--sample", "all", *run.client_runs)
    assert Bench.started == [] and by_view(store).keys() == held.keys()


def test_sample_n_draws_n_views_from_each_stratum_by_its_seed(run):
    lines = read(run.record / "steps")[0]
    strata = {}
    for s in lines:
        view = {k: s[k] for k in measure.VIEW}
        strata.setdefault(measure._stratum(view), set()).add(measure._digest(view))
    drawn = []
    for seed in (3, 3, 4):
        store = run(str(run.record), "--sample", "2", "--seed", str(seed))
        drawn.append(set(by_view(store)))
        store.root.rename(run.tmp / f"store{len(drawn)}")
    rng, expected = random.Random(3), set()
    for stratum in sorted(strata):
        views_in = sorted(strata[stratum])
        expected.update(rng.sample(views_in, min(2, len(views_in))))
    assert drawn[0] == drawn[1] == expected != drawn[2]
    assert any(len(v) > 2 for v in strata.values())


@pytest.mark.parametrize(
    "args, refusal",
    [
        ([], "names no input to replay"),
        (["REC"], "drawn by --sample N --seed S or --sample all"),
        (["--grid", "REC", "--grid-definition", "d", "--sample", "all"], "no replayed"),
        (["REC", "--sample", "2"], "--sample N needs --seed"),
        (["--grid", "REC"], "--grid-definition FILE"),
        (["REC", "--sample", "all", "--grid-definition", "d"], "plans no --grid"),
        (["REC", "--sample", "all", "--client-run", "x", "e"], "names x, which is no"),
        (["REC", "--sample", "all", "--sampling", '{"temp": 1}'], "SamplingParams"),
        (["REC", "--sample", "all", "--sessions", "0:1"], "no batch to measure"),
    ],
)
def test_what_does_not_say_what_to_measure_refuses(run, args, refusal):
    args = [str(run.record) if a == "REC" else a for a in args]
    with pytest.raises(SystemExit, match=refusal):
        run(*args)
    assert Bench.started == []


def published_population(run, monkeypatch) -> str:
    """The record's steps and engine, published as a shape_population; its token.

    The kind has no invalidation row until `trace steps` publishes one, so it
    takes the step grid bundle's here.
    """
    rows = store_module.rows_for
    monkeypatch.setattr(
        store_module,
        "rows_for",
        lambda kind: (
            (Row.STEP_GRID_BUNDLE,) if kind is Kind.SHAPE_POPULATION else rows(kind)
        ),
    )
    key = Key.of(Kind.SHAPE_POPULATION, model="m", workload="w", engine_config="e")
    rank = Topology().ranks()[0]
    files = {("steps", "jsonl"): "dp0.jsonl", ("engine", "json"): "dp0.engine.json"}
    ArtifactStore(run.tmp / "store").publish(
        key,
        provenance=stanza("compass trace steps"),
        topology=Topology(),
        conditions=CONDITIONS,
        gates=GateState.of(),
        members={
            member_name(stem, rank, ext): (run.record / "steps" / name).read_bytes()
            for (stem, ext), name in files.items()
        },
    )
    return key.token()


def test_grid_points_are_measured_with_origin_plan_and_new_shapes_add_only_theirs(
    run, monkeypatch
):
    token = published_population(run, monkeypatch)
    definition = run.tmp / "definition.json"
    definition.write_text(json.dumps(DEFINITION))
    lines = read(run.record / "steps")[0]
    engine = {"r": json.loads((run.record / "steps/dp0.engine.json").read_text())}

    def prefill(d, steps, e):
        """The plan of the prefill steps alone."""
        return plan(d, [(n, s) for n, s in steps if not s["rows"][0][2]], e)

    first = prefill(DEFINITION, [(str(s["step"]), s) for s in lines], engine)["lines"]
    planned = plan(DEFINITION, [(str(s["step"]), s) for s in lines], engine)["lines"]
    assert views(first) < views(planned)

    with monkeypatch.context() as patch:
        patch.setattr(measure, "plan", prefill)
        store = run("--grid", token, "--grid-definition", str(definition))
    assert set(by_view(store)) == views(first)
    Bench.measured.clear()
    store = run("--grid", token, str(run.record), "--grid-definition", str(definition))

    assert len(Bench.measured) == len(views(planned) - views(first))
    plan_digest = measure._digest(json.loads((run.tmp / "out/plan.json").read_text()))
    for view, (_, member) in by_view(store).items():
        assert member["origin"] == "plan"
        assert (member["plan"] == plan_digest) == (view not in views(first))


def test_sessions_keep_the_lines_that_serve_only_those_traces(run, monkeypatch):
    """All four requests are served at once, so decode steps serve all four traces."""
    mixed = record(run.tmp, monkeypatch, "mixed", groups=((0, 1, 2, 3),))
    export = run.tmp / "all.json"
    export.write_text(json.dumps({"requests": client_run([0, 1, 2, 3])}))
    store = run(
        str(mixed),
        "--sample",
        "all",
        "--sessions",
        "0:1",
        "--client-run",
        str(mixed),
        str(export),
    )

    split = served(mixed / "steps", [client_run([0, 1, 2, 3])])[0]
    kept = [
        s for s in split if s["traces"] and set(s["traces"]) <= {"trace-0", "trace-1"}
    ]
    assert kept and set(by_view(store)) == views(kept) < views(split)
    for _, member in entries(store):
        assert set(member["traces"]) <= {"trace-0", "trace-1"}


def test_a_line_a_dummy_run_wrote_is_not_measured(run):
    steps = run.record / "steps/dp0.jsonl"
    dummy = {**json.loads(steps.read_text().splitlines()[-1]), "is_dummy_run": True}
    dummy |= {"rows": [[1, 7, True]], "rung": 1, "batch": [], "named": []}
    steps.write_text(steps.read_text() + json.dumps(dummy) + "\n")
    store = run(str(run.record), "--sample", "all")
    assert set(by_view(store)) == views(read(run.record / "steps")[0][:-1])


def test_the_sampling_joins_the_key_and_reaches_the_rule(run):
    for temperature in (0.0, 0.7):
        sampling = json.dumps({**SAMPLING, "temperature": temperature})
        store = run(str(run.record), "--sample", "all", "--sampling", sampling)
        assert Bench.measured
        for batch in Bench.measured:
            assert batch.temperatures.tolist() == pytest.approx(
                [temperature] * len(batch.req_ids)
            )
        Bench.measured.clear()
    keys = [key for key, _ in entries(store)]
    assert len(keys) == 2 * len(views(read(run.record / "steps")[0]))
    assert {json.loads(k.values["sampling"])["temperature"] for k in keys} == {0.0, 0.7}


@pytest.mark.parametrize(
    "file, change, refusal",
    [
        ("dp0.engine.json", {"kv_cache_dtype": "fp8"}, "kv_cache_dtype 'fp8' there"),
        ("dp1.jsonl", None, "holds 2 DP ranks, and the bench measures one"),
    ],
)
def test_an_input_the_bench_cannot_stand_for_refuses_before_it_starts(
    run, file, change, refusal
):
    steps = run.record / "steps"
    if change:
        (steps / file).write_text(
            json.dumps(json.loads((steps / file).read_text()) | change)
        )
    else:
        (steps / file).write_text((steps / "dp0.jsonl").read_text())
    with pytest.raises(SystemExit, match=refusal):
        run(str(run.record), "--sample", "all")
    assert Bench.started == []


def test_a_batch_the_bench_refuses_is_not_published_and_is_named_after_the_rest(run):
    lines = read(run.record / "steps")[0]
    refused = Bench.refuse = sum(q for q, *_ in lines[0]["rows"])
    try:
        with pytest.raises(SystemExit, match="batches refused:\n.+a planted refusal"):
            run(str(run.record), "--sample", "all")
    finally:
        Bench.refuse = None
    held = entries(ArtifactStore(run.tmp / "store"))
    assert refused not in {sum(q for q, *_ in m["rows"]) for _, m in held}
    assert len(Bench.measured) == len(views(lines)) > len(held)
