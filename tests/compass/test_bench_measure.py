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

import copy
import json
import re
import shutil
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
    Topology,
    member_name,
)
from atom.compass.backends.step_grid_plan import plan
from atom.compass.bench import measure, window
from atom.compass.fit.samples import load
from atom.compass.parity import read
from atom.compass.parity.served import served


class Bench:
    """The bench, measuring each batch at its token count in microseconds."""

    started, measured = [], []
    refuse = None

    def __init__(self, config, num_blocks):
        config.num_kvcache_blocks = num_blocks
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
    with `--sampling` SAMPLING unless given. `run.named(input, sampling)` is
    the `--client-run` arguments naming the record's two client runs for
    `input`, their requests sent under `sampling`, trace i at session i;
    run.json's identity is `run.identity`."""
    rec = record(tmp_path, monkeypatch)
    config = _config()
    config.model, config.trust_remote_code = str(model_dir), False
    config.hf_config = SimpleNamespace(model_type="llama")
    monkeypatch.setattr(measure, "atom_config", lambda atom_args: config)
    monkeypatch.setattr(measure, "identity", lambda *_: copy.deepcopy(run.identity))
    monkeypatch.setattr(measure, "quiet", lambda gpus, out: nullcontext())
    monkeypatch.setattr(measure, "roots_for", lambda _: (ATOM_ROOT, AITER_ROOT))
    monkeypatch.setattr(measure, "start_bench", Bench)
    monkeypatch.setattr(Bench, "started", [])
    monkeypatch.setattr(Bench, "measured", [])

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

    def named(name, sampling=SAMPLING):
        args = []
        for i, group in enumerate(([0, 1], [2, 3])):
            export = tmp_path / f"export{i}-{json.dumps(sampling, sort_keys=True)}"
            sessions = {f"trace-{k}": k for k in group}
            sent = {"requests": client_run(group, sampling), "sessions": sessions}
            export.write_text(json.dumps(sent))
            args += ["--client-run", str(name), str(export)]
        return args

    run.record, run.tmp, run.named, run.config = rec, tmp_path, named, config
    run.identity = json.loads((rec / "run.json").read_text())
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
    rec = str(run.record)
    store = run(rec, "--sample", "all", *run.named(rec))
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
        for f in ("model", "device", "source_root"):
            stated[f] = key.values[f]
        assert stated == {f: config[f] for f in stated}
    phases = json.loads((run.tmp / "out/run.json").read_text())["phases"]
    assert list(phases) == ["warm-up", "measuring"]
    assert all("end" in edges for edges in phases.values())

    Bench.started.clear()
    run(rec, "--sample", "all", *run.named(rec))
    assert Bench.started == [] and by_view(store).keys() == held.keys()


def test_a_held_entry_taken_under_other_readings_refuses_naming_it(run):
    rec = str(run.record)
    store = run(rec, "--sample", "all", *run.named(rec))
    run.identity["stack"]["aiter"] = "v9.9.9"
    Bench.started.clear()
    tokens = "|".join(key.token() for key, _ in entries(store))
    with pytest.raises(
        SystemExit, match=rf"other readings than this run's:\n({tokens}): .*v9\.9\.9"
    ):
        run(rec, "--sample", "all", *run.named(rec))
    assert Bench.started == []


def test_sample_n_draws_n_views_from_each_stratum_by_its_seed(run):
    """This record decodes on rungs and prefills in chunks under 512 tokens."""
    strata = {}
    for s in read(run.record / "steps")[0]:
        decode = s["rows"][0][2]
        assert decode or sum(q for q, *_ in s["rows"]) < 512
        stratum = f"rung {s['rung']}" if decode else s["produces_output"]
        strata.setdefault(stratum, set()).update(views([s]))
    drawn = []
    for seed in (3, 3, 4):
        rec = str(run.record)
        store = run(rec, "--sample", "2", "--seed", str(seed), *run.named(rec))
        drawn.append(set(by_view(store)))
        store.root.rename(run.tmp / f"store{len(drawn)}")
    assert drawn[0] == drawn[1] != drawn[2]
    for held in strata.values():
        for got in drawn:
            assert len(got & held) == min(2, len(held))
    assert any(len(v) > 2 for v in strata.values())


@pytest.mark.parametrize(
    "rows, rung, output, stratum",
    [
        ([[1, 9, True]] * 3, 4, True, "decode rung 4"),
        ([[1, 9, True]] * 40, None, True, "decode eager"),
        ([[511, 511, False]], None, True, "prefill with output tokens 0-512"),
        (
            [[256, 300, False]] * 2,
            None,
            False,
            "prefill without output tokens 512-2048",
        ),
        ([[2048, 2048, False]], None, True, "prefill with output tokens 2048-8192"),
        ([[4096, 4096, False]] * 2, None, True, "prefill with output tokens 8192+"),
    ],
)
def test_a_batch_is_drawn_by_its_kind_and_rung_or_its_prefill_token_band(
    rows, rung, output, stratum
):
    view = {"rows": rows, "rung": rung, "produces_output": output}
    assert measure._stratum(view) == stratum


@pytest.mark.parametrize(
    "args, refusal",
    [
        ([], "names no input to replay"),
        (["REC", "CR"], "drawn by --sample N --seed S or --sample all"),
        (["--grid", "REC", "--grid-definition", "d", "--sample", "all"], "no replayed"),
        (["REC", "CR", "--sample", "2"], "--sample N needs --seed"),
        (["--grid", "REC"], "--grid-definition FILE"),
        (["REC", "CR", "--sample", "all", "--grid-definition", "d"], "plans no --grid"),
        (["REC", "CR", "--sample", "all", "--client-run", "x", "e"], "names x, which"),
        (["REC", "--sample", "all"], "REC is named by no --client-run"),
        (
            ["--grid", "REC", "--grid-definition", "d"],
            "REC is named by no --client-run",
        ),
        (
            ["REC", "CR", "--sample", "all", "--sampling", '{"temp": 1}'],
            "SamplingParams",
        ),
        (["REC", "CR", "--sample", "all", "--sessions", "4:9"], "no batch to measure"),
        (
            ["REC", "CR", "--sample", "all", "--sampling", '{"temperature": 0.7}'],
            (
                'REC served requests under {"temperature": 0.0, "top_p": 1.0}, and '
                '--sampling states {"temperature": 0.7}'
            ),
        ),
    ],
)
def test_what_does_not_say_what_to_measure_refuses(run, args, refusal):
    rec = str(run.record)
    args = [a for arg in args for a in (run.named(rec) if arg == "CR" else [arg])]
    args = [rec if a == "REC" else a for a in args]
    with pytest.raises(SystemExit, match=re.escape(refusal.replace("REC", rec))):
        run(*args)
    assert Bench.started == []


def published_population(run) -> Key:
    """The record's steps and engine, published as a shape_population."""
    key = Key.of(
        Kind.SHAPE_POPULATION, model="m", workload="w", engine_config="e", backend="b"
    )
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
    return key


def test_grid_points_are_measured_with_origin_plan_and_new_shapes_add_only_theirs(
    run, monkeypatch
):
    population = published_population(run)
    token, rec = population.token(), str(run.record)
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

    grid = ["--grid-definition", str(definition), *run.named(token)]
    with monkeypatch.context() as patch:
        patch.setattr(measure, "plan", prefill)
        store = run("--grid", token, *grid)
    assert set(by_view(store)) == views(first)
    Bench.measured.clear()
    store = run("--grid", token, rec, *grid, *run.named(rec))

    assert len(Bench.measured) == len(views(planned) - views(first))
    plan_digest = measure._digest(json.loads((run.tmp / "out/plan.json").read_text()))
    record_source = measure.file_digest(run.record / "steps/dp0.jsonl")
    for view, (_, member) in by_view(store).items():
        new = view not in views(first)
        assert member["origin"] == "plan" and (member["plan"] == plan_digest) == new
        sources = {str(population), record_source} if new else {str(population)}
        assert member["sources"] == sorted(sources)
        assert member["traces"] == [f"trace-{i}" for i in range(4)]


def test_the_planner_reads_the_engine_every_grid_input_states(run):
    rec, other = str(run.record), run.tmp / "other"
    shutil.copytree(run.record, other)
    engine = other / "steps/dp0.engine.json"
    engine.write_text(
        json.dumps(json.loads(engine.read_text()) | {"max_model_len": 1024})
    )
    definition = run.tmp / "definition.json"
    definition.write_text(json.dumps(DEFINITION))
    grid = ["--grid", rec, str(other), "--grid-definition", str(definition)]
    with pytest.raises(
        SystemExit, match="state two engines, which differ on max_model_len"
    ):
        run(*grid, *run.named(rec), *run.named(other))


def kv_blocks(line, block_size) -> int:
    """The KV blocks `line`'s rows hold, each row its own."""
    return sum(-(-context // block_size) for _, context, _ in line["rows"])


def test_points_planned_at_the_inputs_pool_are_measured_where_the_budget_is_smaller(
    run,
):
    """The plan puts decode c nodes at each B node's KV limit for the pool its
    inputs state; the device's own budget, standing in the config before the
    bench starts, gives fewer blocks than some planned points hold."""
    steps, rec = run.record / "steps", str(run.record)
    lines = read(steps)[0]
    stated = json.loads((steps / "dp0.engine.json").read_text())
    size = stated["kv_cache_block_size"]
    stated["num_kvcache_blocks"] = pool = max(
        kv_blocks(line, size) for line in lines if line["rows"][0][2]
    )
    (steps / "dp0.engine.json").write_text(json.dumps(stated))
    named = [(str(s["step"]), s) for s in lines]
    planned = plan(DEFINITION, named, {"r": stated})["lines"]
    at_limit = [
        line
        for line in planned
        if line["rows"][0][2]
        and line["rows"][0][1] == size * (pool // len(line["rows"]))
    ]
    assert at_limit
    run.config.num_kvcache_blocks = max(kv_blocks(line, size) for line in planned) - 1
    definition = run.tmp / "definition.json"
    definition.write_text(json.dumps(DEFINITION))
    store = run("--grid", rec, "--grid-definition", str(definition), *run.named(rec))
    assert set(by_view(store)) == views(planned)
    engine = json.loads((run.tmp / "out/engine.json").read_text())
    assert engine["num_kvcache_blocks"] == pool


def test_the_bench_holds_the_largest_pool_an_input_states(run):
    rec, other = str(run.record), run.tmp / "other"
    shutil.copytree(run.record, other)
    engine = other / "steps/dp0.engine.json"
    stated = json.loads(engine.read_text())
    pool = stated["num_kvcache_blocks"] + 1
    engine.write_text(json.dumps(stated | {"num_kvcache_blocks": pool}))
    run(rec, str(other), "--sample", "all", *run.named(rec), *run.named(other))
    held = json.loads((run.tmp / "out/engine.json").read_text())
    assert held["num_kvcache_blocks"] == run.config.num_kvcache_blocks == pool


def trace(i) -> str:
    """Request i's trace, a cc-trace id, which holds several numbers."""
    return f"a3acb7e89fb41af79510cf1ecf36a703490f{i}"


def mixed(run, monkeypatch, session=lambda i: 256 + i):
    """A record serving all four requests at once, so its decode steps serve
    all four traces; and the `--client-run` naming request i's trace `trace(i)`,
    which its export gives session `session(i)`, or none when that is None."""
    rec = record(run.tmp, monkeypatch, "mixed", groups=((0, 1, 2, 3),))
    requests = client_run([0, 1, 2, 3])
    for i, sent in enumerate(requests.values()):
        sent["trace"] = trace(i)
    sessions = {trace(i): session(i) for i in range(4) if session(i) is not None}
    export = run.tmp / "mixed.json"
    export.write_text(json.dumps({"requests": requests, "sessions": sessions}))
    return rec, requests, ["--client-run", str(rec), str(export)]


def test_sessions_keep_the_lines_serving_only_traces_at_sessions_a_to_b(
    run, monkeypatch
):
    rec, requests, named = mixed(run, monkeypatch)
    # A step holding a request no client run sent serves no trace.
    steps = rec / "steps/dp0.jsonl"
    line = {**json.loads(steps.read_text().splitlines()[-1]), "named": []}
    line |= {"rows": [[1, 7, True]], "rung": 1, "batch": [["unsent", 1, 7]]}
    steps.write_text(steps.read_text() + json.dumps(line) + "\n")
    store = run(str(rec), "--sample", "all", "--sessions", "257:258", *named)

    split = served(rec / "steps", [requests])[0]
    chosen = {trace(1), trace(2)}
    kept = [s for s in split if s["traces"] and set(s["traces"]) <= chosen]
    assert kept and set(by_view(store)) == views(kept) < views(split)
    # Both ends of the span are measured.
    assert {t for _, m in entries(store) for t in m["traces"]} == chosen


def test_a_trace_its_export_gives_no_session_refuses_sessions(run, monkeypatch):
    rec, _, named = mixed(run, monkeypatch, lambda i: None if i == 1 else i)
    with pytest.raises(SystemExit, match=f"served {trace(1)}, which no export gives"):
        run(str(rec), "--sample", "all", "--sessions", "0:9", *named)


def test_a_deployment_above_tensor_parallel_1_refuses_naming_the_skew_issue(run):
    run.config.tensor_parallel_size = 2
    with pytest.raises(SystemExit, match=r"tensor parallel size 2: .*\(#1188\)"):
        run(str(run.record), "--sample", "all", *run.named(run.record))
    assert Bench.started == []


def test_a_line_a_dummy_run_wrote_is_not_measured(run):
    steps = run.record / "steps/dp0.jsonl"
    dummy = {**json.loads(steps.read_text().splitlines()[-1]), "is_dummy_run": True}
    dummy |= {"rows": [[1, 7, True]], "rung": 1, "batch": [], "named": []}
    steps.write_text(steps.read_text() + json.dumps(dummy) + "\n")
    store = run(str(run.record), "--sample", "all", *run.named(run.record))
    assert set(by_view(store)) == views(read(run.record / "steps")[0][:-1])


def test_the_sampling_joins_the_key_and_reaches_the_rule(run):
    rec = str(run.record)
    for temperature in (0.0, 0.7):
        sampling = {**SAMPLING, "temperature": temperature}
        named = run.named(rec, sampling)
        store = run(rec, "--sample", "all", "--sampling", json.dumps(sampling), *named)
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
        run(str(run.record), "--sample", "all", *run.named(run.record))
    assert Bench.started == []


def test_a_batch_the_bench_refuses_is_not_published_and_is_named_after_the_rest(run):
    lines = read(run.record / "steps")[0]
    refused = Bench.refuse = sum(q for q, *_ in lines[0]["rows"])
    try:
        with pytest.raises(SystemExit, match="batches refused:\n.+a planted refusal"):
            run(str(run.record), "--sample", "all", *run.named(run.record))
    finally:
        Bench.refuse = None
    held = entries(ArtifactStore(run.tmp / "store"))
    assert refused not in {sum(q for q, *_ in m["rows"]) for _, m in held}
    assert len(Bench.measured) == len(views(lines)) > len(held)
