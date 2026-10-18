# SPDX-License-Identifier: MIT
"""`compass fit steps --backend step-grid`, run through `cli.main`, its bundle
loaded by `run/step_grid.py::load` and priced by `StepGridBackend`.

The grid is planned over one record whose steps hold every node batch of
`DEFINITION`'s grid for `BOUNDS` and rung 8: decode at B 1 and 8 and c 0, 512
and 1023, and prefill, producing output or not, at B 1 and 8, T 16 and 256 and
c 0 and 768. So the plan is those 22 points, each measured at `time` of its
node, which is linear in each coordinate, so a cell's interpolation reproduces
it. Every target is timed at that lookup times `factor`, 1.25 at a node. The
record and the measurements are taken on `test_run_step_grid.py`'s deployment
at `BOUNDS`, under the readings a run of it takes.
"""

import itertools
import json
import math
import re

import numpy as np
import pytest
from test_artifact_invalidation import moved
from test_artifact_store import stanza
from test_backend_step_linear_fit import _record
from test_bench_window import BUSY, MS, Bench
from test_bench_window import batch as bench_batch
from test_fit_samples import DEVICE, HOST, SOURCE
from test_fit_steps import fit, planted_roots  # noqa: F401
from test_memory_readings import DOCUMENT, STACK
from test_parity_served import SAMPLING
from test_run_step_grid import deployment, readings

from atom.compass.artifacts import (
    ArtifactStore,
    Axis,
    GateState,
    Topology,
    member_name,
)
from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.step_grid import StepGridBackend, equivalent_shape
from atom.compass.backends.step_grid_residual import Residual, order, raggedness
from atom.compass.bench import window
from atom.compass.bench.measure import measurement_key
from atom.compass.fit.samples import MEMBER
from atom.compass.run import step_grid
from atom.compass.spec import MachineSpec

BOUNDS = {"max_num_seqs": 8, "max_num_batched_tokens": 256, "max_model_len": 1024}
DEFINITION = {
    "prefill": {"B": [1], "T": [16], "c": [0]},
    "decode": {"c": [0, 512]},
    "margin": 0,
}
DECODE = [(b, c) for b in (1, 8) for c in (0, 512, 1023)]
PREFILL = [(b, t, c) for b in (1, 8) for t in (16, 256) for c in (0, 768)]
GRID = ("--backend", "step-grid", "--grid")
#: The bench's reply for a batch on `test_bench_window.py`'s planted device,
#: whose kept repeats a measurement publishes as they are.
BENCHED = Bench(itertools.repeat(MS), itertools.repeat(BUSY)).bench_step(bench_batch())


def batch(rows, output=True) -> dict:
    """A step-record line's view of `rows`, rung 8 for decode."""
    rung = 8 if rows[0][2] else None
    return {"rows": [list(r) for r in rows], "rung": rung, "produces_output": output}


def nodes() -> list[dict]:
    """Every node batch of the grid."""
    out = [batch([(1, c + 1, True)] * b) for b, c in DECODE]
    for output in (True, False):
        out += [batch([(t // b, c + t // b, False)] * b, output) for b, t, c in PREFILL]
    return out


def mixed(seed=1, n=20) -> list[dict]:
    """`n` decode batches of 2 to 8 rows, and `n` prefill chunks of 1 to 8 rows of
    each kind, with rows of other lengths, all inside the grid."""
    rng, out = np.random.default_rng(seed), []
    for _ in range(n):
        ks = rng.integers(1, 1025, rng.integers(2, 9))
        out.append(batch([(1, int(k), True) for k in ks]))
    for output in (True, False) * n:
        b = int(rng.integers(1, 9))
        qs, cs = rng.integers(16 // b + 1, 256 // b + 1, b), rng.integers(0, 401, b)
        out.append(batch([(int(q), int(q + c), False) for q, c in zip(qs, cs)], output))
    return out


def view(line) -> BatchView:
    rows = tuple(RequestShape(*r) for r in line["rows"])
    return BatchView(rows, line["rung"], produces_output=line["produces_output"])


def time(line) -> float:
    """The planted measurement at a node, and so the lookup anywhere."""
    b, n_q, c = map(float, equivalent_shape(view(line).requests))
    if line["rows"][0][2]:
        return 2e-3 + 1e-5 * b + 1e-7 * c + 1e-9 * b * c
    return 1e-3 + 1e-5 * b + 2e-6 * b * n_q + 1e-6 * c + 1e-9 * b * n_q * c


def factor(line) -> float:
    rows = view(line).requests
    return 1.25 * math.exp(0.2 * float(order(rows)) - 0.1 * float(raggedness(rows) - 1))


@pytest.fixture
def run(tmp_path):
    """The deployment, its readings and the inputs' configuration."""
    r = deployment(tmp_path, **BOUNDS, capture_sizes=[8])
    here = readings(r)
    configuration = {
        "model": here.reading(Axis.MODEL).value,
        "device": "MI355X gfx950",
        "source_root": "{0.kind}-{0.value}".format(here.reading(Axis.ATOM_SRC)),
    }
    return r, here, configuration


def record(tmp_path, run, lines, name="a", trace="trace-300", session=300):
    """`tmp_path/name`, a real run's record of `lines`, each timed at its lookup
    times `factor`, with the export `client.json` of the client run sending it
    from `trace`, which the export gives `session`."""
    r, _, configuration = run
    directory = tmp_path / name
    path = directory / "steps" / "dp0.jsonl"
    path.parent.mkdir(parents=True)
    _record(path, [(x["rows"], x["rung"], time(x) * factor(x)) for x in lines])
    written, sent = [json.loads(x) for x in path.read_text().splitlines()], {}
    for line, x in zip(written, lines):
        line |= {"produces_output": x["produces_output"]}
        line["named"] = [[q[0], f"r{q[0]}"] for q in line["batch"]]
        sent |= {n: {"trace": trace, "sampling": SAMPLING} for _, n in line["named"]}
    path.write_text("".join(json.dumps(x) + "\n" for x in written))
    export = {"requests": sent, "sessions": {trace: session}}
    (directory / "client.json").write_text(json.dumps(export))
    engine = {**step_grid.keyed(r.config), **BOUNDS, "num_kvcache_blocks": 1024}
    (path.parent / "dp0.engine.json").write_text(json.dumps(engine))
    kind, revision = configuration["source_root"].split("-", 1)
    run_json = {
        "format_version": 2,
        "argv": ["compass", "record", "--out", str(directory)],
        "source": {**SOURCE, "revision": revision, "revision_kind": kind},
        "host_key": {
            **HOST,
            "model": configuration["model"],
            "engine": step_grid.keyed(r.config),
        },
        "device": [DEVICE],
        "stack": STACK,
        "ptl": "disabled",
        "profiled": False,
    }
    (directory / "run.json").write_text(json.dumps(run_json))
    return directory


def measured(tmp_path, run, line, origin="plan", seconds=None, conditions=None):
    """`line`'s step measurement as `measure steps` publishes one, its token."""
    r, here, configuration = run
    engine = step_grid.keyed(r.config)
    key = measurement_key(configuration, engine, SAMPLING, line)
    member = {
        **line,
        "seconds": time(line) if seconds is None else seconds,
        "statistic": "median",
        "repeats": BENCHED["repeats"],
        "origin": origin,
        "sources": ["planted"],
        "traces": ["trace-7"],
        "plan": "sha256:another-plan",
    }
    ArtifactStore(tmp_path / "store").publish(
        key,
        provenance=stanza("compass measure steps"),
        topology=Topology(),
        conditions=conditions or here,
        gates=GateState.of(),
        members={member_name(*MEMBER): json.dumps(member).encode()},
    )
    return key.token()


def fitted(tmp_path, *args, grid="a"):
    """`fit steps --backend step-grid` over each `tmp_path/<name>` of the names
    `grid` lists, and `args`; the readable copy it writes."""
    path = tmp_path / "definition.json"
    path.write_text(json.dumps(DEFINITION))
    inputs = [tmp_path / name for name in grid.split()]
    fit(tmp_path, "fit steps", *GRID, *inputs, "--grid-definition", path, *args)
    return json.loads((tmp_path / "out/step_grid_bundle.json").read_text())


def clients(directory):
    return ["--client-run", directory, directory / "client.json"]


def loaded(tmp_path, run, copy):
    spec = {"store": str(tmp_path / "store"), **copy["key"]}
    return step_grid.load(spec, run[0].config, MachineSpec.from_mapping(DOCUMENT))[0]


def test_a_fitted_bundle_prices_each_grid_node_at_its_measured_time_times_the_factor(
    tmp_path, run
):
    a = record(tmp_path, run, nodes() * 4 + mixed())
    for line in nodes():
        measured(tmp_path, run, line)
    # A measurement no plan places is no point.
    measured(tmp_path, run, batch([(1, 300, True)] * 4))

    copy = fitted(tmp_path, a, *clients(a))
    backend = loaded(tmp_path, run, copy)

    residual = Residual(copy["params"]["residual"])
    for line in nodes():
        priced = backend.price(view(line)).seconds
        assert priced == time(line) * residual.factor(view(line))[0]
        assert priced == pytest.approx(time(line) * 1.25, rel=1e-3)
    grid = copy["params"]["grid"]
    points = grid["decode"]["points"] + [
        p for kind in ("output", "no_output") for p in grid["prefill"][kind]
    ]
    assert len(points) == len(nodes()) == 22
    # A point counts its measurement's kept repeats.
    assert {(p["statistic"], p["repeats"]) for p in points} == {
        ("median", window.REPEATS)
    }
    assert (grid["unit"], grid["value"]) == ("s", "bench window")
    assert copy["params"]["definition"] == DEFINITION
    assert copy["params"]["bounds"] == BOUNDS
    assert copy["params"]["sources"] == ["record"]


def test_a_fitted_bundle_validates_its_held_out_batches_and_not_a_batch_outside_them(
    tmp_path, run
):
    """Every target lies in the box whose corners are the prefill nodes, so a
    prefill node is inside the hull of the held-out batches iff one of them has
    its shape. A held-out batch outside the residual's hull, which is of the
    batches fitted, is refused."""
    lines = nodes() * 4 + mixed()
    a = record(tmp_path, run, lines)
    for line in nodes():
        measured(tmp_path, run, line)

    copy = fitted(tmp_path, a, *clients(a))
    backend = loaded(tmp_path, run, copy)

    def validated(line, priced_by=backend):
        try:
            (term,) = priced_by.price(view(line)).terms
        except CostRefused:
            return None
        return term.provenance.validated

    def shape(line):
        return line["produces_output"], equivalent_shape(view(line).requests)

    report = copy["report"]
    assert (report["unpriced"], report["cut"]) == ([], {})  # triple i is lines[i]
    held = [lines[i] for k in report["kinds"].values() for i in k["held_out"]]
    outcomes = [validated(x) for x in held]
    assert True in outcomes and set(outcomes) <= {True, None}
    corners = [x for x in nodes() if not x["rows"][0][2]]
    inside = [shape(x) in set(map(shape, held)) for x in corners]
    assert [validated(x) for x in corners] == inside
    assert any(inside) and not all(inside)
    # A bundle that records no held-out batches validates none.
    residual = copy["params"]["residual"]
    for fitted_kind in residual["kinds"].values():
        del fitted_kind["held_out"]
    bare = StepGridBackend(backend.grid, Residual(residual), "bare")
    assert [validated(x, bare) for x in held] == [o and False for o in outcomes]


def test_a_planned_point_the_store_does_not_hold_refuses_naming_each(tmp_path, run):
    a = record(tmp_path, run, nodes() * 4 + mixed())
    held, lost = nodes()[2:], nodes()[:2]
    for line in held:
        measured(tmp_path, run, line)
    engine = step_grid.keyed(run[0].config)
    named = [str(measurement_key(run[2], engine, SAMPLING, x)) for x in lost]

    with pytest.raises(SystemExit) as refused:
        fitted(tmp_path, a, *clients(a))
    assert "2 of the 22 planned points are not measured" in str(refused.value)
    assert all(key in str(refused.value) for key in named)


def test_a_grid_measured_under_sessions_fits_given_the_same_sessions(tmp_path, run):
    """`a` serves trace 3 and `b` trace 7, so `measure steps --grid a b
    --sessions 0:4` plans over `a`'s lines alone: every node but decode's at c
    1023, which only `b`'s lines reach."""
    top = [x for x in nodes() if x["rows"][0][2] and x["rows"][0][1] == 1024]
    rest = [x for x in nodes() if x not in top]
    a = record(tmp_path, run, rest, trace="trace-3", session=3)
    b = record(tmp_path, run, top, name="b", trace="trace-7", session=7)
    for line in nodes():
        if line not in top:
            measured(tmp_path, run, line)
    t = record(tmp_path, run, mixed(), name="t")
    named = [*clients(a), *clients(b), *clients(t)]

    with pytest.raises(SystemExit, match="2 of the 22 planned points are not measured"):
        fitted(tmp_path, t, *named, grid="a b")
    copy = fitted(tmp_path, t, "--sessions", "0:4", *named, grid="a b")

    grid = copy["params"]["grid"]
    decode = [p["at"] for p in grid["decode"]["points"]]
    assert sorted(decode) == sorted([8, n, c] for n in (1, 8) for c in (0, 512))
    assert sum(len(grid["prefill"][k]) for k in ("output", "no_output")) == 16


@pytest.mark.parametrize("sources", [["record"], ["replay"], ["record", "replay"]])
def test_the_residual_is_fitted_on_records_replays_or_both_and_reported_by_source(
    tmp_path, run, sources
):
    record(tmp_path, run, nodes() + mixed(seed=2), name="grid")
    for line in nodes():
        measured(tmp_path, run, line)
    targets, named = [], []
    if "record" in sources:
        a = record(tmp_path, run, mixed(seed=3))
        targets, named = [a], clients(a)
    if "replay" in sources:
        targets += [
            measured(tmp_path, run, x, "replay", time(x) * factor(x)) for x in mixed()
        ]

    copy = fitted(tmp_path, *targets, *named, grid="grid")

    assert copy["params"]["sources"] == sources
    for report in copy["report"]["kinds"].values():
        counts = [(s["fitted"], s["held_out"]) for s in report["sources"].values()]
        assert list(report["sources"]) == sources
        held = len(report["held_out"])
        assert (
            sum(map(sum, counts)) == (20 * len(sources))
            and sum(h for _, h in counts) == held
        )
        for spread in report["sources"].values():
            assert abs(spread["centre"]) < 1e-3 and spread["held_out_error"] < 1e-3
    *runs, planned = copy["report"]["provenance"]
    assert sorted({r["source"] for r in runs}) == sources
    assert (planned["source"], planned["traces"]) == ("plan", ["trace-7"])


def test_a_planned_point_named_as_a_target_refuses(tmp_path, run):
    a = record(tmp_path, run, nodes() * 4 + mixed())
    tokens = [measured(tmp_path, run, line) for line in nodes()]

    with pytest.raises(SystemExit, match="measured a planned point"):
        fitted(tmp_path, a, tokens[0], *clients(a))


@pytest.mark.parametrize("at", ["point", "replay"])
def test_a_measurement_taken_under_other_readings_refuses_naming_the_cell(
    tmp_path, run, at
):
    a = record(tmp_path, run, nodes() * 4 + mixed())
    other = moved(run[1], Axis.TORCH)
    for i, line in enumerate(nodes()):
        measured(
            tmp_path, run, line, conditions=other if at == "point" and not i else None
        )
    extra = mixed(seed=4)[0]
    replay = measured(
        tmp_path, run, extra, "replay", time(extra), other if at == "replay" else None
    )

    with pytest.raises(SystemExit, match=re.escape("`step_measurement` x `torch`")):
        fitted(tmp_path, a, replay, *clients(a))


def test_a_target_the_grid_refuses_is_named_and_a_cut_one_counted(tmp_path, run):
    outside = batch([(8, 8, False)])  # T = 8, below the T nodes
    both = batch([(8, 8, False), (1, 40, True)])
    a = record(tmp_path, run, nodes() * 4 + mixed() + [outside, both])
    for line in nodes():
        measured(tmp_path, run, line)

    report = fitted(tmp_path, a, *clients(a))["report"]

    [step] = report["unpriced"]
    assert step["step"].endswith(f" dp0 step {len(nodes()) * 4 + 60}")
    assert "T=8 is outside the prefill output grid's nodes 16..256" in step["reason"]
    assert report["cut"] == {"mixed": 1}


@pytest.mark.parametrize(
    "args, named",
    [
        (["--form", "base"], "--form is read by step-linear, not step-grid"),
        (["--grid-definition"], "step-grid plans over --grid INPUT..."),
    ],
)
def test_a_flag_step_grid_does_not_read_or_lacks_refuses(tmp_path, run, args, named):
    a = record(tmp_path, run, nodes())
    if args == ["--grid-definition"]:
        argv = ["--backend", "step-grid", a, "--grid", a, *clients(a)]
    else:
        argv = [*GRID, a, "--grid-definition", tmp_path, *args, a, *clients(a)]

    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit steps", *argv)
    assert not (tmp_path / "store" / "step_grid_bundle").exists()
