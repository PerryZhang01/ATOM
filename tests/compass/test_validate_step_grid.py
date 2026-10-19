# SPDX-License-Identifier: MIT
"""Grading a step-grid cell: `python -m atom.compass.validate` on a cell whose
simulated run is `step_grid/`, as `compass run --step-grid` writes it, or as
`cctraces_sim.sh` does.

The cell is `test_validate_main`'s, its simulated run moved to `step_grid/`.
Its run file names a bundle published as `test_run_step_grid` publishes one,
under that file's deployment read in another ATOM tree, and its run summary
records the bundle's digest. The bundle's grid measures each real step
at the planted law's price on that step's own node, its residual's factor is
one, and its held-out hull holds every step, so the grid prices each real step
as the law does.
"""

import hashlib
import json
import re

import pytest
from test_artifact_invalidation import moved
from test_backend_step_grid import fitted, linear, point
from test_memory_readings import DOCUMENT
from test_run_step_grid import BOUNDS, deployment, publish, readings
from test_validate_cost import LAW, _record, _steps
from test_validate_main import _as_recorded, _graded, _rows

from atom.compass.artifacts import ArtifactRefusal, Axis
from atom.compass.backends import (
    StepGrid,
    StepGridBackend,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_grid import coordinates
from atom.compass.backends.step_grid_fit import VALUE
from atom.compass.backends.step_grid_residual import Residual
from atom.compass.drive import run as drive_run
from atom.compass.run import step_grid
from atom.compass.validate.__main__ import main
from atom.compass.validate.cost import cost_accuracy

ATOM_ARGS = ["--model", "planted", "--max-model-len", "2048"]


def _box(n):
    """Facets bounding each of `n` coordinates to [0, 1e6]."""
    return [
        [s * (i == j) for j in range(n)] + [-1e6 * (s > 0)]
        for i in range(n)
        for s in (1, -1)
    ]


def _bundle(steps, missing=()):
    """A replay bundle measuring each of `steps` at the law's price on nodes at
    its coordinates, those numbered in `missing` on nodes with no point."""
    law = StepLinearBackend(StepLinearLaw(LAW))
    found = {True: [], False: []}  # by whether a step is prefill
    for i, (rows, rung) in enumerate(steps):
        view = BatchView(
            tuple(RequestShape(*r) for r in rows),
            capture_rung=rung,
            produces_output=True,
        )
        at = [int(x) for x in coordinates(view.requests, rung is None)]
        found[rung is None].append((i, at, law.price(view).seconds))

    def nodes(prefill, axis):
        return sorted({at[axis] for _, at, _ in found[prefill]})

    def points(prefill, *rung):
        return [
            point(*rung, *at, time=t) for i, at, t in found[prefill] if i not in missing
        ]

    residual = fitted({"unit": linear()}, {"unit": 1.0})
    for kind, fit in residual["kinds"].items():
        fit["held_out"] = {"facets": _box(2 if kind == "decode" else 3)}
    grid = {
        "graph_mode": "replay",
        "scope": "forward",
        "unit": "s",
        "value": VALUE,
        "prefill": {
            "nodes": {a: nodes(True, i) for i, a in enumerate("BTc")},
            "output": points(True),
            "no_output": [],
        },
        "decode": {
            "nodes": {"B": {"1": nodes(False, 0)}, "c": {"1": nodes(False, 1)}},
            "points": points(False, 1),
        },
    }
    return {"grid": grid, "residual": residual, "bounds": BOUNDS}


def _as_grid(tmp_path, monkeypatch, missing=(), digest=None, parallel=None, **key):
    """`tmp_path`'s cell with its simulated run moved to `step_grid/` over
    `_bundle`, published under the deployment's readings in another ATOM tree,
    and with `parallel` its entry also restates that key field, which this
    tree's keys no longer have, as one published before they dropped it does:
    a `compass run --step-grid` directory whose ATOM arguments resolve to the
    deployment, or a `cctraces_sim.sh` one, whose run.json is the run file. Its
    run file names the bundle at `key`'s fields over the entry's, and its
    summary records `digest`, the bundle's when None. The configs a grading
    resolves are returned."""
    r = deployment(tmp_path)
    there = moved(readings(r), Axis.ATOM_SRC)
    entry = publish(tmp_path, r, member=_bundle(_steps(), missing), conditions=there)
    path = entry.directory / "entry.json"
    document = json.loads(path.read_bytes())
    if parallel:
        document["key"]["parallel"] = parallel
        path.chmod(0o644)
        path.write_text(json.dumps(document))
    resolved = []
    monkeypatch.setattr(
        drive_run, "_config", lambda args: resolved.append(args) or r.config
    )
    sim = (tmp_path / "c4" / "step_linear").rename(tmp_path / "c4" / "step_grid")
    run = json.loads((sim / "run.json").read_text())
    grid = {"step_grid": {"store": str(tmp_path / "store"), **document["key"], **key}}
    if "format_version" in run:
        run["argv"] = ["-c", "run", "--step-grid", "planted", "--", *ATOM_ARGS]
        run["run_file"] = grid | {"host": {"laws": {}}, "machine": DOCUMENT}
        run["coverage"] = {"inputs": ["planted"], "uncovered": []}
    else:
        run = {"workload": run["workload"], "host": run["host"], **grid}
    (sim / "run.json").write_text(json.dumps(run))
    summary = json.loads((sim / "summary.json").read_text())
    digest = digest or f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    summary["step_grid"] = {"engine": {"key": document["key"], "digest": digest}}
    (sim / "summary.json").write_text(json.dumps(summary))
    return resolved


def _renamed(rows):
    named = json.dumps(list(rows.items())).replace("step_linear", "step_grid")
    return {tuple(k): v for k, v in json.loads(named)}


@pytest.mark.parametrize("parallel", [None, "tp1-dp1"])
def test_a_step_grid_cell_grades_every_row_the_step_linear_cell_does(
    capsys, tmp_path, monkeypatch, parallel
):
    _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")
    linear_rows = _graded(capsys, tmp_path / "c4")
    resolved = _as_grid(tmp_path, monkeypatch, parallel=parallel)

    rows = _graded(capsys, tmp_path / "c4")
    assert rows == _renamed(linear_rows)
    # Read by key and digest: no config resolved, another tree's bundle read.
    assert resolved == []
    assert rows["cost real0", "decode validated"][0].startswith("10 steps")
    assert rows["cost real0", "prefill validated"][0].startswith("9 steps")
    assert rows["summary step_grid", "refusals"][1] == "none"


def test_a_real_step_the_grid_declines_is_counted_apart_as_a_fallback(
    capsys, tmp_path, monkeypatch
):
    _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")
    # Decode step 15's node holds no point: the grid refuses that step alone.
    _as_grid(tmp_path, monkeypatch, missing={15})

    rows = _graded(capsys, tmp_path / "c4")
    for real in ("real0", "real1", "real2"):
        steps, value = rows[f"cost {real}", "fallback decode rung 1"][0].split(", ")
        assert steps == "1 steps" and value.startswith("real ")
        assert rows[f"cost {real}", "decode validated"][0].startswith("9 steps")


def test_a_step_the_grid_declines_is_a_fallback_and_one_a_law_declines_refused(
    tmp_path,
):
    """A run prices a step the grid declines by the step roofline below it; a
    graded run has nothing below a law."""
    member = _bundle(_steps(), missing={15})
    grid = StepGridBackend(StepGrid(member["grid"]), Residual(member["residual"]), "b")
    record = _record(tmp_path / "steps", _steps())

    by_grid = cost_accuracy(record, grid)
    by_law = cost_accuracy(record, {**LAW, "decode": {}})

    assert by_grid["backend"] == grid.describe()
    assert by_law["backend"] == "planted"
    assert by_grid["refused"] == by_law["fallback"] == {}
    assert by_grid["fallback"]["decode rung 1"]["steps"] == 1
    assert by_law["refused"]["decode rung 1"]["steps"] == 10


def test_a_cell_cctraces_sim_wrote_grades_with_no_argv_as_its_step_linear_cell(
    capsys, tmp_path, monkeypatch
):
    linear_rows = _rows(capsys, tmp_path)
    _as_grid(tmp_path, monkeypatch)

    assert _graded(capsys, tmp_path / "c4") == _renamed(linear_rows)


#: The refusals: the summary's digest is not the entry's, the run file's key has
#: no entry, and the entry's member changed after it was published.
DIFFERS = r"holds 1 entries keyed .*\. the run recorded the one digesting sha256:0$"
MISSING = (
    r'holds 0 entries keyed step_grid_bundle {.*"grid": "other".*}, digesting \[\]'
)
CHANGED = r"`params\.\S+\.json` digests sha256:\w+, and this entry records sha256:\w+"


@pytest.mark.parametrize(
    "digest, key, member, refused",
    [
        ("sha256:0", {}, False, DIFFERS),
        (None, {"grid": "other"}, False, MISSING),
        (None, {}, True, CHANGED),
    ],
)
def test_a_bundle_other_than_the_one_the_run_priced_with_refuses_the_cell(
    capsys, tmp_path, monkeypatch, digest, key, member, refused
):
    _rows(capsys, tmp_path)
    _as_grid(tmp_path, monkeypatch, digest=digest, **key)
    if member:
        (path,) = (tmp_path / "store" / "step_grid_bundle").glob("*/params.*")
        path.chmod(0o644)
        path.write_text(json.dumps(_bundle(_steps(), missing={15})))

    with pytest.raises(ArtifactRefusal, match=refused):
        main([str(tmp_path / "c4")])


def _strays(tmp_path):
    """A malformed neighbour beside the bundle, and the staging directory a
    publisher killed before its rename leaves, holding a half-written entry."""
    folder = tmp_path / "store" / "step_grid_bundle"
    (entry,) = folder.iterdir()
    for stray, document in (("stray", "{not json"), (f".{entry.name}.k1", '{"key"')):
        (folder / stray).mkdir()
        (folder / stray / "entry.json").write_text(document)
    return folder / "stray", folder / f".{entry.name}.k1"


def test_a_malformed_neighbour_or_a_staging_directory_stops_no_grade(
    capsys, tmp_path, monkeypatch
):
    linear_rows = _rows(capsys, tmp_path)
    _as_grid(tmp_path, monkeypatch)
    _strays(tmp_path)

    assert _graded(capsys, tmp_path / "c4") == _renamed(linear_rows)


def test_a_missing_bundle_names_each_neighbour_that_is_not_an_entry(
    capsys, tmp_path, monkeypatch
):
    _rows(capsys, tmp_path)
    _as_grid(tmp_path, monkeypatch, digest="sha256:0")
    stray, staging = _strays(tmp_path)

    named = re.escape(f"; {stray / 'entry.json'} is not an entry document: ")
    with pytest.raises(ArtifactRefusal, match=f"{named}JSONDecodeError") as refused:
        main([str(tmp_path / "c4")])
    assert staging.name not in str(refused.value)


def test_a_member_the_bundle_does_not_hold_refuses_naming_what_it_holds(
    capsys, tmp_path, monkeypatch
):
    """A `member_name` change: this tree asks for a name the bundle never had."""
    _rows(capsys, tmp_path)
    _as_grid(tmp_path, monkeypatch, parallel="tp1-dp1")
    monkeypatch.setattr(step_grid, "MEMBER", ("grid", *step_grid.MEMBER[1:]))

    held = r"`grid\.\S+\.json` is not in step_grid_bundle\(.*parallel='tp1-dp1'.*\)"
    with pytest.raises(ArtifactRefusal, match=rf"{held}\. this entry holds params\."):
        main([str(tmp_path / "c4")])


def test_a_run_summary_recording_no_bundle_refuses_the_cell(
    capsys, tmp_path, monkeypatch
):
    _rows(capsys, tmp_path)
    _as_grid(tmp_path, monkeypatch)
    path = tmp_path / "c4" / "step_grid" / "summary.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"step_grid": {}}))

    with pytest.raises(ValueError, match=re.escape("bundle digests [], and grading")):
        main([str(tmp_path / "c4")])


@pytest.mark.parametrize("both", [True, False])
def test_a_cell_of_two_simulated_runs_or_none_is_refused(capsys, tmp_path, both):
    _rows(capsys, tmp_path)
    cell = tmp_path / "c4"
    if both:
        (cell / "step_grid").mkdir()
    else:
        (cell / "step_linear").rename(cell / "real3")
    held = "['step_linear', 'step_grid']" if both else "[]"
    with pytest.raises(ValueError, match=re.escape(f"holds {held} of step_linear/")):
        main([str(cell)])


def test_a_real_step_past_the_residuals_hull_is_priced_unvalidated_and_counted(
    tmp_path,
):
    """Decode step i reads N_KV = 2000 + 5000 i; a decode hull bounding
    query_context to 10000 / 2^20 leaves steps 2 to 9 past it."""
    member = _bundle(_steps())
    member["residual"]["kinds"]["decode"]["hull"] = {
        "facets": [[0, 1, 0, 0, -10000 / 2**20]]
    }
    grid, residual = StepGrid(member["grid"]), Residual(member["residual"])

    report = cost_accuracy(
        _record(tmp_path / "steps", _steps()), StepGridBackend(grid, residual, "b")
    )

    assert (report["extrapolated"], report["fallback"]) == (8, {})
    assert report["unvalidated"]["decode"]["steps"] == 8
    assert report["validated"]["decode"]["steps"] == 2
