# SPDX-License-Identifier: MIT
"""Tier a's fit: step records written from a known law, and the law it gives back.

Each record is the JSON lines a real runner writes, its `t_enter_ns` laid out
so each step's period is the known law's price. The fitted law is read by
`CoarseLaw` and priced by `CoarseBackend.estimate`.
"""

import json

import numpy as np
import pytest

from atom.compass.backends import (
    BatchView,
    CoarseBackend,
    CoarseLaw,
    CostRefused,
    RequestShape,
    Species,
)
from atom.compass.backends.coarse_fit import fit, held_out, main, samples

PREFILL = {"step": 0.02, "tokens": 6e-5, "query_square": 2e-10, "query_cached": 3e-10}
DECODE = {
    "eager": {"step": 0.03, "context": 2e-7, "graph_padding": 0.0},
    "2": {"step": 0.021, "context": 1e-7, "graph_padding": 4e-8},
    "4": {"step": 0.022, "context": 1.2e-7, "graph_padding": 3e-8},
}


def _price(rows, rung, history=0.0):
    view = BatchView(tuple(RequestShape(*r) for r in rows), capture_rung=rung)
    if view.prefill:
        c = PREFILL
        q = np.array([r[0] for r in rows])
        n = np.array([r[1] for r in rows])
        return (
            c["step"]
            + c["tokens"] * q.sum()
            + c["query_square"] * (q * q).sum()
            + c["query_cached"] * (q * (n - q)).sum()
            + history * (n - q).sum()
        )
    c = DECODE["eager" if rung is None else str(rung)]
    return (
        c["step"]
        + c["context"] * sum(r[1] for r in rows)
        + c["graph_padding"] * view.graph_padding
    )


def _steps(seed, n=240, cached=True, history=0.0):
    """Alternating prefill and decode steps of a random shape, each with its price."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        if i % 2:
            rung = (None, 2, 4)[i % 3]
            width = 1 if rung is None else int(rng.integers(1, rung + 1))
            rows = [(1, int(rng.integers(1, 60000)), True) for _ in range(width)]
        else:
            rung = None
            rows = []
            for _ in range(int(rng.integers(1, 4))):
                q = int(rng.integers(16, 8192))
                rows.append(
                    (q, q + (int(rng.integers(0, 60000)) if cached else 0), False)
                )
        out.append((rows, rung, _price(rows, rung, history)))
    return out


def _record(path, steps, idle_after=(), stretch=None):
    """Write `steps` as a real record. Steps in `idle_after` share no request with the next."""
    stretch = stretch or {}
    t, lines, owner = 0, [], 0
    for i, (rows, rung, seconds) in enumerate(steps):
        if i - 1 in idle_after:
            owner += 1
        ids = [owner] + [10_000 + i * 10 + j for j in range(len(rows) - 1)]
        lines.append(
            {
                "step": i,
                "dp_rank": 0,
                "is_dummy_run": False,
                "produces_output": True,
                "batch": [[r, q, n] for r, (q, n, _) in zip(ids, rows)],
                "named": [],
                "rows": [list(r) for r in rows],
                "rung": rung,
                "t_enter_ns": t,
                "t_exit_ns": t + 1,
            }
        )
        t += round(seconds * stretch.get(i, 1.0) * 1e9)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _view(rows, rung=None):
    return BatchView(tuple(RequestShape(*r) for r in rows), capture_rung=rung)


def test_the_fit_recovers_its_law_and_drops_planted_outliers_by_name(tmp_path):
    steps = _steps(0)
    outliers = {10: 3.0, 11: 0.2, 40: 2.5, 77: 4.0}
    law = fit([_record(tmp_path / "dp0.jsonl", steps, stretch=outliers)])

    assert law["prefill"]["fit"]["dropped"] + sum(
        v["fit"]["dropped"] for v in law["decode"].values()
    ) == len(outliers)
    got = law["prefill"]["coefficients"]
    assert got == pytest.approx(PREFILL, rel=1e-6)
    for rung, coefficients in DECODE.items():
        expected = {k: v for k, v in coefficients.items() if rung != "eager" or v}
        assert {
            k: v
            for k, v in law["decode"][rung]["coefficients"].items()
            if k in expected
        } == pytest.approx(expected, rel=1e-6)

    backend = CoarseBackend(CoarseLaw(law))
    for i, (rows, rung, seconds) in enumerate(steps[:-1]):
        if i in outliers:
            continue
        cost = backend.estimate(_view(rows, rung))
        assert cost.seconds == pytest.approx(seconds, rel=1e-6)
        assert {t.provenance.species for t in cost.terms} == {Species.FITTED}
    far = backend.estimate(_view([(1, 10**6, True), (1, 10**6, True)], 2))
    assert {t.provenance.species for t in far.terms} == {Species.EXTRAPOLATED}


def test_a_feature_with_no_variance_returns_zero_named(tmp_path):
    law = fit([_record(tmp_path / "dp0.jsonl", _steps(1, cached=False))])

    prefill = law["prefill"]
    assert prefill["fit"]["zero"] == ["query_cached"]
    assert prefill["coefficients"]["query_cached"] == 0.0
    assert prefill["hull"]["equal"] == {"query_cached": 0}
    eager = law["decode"]["eager"]
    assert eager["fit"]["zero"] == ["graph_padding"]
    for vector in [prefill, *law["decode"].values()]:
        assert set(vector["fit"]) == {"samples", "warm", "dropped", "condition", "zero"}
        assert vector["fit"]["samples"] > 0 and vector["fit"]["condition"] >= 1.0
    cost = CoarseBackend(CoarseLaw(law)).estimate(_view([(100, 100, False)]))
    assert cost.terms[-1].seconds == 0.0


def test_warm_steps_and_the_period_holding_an_idle_gap_are_not_fitted(tmp_path):
    steps = _steps(2)
    # Two leading compiles, then a gap after step 50: its period is cut, step 51's
    # starts after the gap and is fitted. A later compile (step 90) is warm too;
    # a later step at four times its price (step 120) is dropped, not warm.
    stretch = {0: 40.0, 1: 30.0, 50: 9.0, 90: 20.0, 120: 4.0}
    record = _record(tmp_path / "dp0.jsonl", steps, {50}, stretch)
    rows, cut = samples(record)

    assert cut == {"idle": 1, "mixed": 0, "last": 1}
    assert [(r[3], r[2]) for r in rows if r[3] in (50, 51)] == [
        (51, pytest.approx(steps[51][2], rel=1e-6))
    ]
    law = fit([record])
    assert law["warm_steps"] == [[0, 90, 1]]
    assert law["cut"]["warm"] == 3 and law["cut"]["warm_s"] == pytest.approx(
        40 * steps[0][2] + 30 * steps[1][2] + 20 * steps[90][2], rel=1e-6
    )
    assert (law["prefill"]["fit"]["warm"], law["prefill"]["fit"]["dropped"]) == (2, 1)
    assert law["decode"]["2"]["fit"]["warm"] == 1
    assert law["prefill"]["coefficients"] == pytest.approx(PREFILL, rel=1e-6)


def test_three_forms_are_scored_on_a_record_held_out_of_their_fit(tmp_path):
    # Prefill pays a history term the law's form cannot see.
    records = [
        _record(tmp_path / f"dp0-{s}.jsonl", _steps(s, history=2e-7)) for s in (3, 4, 5)
    ]
    scores = {form: held_out(records, form) for form in ("law", "history", "shared")}

    history = np.abs(
        np.concatenate([e for b, e in scores["history"].items() if "prefill" in b])
    )
    plain = np.abs(
        np.concatenate([e for b, e in scores["law"].items() if "prefill" in b])
    )
    assert history.max() < 1e-6 < np.median(plain)
    assert {b for b in scores["law"]} == {b for b in scores["shared"]}
    assert (
        "decode rung 2" in scores["law"] and "prefill tokens 2048-8192" in scores["law"]
    )

    shared = fit(records, "shared")
    slopes = {v["coefficients"]["context"] for v in shared["decode"].values()}
    assert len(slopes) == 1
    CoarseBackend(CoarseLaw(shared)).estimate(_view([(1, 900, True)], 2))
    rows, _, seconds = _steps(3, history=2e-7)[4]
    cost = CoarseBackend(CoarseLaw(fit(records, "history"))).estimate(_view(rows))
    assert cost.seconds == pytest.approx(seconds, rel=1e-6)
    assert cost.terms[-1].name == "prefill.history"


def test_the_command_writes_a_law_a_run_file_can_carry(tmp_path, capsys):
    records = [str(_record(tmp_path / f"dp0-{s}.jsonl", _steps(s))) for s in (6, 7)]
    main([*records, "--out", str(tmp_path / "law.json")])

    law = CoarseLaw(json.loads((tmp_path / "law.json").read_text()))
    assert "2 real records" in law.provenance
    out = capsys.readouterr().out
    for form in ("law", "history", "shared"):
        assert f"held out, form {form}" in out


def test_a_rung_too_thin_to_fit_is_named_and_its_steps_refused(tmp_path):
    steps = _steps(8)
    fours = [i for i, s in enumerate(steps) if s[1] == 4][2:]
    steps = [s for i, s in enumerate(steps) if i not in fours]
    law = fit([_record(tmp_path / "dp0.jsonl", steps)])

    assert law["unfitted"] == {"4": "2 steps for 3 coefficients"}
    assert "4" not in law["decode"]
    with pytest.raises(CostRefused, match="no decode vector for rung 4"):
        CoarseBackend(CoarseLaw(law)).estimate(_view([(1, 900, True)], 4))


def test_the_outlier_scale_is_read_over_every_step_so_it_does_not_shrink(tmp_path):
    # Half the prefill steps scatter by 1%, half by 15%. A scale read over the
    # kept steps alone shrinks each pass toward the 1% half and drops 40%.
    steps = _steps(9, n=400)
    rng = np.random.default_rng(9)
    noise = {
        i: 1 + rng.uniform(-0.15, 0.15) if i % 4 else 1 + rng.uniform(-0.01, 0.01)
        for i in range(400)
    }
    law = fit([_record(tmp_path / "dp0.jsonl", steps, stretch=noise)])

    fitted = law["prefill"]["fit"]
    assert (
        fitted["dropped"] + fitted["warm"] < (fitted["samples"] + fitted["dropped"]) / 3
    )
