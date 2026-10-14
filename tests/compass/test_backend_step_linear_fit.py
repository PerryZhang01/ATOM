# SPDX-License-Identifier: MIT
"""The step linear fit: step records written from a known law, and the law it
gives back.

Each record is the JSON lines a real runner writes, each step's device busy
time (`d_busy_ns`) the known law's price. The fitted law is read by
`StepLinearLaw` and priced by `StepLinearBackend.price`.
"""

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from atom.compass.backends import (
    BatchView,
    CostRefused,
    Method,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear_fit import (
    _solve,
    fit,
    held_out,
    main,
    mark_held_out,
    samples,
)
from atom.compass.validate.cost import cost_accuracy

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


def _record(path, steps, stretch=None):
    """Write `steps` as a real record whose device busy time is each step's price.

    The host's token copy wait ends as the device finishes, so no step is
    launch-exposed.
    """
    stretch = stretch or {}
    t, lines = 0, []
    for i, (rows, rung, seconds) in enumerate(steps):
        busy = round(seconds * stretch.get(i, 1.0) * 1e9)
        ids = [0] + [10_000 + i * 10 + j for j in range(len(rows) - 1)]
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
                "t_launch_ns": [t, t],
                "t_recv_ns": [t, t],
                "d_busy_ns": busy,
                "t_exit_ns": t + busy,
            }
        )
        t += busy
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

    backend = StepLinearBackend(StepLinearLaw(law))
    for i, (rows, rung, seconds) in enumerate(steps[:-1]):
        if i in outliers:
            continue
        cost = backend.price(_view(rows, rung))
        assert cost.seconds == pytest.approx(seconds, rel=1e-6)
        assert {t.provenance.method for t in cost.terms} == {Method.ESTIMATED}
    far = backend.price(_view([(1, 10**6, True), (1, 10**6, True)], 2))
    assert {t.provenance.method for t in far.terms} == {Method.ESTIMATED}
    assert {t.provenance.source for t in far.terms} == {"step-linear-extrapolated"}


def test_a_feature_with_no_variance_returns_zero_named(tmp_path):
    law = fit([_record(tmp_path / "dp0.jsonl", _steps(1, cached=False))])

    prefill = law["prefill"]
    assert prefill["fit"]["zero"] == ["query_cached"]
    assert prefill["coefficients"]["query_cached"] == 0.0
    assert prefill["hull"]["equal"] == {"query_cached": 0}
    eager = law["decode"]["eager"]
    assert eager["fit"]["zero"] == ["graph_padding"]
    for vector in [prefill, *law["decode"].values()]:
        assert set(vector["fit"]) == {
            "samples",
            "warm",
            "dropped",
            "condition",
            "centre",
            "mad",
            "zero",
        }
        assert vector["fit"]["samples"] > 0 and vector["fit"]["condition"] >= 1.0
    cost = StepLinearBackend(StepLinearLaw(law)).price(_view([(100, 100, False)]))
    assert cost.terms[-1].seconds == 0.0


def test_warm_steps_are_not_fitted(tmp_path):
    steps = _steps(2)
    # Two leading compiles; a later compile (step 90) is warm too; a later step
    # at four times its price (step 120) is dropped, not warm.
    stretch = {0: 40.0, 1: 30.0, 90: 20.0, 120: 4.0}
    record = _record(tmp_path / "dp0.jsonl", steps, stretch)
    law = fit([record])
    assert law["warm_steps"] == [[0, 90, 1]]
    assert law["cut"]["warm"] == 3 and law["cut"]["warm_s"] == pytest.approx(
        40 * steps[0][2] + 30 * steps[1][2] + 20 * steps[90][2], rel=1e-6
    )
    assert (law["prefill"]["fit"]["warm"], law["prefill"]["fit"]["dropped"]) == (2, 1)
    assert law["decode"]["2"]["fit"]["warm"] == 1
    assert law["prefill"]["coefficients"] == pytest.approx(PREFILL, rel=1e-6)


def test_three_forms_are_scored_on_a_record_held_out_of_their_fit(tmp_path):
    # Prefill pays a history term the base form cannot see.
    records = [
        _record(tmp_path / f"dp0-{s}.jsonl", _steps(s, history=2e-7)) for s in (3, 4, 5)
    ]
    scores = {form: held_out(records, form) for form in ("base", "history", "shared")}

    history = np.abs(
        np.concatenate([e for b, e in scores["history"].items() if "prefill" in b])
    )
    plain = np.abs(
        np.concatenate([e for b, e in scores["base"].items() if "prefill" in b])
    )
    assert history.max() < 1e-6 < np.median(plain)
    assert {b for b in scores["base"]} == {b for b in scores["shared"]}
    assert (
        "decode rung 2" in scores["base"]
        and "prefill tokens 2048-8192" in scores["base"]
    )

    shared = fit(records, "shared")
    slopes = {v["coefficients"]["context"] for v in shared["decode"].values()}
    assert len(slopes) == 1
    StepLinearBackend(StepLinearLaw(shared)).price(_view([(1, 900, True)], 2))
    rows, _, seconds = _steps(3, history=2e-7)[4]
    cost = StepLinearBackend(StepLinearLaw(fit(records, "history"))).price(_view(rows))
    assert cost.seconds == pytest.approx(seconds, rel=1e-6)
    assert cost.terms[-1].name == "prefill.history"


def test_a_rung_a_mad_pass_drops_entirely_gets_coefficient_zero():
    # `shared` gives each rung its own intercept column (named `step@...`).
    # Rung B's two rows share a context feature but carry labels no single
    # non-negative intercept reconciles with rung A's tight fit, so a MAD
    # pass drops both: rung B's column is then all zero among kept rows.
    names = ["context", "step@A", "step@B"]
    rng = np.random.default_rng(1)
    context_a = rng.uniform(1, 2, 60)
    y_a = 0.02 * context_a + 0.03 + rng.normal(0, 1e-5, 60)
    x_a = np.column_stack([context_a, np.ones(60), np.zeros(60)])
    x_b = np.column_stack([[1.5, 1.5], [0, 0], [1, 1]])
    y_b = np.array([3.0, 30.0])

    solved = _solve(np.vstack([x_a, x_b]), np.concatenate([y_a, y_b]), names)

    assert not solved["keep"][-2:].any()
    assert solved["coef"][2] == 0.0
    assert np.isfinite(solved["coef"]).all()


def test_the_command_writes_a_law_a_run_file_can_carry(tmp_path, capsys):
    records = [str(_record(tmp_path / f"dp0-{s}.jsonl", _steps(s))) for s in (6, 7)]
    main([*records, "--out", str(tmp_path / "law.json")])

    law = StepLinearLaw(json.loads((tmp_path / "law.json").read_text()))
    assert "2 real records" in law.fitted_on
    out = capsys.readouterr().out
    for form in ("base", "history", "shared"):
        assert f"held out, form {form}" in out


def test_a_rung_too_thin_to_fit_is_named_and_its_steps_refused(tmp_path):
    steps = _steps(8)
    fours = [i for i, s in enumerate(steps) if s[1] == 4][2:]
    steps = [s for i, s in enumerate(steps) if i not in fours]
    law = fit([_record(tmp_path / "dp0.jsonl", steps)])

    assert law["unfitted"] == {"4": "2 steps for 3 coefficients"}
    assert "4" not in law["decode"]
    with pytest.raises(CostRefused, match="no decode vector for rung 4"):
        StepLinearBackend(StepLinearLaw(law)).price(_view([(1, 900, True)], 4))


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


@pytest.mark.parametrize("kept", [3, 0])
def test_a_law_with_prefill_too_thin_to_fit_loads_and_refuses_prefill(tmp_path, kept):
    steps = _steps(10)
    prefill = [i for i, s in enumerate(steps) if s[1] is None and not s[0][0][2]]
    steps = [s for i, s in enumerate(steps) if i not in prefill[kept:]]
    law = fit([_record(tmp_path / "dp0.jsonl", steps)])

    assert law["unfitted"] == {"prefill": f"{kept} steps for 4 coefficients"}
    backend = StepLinearBackend(StepLinearLaw(law))
    with pytest.raises(CostRefused, match="no prefill vector"):
        backend.price(_view([(100, 100, False)]))
    assert backend.price(_view([(1, 900, True)], 2)).seconds > 0


def test_a_step_mixing_prefill_and_decode_is_cut_not_fitted(tmp_path):
    steps = _steps(11)
    steps[20] = ([(100, 100, False), (1, 900, True)], None, 1.0)
    rows, cut = samples(_record(tmp_path / "dp0.jsonl", steps))

    assert cut == {"exposed": 0, "mixed": 1}
    assert 20 not in [r[3] for r in rows]
    law = fit([tmp_path / "dp0.jsonl"])
    assert law["prefill"]["coefficients"] == pytest.approx(PREFILL, rel=1e-6)


def test_the_cost_report_names_the_leading_warm_steps_the_law_names(tmp_path):
    # One record scatters by 1%, the other by 25% after a first step at 1.5
    # times its price. The pooled scatter is narrow, so the fit names that step
    # warm though the wide record's own scatter would not.
    rng = np.random.default_rng(12)
    narrow = {i: 1 + rng.uniform(-0.01, 0.01) for i in range(400)}
    wide = {i: 1 + rng.uniform(-0.25, 0.25) for i in range(240)} | {0: 1.5}
    (tmp_path / "wide").mkdir()
    records = [
        _record(tmp_path / "narrow.jsonl", _steps(12, n=400), stretch=narrow),
        _record(tmp_path / "wide" / "dp0.jsonl", _steps(13), stretch=wide),
    ]
    law = fit(records)

    assert 0 in law["warm_steps"][1]
    named = [w["step"] for w in cost_accuracy(tmp_path / "wide", law)["warm"]]
    assert sorted(named) == sorted(law["warm_steps"][1])


# Host laws, ns-exact on (1, tokens, seqs). Decode's seqs equal its tokens, so
# its law is written on tokens alone, as the fit returns it.
HOST = {
    "decode": {
        "E": [2e-4, 3e-6, 0.0],
        "W_pre": [3e-4, 5e-6, 0.0],
        "W_launch": [1e-4, 2e-6, 0.0],
        "W_post": [1.5e-4, 4e-6, 0.0],
    },
    "prefill": {
        "E": [4e-4, 2e-8, 5e-5],
        "W_pre": [6e-4, 1e-7, 8e-5],
        "W_launch": [2e-4, 3e-8, 2e-5],
        "W_post": [3e-4, 1e-9, 6e-5],
    },
    "middle": {
        "E": [5e-4, 3e-8, 4e-5],
        "W_pre": [7e-4, 9e-8, 7e-5],
        "W_launch": [2.5e-4, 2e-8, 3e-5],
    },
}
KEY = {"host_cpu": "test cpu", "atom": "0" * 64, "model": None, "engine": {}}
IDLE_NS = 10**9
#: Device work between forwards, which a decode's label takes and a prefill's does not.
GAP_NS = 40_000


def _timed_record(path, steps, idle_after=(), key=KEY, gap=0):
    """`steps` as a real record with device windows and host stamps.

    Each forward runs the host laws' segments and its price on the device, by
    the recurrence: a device that reaches a forward before its launch ends
    waits for it, which stretches the window. Every fourth step's prefill is a
    middle chunk, and `gap` ns of device work runs between forwards. Returns the
    record and its launch-exposed steps.
    """
    t, device_end, output_end, owner, lines, exposed = 0, None, None, 0, [], []
    for i, (rows, rung, seconds) in enumerate(steps):
        if i - 1 in idle_after:
            owner += 1
            t += IDLE_NS
        ids = [owner] + [10_000 + i * 10 + j for j in range(len(rows) - 1)]
        prefill, output = not rows[0][2], bool(rows[0][2] or i % 4)
        kind = "decode" if not prefill else "prefill" if output else "middle"
        x = (1, sum(r[0] for r in rows), len(rows))
        ns = {k: round(float(np.dot(c, x)) * 1e9) for k, c in HOST[kind].items()}
        enter = t + ns["E"]
        g0 = enter + ns["W_pre"] // 2
        g1 = g0 + 1000
        l0 = g1 + ns["W_pre"] - ns["W_pre"] // 2
        l1 = l0 + ns["W_launch"]
        start = l0 if device_end is None else max(device_end + gap, l0)
        busy = round(seconds * 1e9) + max(0, l1 - start)
        if start < l1:
            exposed.append(i)
        line = {
            "step": i,
            "dp_rank": 0,
            "is_dummy_run": False,
            "produces_output": output,
            "batch": [[r, q, n] for r, (q, n, _) in zip(ids, rows)],
            "named": [],
            "rows": [list(r) for r in rows],
            "rung": rung,
            "t_enter_ns": enter,
            "t_gate_ns": [g0, g1],
            "t_launch_ns": [l0, l1],
            "d_busy_ns": busy,
        }
        if device_end is not None:
            line["d_gap_ns"] = start - device_end
        device_end = start + busy
        t = l1
        if output:
            r0 = l1 + ns["W_post"] // 2
            r1 = r0 if output_end is None else max(r0, output_end)
            line["t_recv_ns"] = [r0, r1]
            t = r1 + ns["W_post"] - ns["W_post"] // 2
            output_end = device_end
        line["t_exit_ns"] = t
        lines.append(line)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    if key is not None:
        path.with_suffix(".key.json").write_text(json.dumps(key))
    return path, exposed


def test_device_busy_time_and_host_laws_are_recovered_from_a_timed_record(tmp_path):
    steps = _steps(14, n=400)
    record, exposed = _timed_record(
        tmp_path / "dp0.jsonl", steps, {60, 201}, gap=GAP_NS
    )

    # The first forward and the two after idle find the device waiting on them:
    # the decode is cut, the prefills keep the wait in their label.
    assert exposed == [0, 61, 202]
    rows, cut = samples(record)
    assert cut == {"exposed": 1, "mixed": 0}
    labelled = {r[3]: r[2] for r in rows}
    assert 61 not in labelled and labelled[202] > steps[202][2]
    assert labelled[62] == pytest.approx(steps[62][2], rel=1e-6)
    assert labelled[63] == pytest.approx(steps[63][2] + GAP_NS * 1e-9, rel=1e-6)

    law = fit([record])
    assert law["prefill"]["coefficients"] == pytest.approx(PREFILL, rel=1e-5)
    decode = dict(DECODE["2"], step=DECODE["2"]["step"] + GAP_NS * 1e-9)
    assert law["decode"]["2"]["coefficients"] == pytest.approx(decode, rel=1e-5)

    assert law["host"]["key"] == KEY
    assert law["host_fit"]["cut"] == {"idle": 2}
    assert law["host_fit"]["unfitted"] == {}
    assert law["host_fit"]["decode.E"]["zero"] == ["seqs"]
    assert set(law["host"]["laws"]["middle"]) == {"E", "W_pre", "W_launch"}
    for kind, segments in HOST.items():
        for segment, c in segments.items():
            got = law["host"]["laws"][kind][segment]
            assert got == pytest.approx(c, rel=1e-4, abs=1e-12), f"{kind}.{segment}"


def test_records_under_two_host_keys_are_refused(tmp_path):
    other = dict(KEY, host_cpu="another cpu")
    a = _timed_record(tmp_path / "a.jsonl", _steps(15))[0]
    b = _timed_record(tmp_path / "b.jsonl", _steps(16), key=other)[0]
    c = _timed_record(tmp_path / "c.jsonl", _steps(17), key=None)[0]

    for pair in ([a, b], [a, c]):
        with pytest.raises(ValueError, match="2 host keys"):
            fit(pair)
    assert "host" not in fit([c])


def test_a_record_without_device_windows_is_refused_by_the_fit_and_the_cost_check(
    tmp_path,
):
    (tmp_path / "steps").mkdir()
    record = _record(tmp_path / "steps" / "dp0.jsonl", _steps(18))
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    record.write_text(
        "".join(
            json.dumps({k: v for k, v in x.items() if k != "d_busy_ns"}) + "\n"
            for x in lines
        )
    )

    with pytest.raises(ValueError, match="has no device windows"):
        fit([record])
    with pytest.raises(ValueError, match="has no device windows"):
        cost_accuracy(tmp_path / "steps", {"provenance": "planted", "decode": {}})


def test_the_command_writes_host_laws_a_run_file_takes_and_scores_them_held_out(
    tmp_path, capsys
):
    records = [
        str(_timed_record(tmp_path / f"dp0-{s}.jsonl", _steps(s))[0]) for s in (19, 20)
    ]
    main([*records, "--out", str(tmp_path / "law.json")])

    law = json.loads((tmp_path / "law.json").read_text())
    StepLinearLaw(law)
    assert set(law["host"]) == {"key", "laws"}
    assert {len(c) for s in law["host"]["laws"].values() for c in s.values()} == {3}
    out = capsys.readouterr().out
    assert "held out, host laws" in out and "host decode W_post | " in out


def test_the_host_key_reads_atom_outside_compass_and_the_real_runner_writes_it(
    tmp_path, monkeypatch
):
    from test_parity_record import ENV, _config, _drive, _Runner

    import atom
    from atom.compass.run import host_key

    tree = tmp_path / "atom"
    (tree / "compass").mkdir(parents=True)
    (tree / "engine.py").write_text("a = 1\n")
    (tree / "compass" / "sim.py").write_text("b = 1\n")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text('{"hidden_size": 8}')
    monkeypatch.setattr(atom, "__file__", str(tree / "__init__.py"))
    config = SimpleNamespace(
        model=str(tmp_path / "model"),
        enforce_eager=True,
        kv_cache_block_size=16,
        kv_cache_dtype="bf16",
        quant_config=SimpleNamespace(compute_hash=lambda: "q"),
        tensor_parallel_size=1,
    )
    key = host_key(config)
    assert key["host_cpu"] and key["model"].startswith("sha256:")
    assert key["engine"]["kv_cache_block_size"] == 16
    (tree / "compass" / "sim.py").write_text("b = 2\n")
    assert host_key(config) == key
    (tree / "engine.py").write_text("a = 2\n")
    assert host_key(config)["atom"] != key["atom"]

    batches = _drive(tmp_path / "source", monkeypatch, {0: [0]})[0][0]
    monkeypatch.setenv(ENV, str(tmp_path / "out"))
    runner = _Runner(_config())
    runner.host_key = key
    runner.forward(batches[0])
    assert json.loads((tmp_path / "out" / "dp0.key.json").read_text()) == key


def test_the_host_key_leaves_out_what_only_a_simulated_run_reaches(
    tmp_path, monkeypatch
):
    import shutil
    from pathlib import Path

    import atom
    from atom.compass.run import host_key

    src, tree = Path(atom.__file__).parent, tmp_path / "atom"
    (tree / "utils").mkdir(parents=True)
    (tree / "model_engine").mkdir()
    for name in ("clock", "compass_loop", "zmq_shim"):
        shutil.copy(src / f"utils/{name}.py", tree / f"utils/{name}.py")
    shutil.copy(src / "model_engine/scheduler.py", tree / "model_engine/scheduler.py")
    monkeypatch.setattr(atom, "__file__", str(tree / "__init__.py"))
    config = SimpleNamespace(
        model=str(tmp_path / "model"),
        enforce_eager=True,
        kv_cache_block_size=16,
        kv_cache_dtype="bf16",
        quant_config=SimpleNamespace(compute_hash=lambda: "q"),
        tensor_parallel_size=1,
    )
    key = host_key(config)["atom"]

    def edited(name, old, new):
        text = (src / name).read_text()
        assert old in text
        (tree / name).write_text(text.replace(old, new, 1))
        try:
            return host_key(config)["atom"]
        finally:
            shutil.copy(src / name, tree / name)

    body = "\n    edited = 1\n"
    assert (
        edited("utils/clock.py", "class LPRuntime:\n", "class LPRuntime:" + body) == key
    )
    assert (
        edited("utils/zmq_shim.py", "class _Socket:\n", "class _Socket:" + body) == key
    )
    assert edited("utils/compass_loop.py", "\n", "\nedited = 1\n") == key
    assert edited("utils/clock.py", "\ndef now(", "\nedited = 1\ndef now(") != key
    assert edited("utils/zmq_shim.py", "def __getattr__", "def _getattr") != key
    assert edited("model_engine/scheduler.py", "\n", "\nedited = 1\n") != key


def test_a_run_refuses_host_laws_fitted_under_another_key(tmp_path, monkeypatch):
    from test_memory_readings import META_8B
    from test_run_step_grid import deployment, publish, run_file

    from atom.compass import run as compass_run

    laws = {"decode": {"E": [1e-4, 0.0, 0.0]}}
    cases = (("same", {}), ("other", {"host_cpu": "another cpu"}), ("none", None))
    for name, other in cases:
        (tmp_path / name).mkdir()
        r = deployment(tmp_path / name)
        publish(tmp_path / name, r)
        key = compass_run.host_key(r.config)
        host = {"laws": laws}
        if other is not None:
            host["key"] = dict(key, **other)
        run = run_file(tmp_path / name, monkeypatch, host=host)
        monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / name / "now.bin"))
        compass_run._PublishedClock(
            compass_run.ENGINE, compass_run.channel_table(run), None
        )
        if other == {}:
            compass_run.runner(r, META_8B)
            assert r.compass_host.laws == laws
        else:
            differ = list(other or key)
            with pytest.raises(ValueError, match=re.escape(f"another {differ}")):
                compass_run.runner(r, META_8B)


ROWMOD = {"step": 0.02, "context": 1e-8, "graph_padding": 5e-9, "rowmod": 3e-7}


def _rowmod_steps(seed, n, k=4, uniform=False, faint=False):
    """Rung-8 decode steps whose cost reads `rowmod(k)` of their rows in order.

    Every step is wider than 4 rows, so its order moves `rowmod(k)`. With
    `uniform`, every row of a step holds one context and a step is 1 to 8 rows
    wide: no order moves `rowmod(k)`, and the rung's three features cannot
    express it. With `faint`, the `rowmod` term carries 12% of its weight and
    each step's seconds a 3% scatter: the column cuts the MAD by under a tenth.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        width = int(rng.integers(1 if uniform else 5, 9))
        context = rng.integers(1000, 60000, size=width)
        if uniform:
            context[:] = context[0]
        rows = [(1, int(c), True) for c in context]
        view = _view(rows, 8)
        modded = max(int(context[r::k].sum()) for r in range(min(k, width)))
        seconds = (
            ROWMOD["step"]
            + ROWMOD["context"] * int(context.sum())
            + ROWMOD["graph_padding"] * view.graph_padding
            + ROWMOD["rowmod"] * (0.12 if faint else 1.0) * modded
        )
        if faint:
            seconds *= rng.normal(1.0, 0.03)
        out.append((rows, 8, seconds))
    return out


def test_a_rung_whose_cost_follows_row_order_adopts_rowmod_and_prices_the_order(
    tmp_path,
):
    records = [
        _record(tmp_path / f"dp0-{s}.jsonl", _rowmod_steps(s, 150)) for s in (20, 21)
    ]
    law = fit(records)

    vector = law["decode"]["8"]
    assert vector["rowmod"] == 4
    assert vector["coefficients"] == pytest.approx(ROWMOD, rel=1e-4, abs=1e-12)
    evidence = vector["fit"]["rowmod"]
    assert (
        evidence["shuffled_mad"] > 0.9 * evidence["mad_without"] > vector["fit"]["mad"]
    )
    assert set(evidence["held_out_p50"]) == {"0", "1"}

    # One multiset of rows in two orders: rows 0 and 4 share a residue mod 4.
    backend = StepLinearBackend(StepLinearLaw(law))
    rows = [(1, c, True) for c in (50000, 1000, 1000, 1000, 50000, 1000)]
    apart = rows[:3] + rows[4:5] + rows[3:4] + rows[5:]
    together = backend.price(_view(rows, 8))
    assert together.terms[-1].name == "decode.rowmod"
    assert together.seconds - backend.price(_view(apart, 8)).seconds == (
        pytest.approx(ROWMOD["rowmod"] * 49000, rel=1e-3)
    )


@pytest.mark.parametrize(
    "steps, k, uniform, faint",
    [
        ((150, 99), 4, False, False),
        ((150, 150), 4, True, False),
        ((150, 150), 2, False, False),
        ((150, 150), 4, False, True),
    ],
    ids=[
        "a cell under 100 steps",
        "an order shuffling cannot move",
        "a held-out cell priced worse",
        "a column cutting the MAD by under a tenth",
    ],
)
def test_a_rung_keeps_its_law_unless_every_cell_and_the_order_support_rowmod(
    tmp_path, steps, k, uniform, faint
):
    # The second record's cost follows `rowmod(k)`, the first's `rowmod(4)`.
    records = [
        _record(
            tmp_path / "dp0-20.jsonl", _rowmod_steps(20, steps[0], 4, uniform, faint)
        ),
        _record(
            tmp_path / "dp0-21.jsonl", _rowmod_steps(21, steps[1], k, uniform, faint)
        ),
    ]
    law = fit(records)

    assert "rowmod" not in law["decode"]["8"]
    assert set(law["decode"]["8"]["coefficients"]) == {
        "step",
        "context",
        "graph_padding",
    }


def _validated(backend, rows, rung=None):
    """Whether every term of the step is validated; one answer per step."""
    flags = {t.provenance.validated for t in backend.price(_view(rows, rung)).terms}
    assert len(flags) == 1
    return flags.pop()


def test_the_command_writes_held_out_errors_and_validates_only_inside_their_hull(
    tmp_path,
):
    records = [str(_record(tmp_path / f"dp0-{s}.jsonl", _steps(s))) for s in (6, 7)]
    main([*records, "--out", str(tmp_path / "law.json")])
    written = json.loads((tmp_path / "law.json").read_text())

    bands = written["decode"]["2"]["held_out"]["bands"]
    assert set(bands) == {"decode rung 2"}
    assert bands["decode rung 2"]["steps"] == sum(
        rung == 2 for s in (6, 7) for _, rung, _ in _steps(s)
    )
    assert abs(bands["decode rung 2"]["p50"]) < 1e-6
    assert written["prefill"]["held_out"]["bands"]
    backend = StepLinearBackend(StepLinearLaw(written))
    # Every step of both records was priced held out, so every one is inside.
    for s in (6, 7):
        for rows, rung, _ in _steps(s):
            assert _validated(backend, rows, rung)
    assert not _validated(backend, [(1, 10**6, True), (1, 10**6, True)], 2)
    assert not _validated(backend, [(9000, 9000, False)])


def test_a_rung_only_one_record_holds_gets_no_held_out_errors(tmp_path):
    no_fours = [s for s in _steps(7) if s[1] != 4]
    records = [
        str(_record(tmp_path / "dp0-6.jsonl", _steps(6))),
        str(_record(tmp_path / "dp0-7.jsonl", no_fours)),
    ]
    main([*records, "--out", str(tmp_path / "law.json")])
    written = json.loads((tmp_path / "law.json").read_text())

    assert "held_out" in written["decode"]["2"]
    assert "held_out" not in written["decode"]["4"]
    backend = StepLinearBackend(StepLinearLaw(written))
    fours = [rows for rows, rung, _ in _steps(6) if rung == 4]
    assert not any(_validated(backend, rows, 4) for rows in fours)


def test_a_shape_only_a_dropped_outlier_reached_prices_extrapolated_and_validated(
    tmp_path,
):
    # Two rows past every recorded context, three times slower than the law:
    # the fit drops it, the law fitted on the other record prices it.
    far = [(1, 150_000, True), (1, 150_000, True)]
    steps = _steps(6)
    steps.insert(101, (far, 2, _price(far, 2)))
    records = [
        str(_record(tmp_path / "dp0-6.jsonl", steps, stretch={101: 3.0})),
        str(_record(tmp_path / "dp0-7.jsonl", _steps(7))),
    ]
    main([*records, "--out", str(tmp_path / "law.json")])
    written = json.loads((tmp_path / "law.json").read_text())

    assert written["decode"]["2"]["fit"]["dropped"] == 1
    cost = StepLinearBackend(StepLinearLaw(written)).price(_view(far, 2))
    assert {t.provenance.source for t in cost.terms} == {"step-linear-extrapolated"}
    assert {t.provenance.validated for t in cost.terms} == {True}


def test_a_law_without_held_out_errors_prices_every_term_unvalidated(tmp_path):
    record = str(_record(tmp_path / "dp0-6.jsonl", _steps(6)))
    main([record, "--out", str(tmp_path / "law.json")])
    written = json.loads((tmp_path / "law.json").read_text())

    assert "held_out" not in json.dumps(written)
    backend = StepLinearBackend(StepLinearLaw(written))
    assert not any(_validated(backend, rows, rung) for rows, rung, _ in _steps(6))


def test_a_rowmod_rung_is_validated_on_its_rowmod_coordinate_too(tmp_path):
    records = [
        _record(tmp_path / f"dp0-{s}.jsonl", _rowmod_steps(s, 150)) for s in (20, 21)
    ]
    law = fit(records)
    mark_held_out(law, records)
    backend = StepLinearBackend(StepLinearLaw(law))

    assert law["decode"]["8"]["rowmod"] == 4
    assert all(_validated(backend, rows, 8) for rows, _, _ in _rowmod_steps(20, 150))
    # Five rows the records' contexts span, ordered so rowmod(4) holds 98% of
    # their context, which no recorded order came near.
    lopsided = [(1, c, True) for c in (60000, 1000, 1000, 1000, 60000)]
    assert backend.price(_view(lopsided, 8)).terms[-1].name == "decode.rowmod"
    assert not _validated(backend, lopsided, 8)
