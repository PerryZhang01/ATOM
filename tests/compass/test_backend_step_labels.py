# SPDX-License-Identifier: MIT
"""One labelling of a real run's recorded steps, read by the fit, the cost check
and the per-step table alike.

Records are written two ways: by `StepRecord` on `test_parity_device_window.py`'s
planted device, through a real runner's stamps, and as
`test_backend_step_linear_fit.py` plants them, from a known law.
"""

import json

import pytest
from test_backend_step_linear_fit import (
    GAP_NS,
    SPEC,
    _record,
    _rowmod_steps,
    _steps,
    _timed_record,
    _view,
)
from test_memory_readings import DOCUMENT, META_8B
from test_parity_device_window import BUSY, GAP, _recorded
from test_parity_record import _config, _drive

from atom.compass.artifacts import machine_spec_digest
from atom.compass.backends import StepLinearBackend, StepLinearLaw
from atom.compass.backends.step_labels import (
    TABLE_VERSION,
    digest,
    label,
    write_table,
)
from atom.compass.backends.step_linear_fit import (
    fit,
    label_records,
    main,
    mark_held_out,
)
from atom.compass.parity import ENV
from atom.compass.spec import MachineSpec

KEY = {"host_cpu": "planted cpu", "atom": "0" * 64, "model": None, "engine": {}}


def _stamped(tmp_path, monkeypatch):
    """A record `StepRecord` wrote on the planted device, under a host key."""
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    recorded, device = _recorded(batches[-1])
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = recorded(0, _config())
    runner.host_key = KEY
    for batch in batches:
        runner.forward(batch)
    device.tick = BUSY // 3
    runner.exit()
    return tmp_path / "real" / "dp0.jsonl"


def test_a_stamped_record_is_labelled_by_step_kind(tmp_path, monkeypatch):
    record = _stamped(tmp_path, monkeypatch)
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    samples, table = label(record)

    assert [s.step for s in samples] == [x["step"] for x in lines]
    kinds = {}
    for n, (s, line, row) in enumerate(zip(samples, lines, table)):
        rows = [[r.query_tokens, r.context_tokens, r.decode] for r in s.view.requests]
        assert rows == line["rows"]
        assert s.view.capture_rung == line["rung"]
        assert s.view.produces_output == line["produces_output"]
        assert s.view.moe_rows is None and s.cut is None
        busy = BUSY + 2 * n + 1
        decode = bool(s.view.decode)
        kinds.setdefault((decode, line["produces_output"]), []).append(n)
        # A decode takes the device gap before it; no prefill chunk is ever cut.
        assert s.device_s == pytest.approx((busy + GAP * decode) * 1e-9, rel=1e-12)
        assert ("W_post" in s.host) == line["produces_output"]
        assert s.provenance == {
            "source": "record",
            "run": digest(record),
            "requests": [r[0] for r in line["batch"]],
            "config": KEY,
        }
        assert row["label_s"] == s.device_s and row["idle"] == s.idle
        assert {k: row[k] for k in s.host} == s.host
    # Middle chunks, output prefills and decodes all ran; the planted device
    # starts every prefill before its host launch ends, and none is cut.
    assert set(kinds) == {(False, False), (False, True), (True, True)}
    exposed = [row["step"] for row in table if row["launch_exposed"]]
    assert exposed and all(not samples[n].view.decode for n in exposed)
    # The second request's first chunk follows the first's last: no request in
    # common, so its `E` is cut as idle, as the record's first forward has none.
    assert [s.step for s in samples if s.idle] == [4]
    assert [s.step for s in samples if "E" not in s.host] == [0, 4]


def test_a_rank_s_moe_rows_ride_in_each_sample_s_view(tmp_path):
    from test_step_linear_dp import _groups, _records

    groups = _groups(1, n=24)
    for rank, record in enumerate(_records(tmp_path, groups)):
        samples = label(record)[0]
        assert [s.view.moe_rows for s in samples] == [
            views[rank].moe_rows for views, _ in groups
        ]
        assert {s.rank for s in samples} == {rank}


def test_the_command_writes_each_record_s_table_with_the_labels_it_fitted(tmp_path):
    # A decode after idle finds the device waiting on its launch: it is cut.
    records = [
        _timed_record(tmp_path / f"dp0-{s}.jsonl", _steps(s), {60}, gap=GAP_NS)[0]
        for s in (19, 20)
    ]
    main([*map(str, records), "--out", str(tmp_path / "law.json"), *SPEC])
    law = StepLinearBackend(
        StepLinearLaw(json.loads((tmp_path / "law.json").read_text()))
    )

    cuts = set()
    for record in records:
        header, *table = map(
            json.loads, (tmp_path / f"{record.stem}.cost.jsonl").open()
        )
        assert header == {"version": TABLE_VERSION}
        fitted = {s.step: s for s in label(record)[0]}
        assert [row["step"] for row in table] == list(fitted)
        for row in table:
            s = fitted[row["step"]]
            assert row["label_s"] == s.device_s and row["cut"] == s.cut
            if row["cut"] is None and s.view.decode:
                gap = pytest.approx(row["d_busy_s"] + row["d_gap_s"], rel=1e-12)
                assert row["label_s"] == gap
                # The law was fitted on these labels: it prices each one back.
                price = law.price(s.view).seconds
                assert price == pytest.approx(row["label_s"], rel=1e-5)
            elif row["cut"] is None:
                assert row["label_s"] == row["d_busy_s"]
        cuts |= {row["cut"] for row in table}
    assert cuts == {None, "exposed"}


def test_the_report_states_inputs_vectors_and_held_out_errors_per_run(tmp_path):
    records = [_record(tmp_path / f"dp0-{s}.jsonl", _steps(s)) for s in (6, 7)]
    samples = label_records(records)[0]
    law, report = fit(samples)

    # Labelling writes nothing; the command writes the tables.
    assert not list(tmp_path.glob("*.cost.jsonl"))
    runs = [digest(r) for r in records]
    assert report["provenance"] == [
        {"source": "record", "run": run, "config": None} for run in runs
    ]
    vectors = {"prefill": law["prefill"], **law["decode"]}
    assert report["vectors"] == {
        k: {**v["fit"], "hull": v["hull"]} for k, v in vectors.items()
    }
    assert report["held_out"] == {}

    mark_held_out(law, report, samples)
    assert list(report["held_out"]) == runs
    for key, vector in vectors.items():
        for band, b in vector["held_out"]["bands"].items():
            per_run = [report["held_out"][run].get(band, {}) for run in runs]
            assert sum(r.get("steps", 0) for r in per_run) == b["steps"], key


def test_a_seed_fixes_the_shuffles_and_the_same_inputs_give_the_same_bytes(tmp_path):
    records = [
        _record(tmp_path / f"dp0-{s}.jsonl", _rowmod_steps(s, 150)) for s in (20, 21)
    ]
    samples = label_records(records)[0]
    once, again = fit(samples), fit(label_records(records)[0])

    assert json.dumps(once) == json.dumps(again)
    reseeded = fit(samples, seed=1)[0]["decode"]["8"]["fit"]["rowmod"]
    evidence = once[0]["decode"]["8"]["fit"]["rowmod"]
    assert reseeded["shuffled_mad"] != evidence["shuffled_mad"]
    assert reseeded["mad_without"] == evidence["mad_without"]


def test_record_readers_take_rank_records_and_leave_their_tables(tmp_path):
    from atom.compass.parity import read
    from atom.compass.validate.runs import _steps as graded_steps

    (tmp_path / "steps").mkdir()
    record = _record(tmp_path / "steps" / "dp0.jsonl", _steps(6, n=40))
    write_table(record, label(record)[1])

    assert (tmp_path / "steps" / "dp0.cost.jsonl").exists()
    assert list(read(tmp_path / "steps")) == [0]
    assert len(graded_steps(tmp_path)) == 40


def test_the_command_s_laws_load_in_a_run_file(tmp_path, monkeypatch):
    from test_run_step_grid import deployment, run_file

    from atom.compass import run as compass_run

    r = deployment(tmp_path, enable_dp_attention=False, enable_expert_parallel=False)
    key = compass_run.host_key(r.config)
    records = [
        str(_timed_record(tmp_path / f"dp0-{s}.jsonl", _steps(s), key=key)[0])
        for s in (19, 20)
    ]
    main(
        [
            *records,
            "--out",
            str(tmp_path / "law.json"),
            "--machine-spec",
            machine_spec_digest(MachineSpec.from_mapping(DOCUMENT)),
        ]
    )
    law = json.loads((tmp_path / "law.json").read_text())

    run = run_file(tmp_path, monkeypatch, step_linear=law, host=law["host"])
    del run["step_grid"]
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    assert isinstance(r.compass_backend, StepLinearBackend)
    assert r.compass_host.laws == law["host"]["laws"]
    rows = [(1, 900, True), (1, 400, True)]
    assert r.compass_backend.price(_view(rows, 2)).seconds > 0


def test_host_laws_fitted_under_a_stated_engine_load_at_that_engine(
    tmp_path, monkeypatch
):
    from test_run_step_grid import deployment, run_file

    from atom.compass import run as compass_run

    r = deployment(tmp_path, enable_dp_attention=False, enable_expert_parallel=False)
    key = compass_run.host_key(r.config)
    later = ("data_parallel_size", "enable_dp_attention", "enable_expert_parallel")
    later += ("enable_tbo", "enable_tbo_decode", "env")
    # Records whose host key predates those engine fields, as a capture's do.
    old = key | {"engine": {k: v for k, v in key["engine"].items() if k not in later}}
    records = [
        str(_timed_record(tmp_path / f"dp0-{s}.jsonl", _steps(s), key=old)[0])
        for s in (19, 20)
    ]
    stated = tmp_path / "stated.engine.json"
    stated.write_text(json.dumps(key["engine"]))
    spec = machine_spec_digest(MachineSpec.from_mapping(DOCUMENT))
    out = ["--out", str(tmp_path / "law.json"), "--machine-spec", spec]
    main([*records, *out, "--engine", str(stated)])
    law = json.loads((tmp_path / "law.json").read_text())
    assert law["host"]["key"]["engine"] == law["engine_config"] == key["engine"]

    run = run_file(tmp_path, monkeypatch, step_linear=law, host=law["host"])
    del run["step_grid"]
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    assert r.compass_host.laws == law["host"]["laws"]
