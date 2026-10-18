# SPDX-License-Identifier: MIT
"""A forward's charge: the backend's price composed with host time per runner.

Batches are ATOM's own, from `test_runner_step_semantics`' driven streak: each
request prefills in three middle chunks and a final chunk, then decodes.
Each is priced by `StepStubBackend` through the runner's `_group_step_seconds`;
`clock` stands in for the engine LP, advanced by each charge as
`atom.utils.clock.charge` advances it.
"""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
from test_dp_step_max import runner
from test_memory_readings import META_8B
from test_parity_record import _config, _Runner
from test_run_step_grid import B37, decode, deployment, publish, run_file
from test_runner_step_semantics import _drive
from test_validate_runs import _cell, _schedule

from atom.compass import parity
from atom.compass import run as compass_run
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.step_stub import SOURCE, StepStubBackend
from atom.compass.runner import overrides
from atom.compass.runner.overrides import (
    FORWARD_KINDS,
    HOST_SEGMENTS,
    MTP_SEGMENT,
    HostDeviceRecurrence,
)
from atom.compass.validate import runs

BACKEND = StepStubBackend()


@pytest.fixture(scope="module")
def streak():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
        return _drive().batches


def prices(batches):
    """Today's charge of each forward: its price alone."""
    r = runner(BACKEND)
    return [overrides._group_step_seconds(r, b, BACKEND) for b in batches]


def charges(batches, laws=None, skip=()):
    """Each forward's charge, called at the clock the charges before it moved."""
    r, clock = runner(BACKEND), [0.0]
    r.compass_host = HostDeviceRecurrence(laws or {}, skip, lambda: clock[0])
    out = []
    for b in batches:
        out.append(overrides._group_step_seconds(r, b, BACKEND))
        clock[0] += out[-1]
    return out


def cost_lines(batches, directory, monkeypatch, laws=None, skip=(), rank=0):
    """A runner recording to `directory`: its charge of each batch, under
    `laws` when given, and the cost lines it wrote there, below the version
    line."""
    monkeypatch.setenv(parity.ENV, str(directory))
    r, clock = _Runner(_config(), rank), [0.0]
    if laws is not None:
        r.compass_host = HostDeviceRecurrence(laws, skip, lambda: clock[0])
    charged = []
    for b in batches:
        charged.append(r.forward(b).predicted_s)
        clock[0] += charged[-1]
    path = directory / "dp0.cost.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    return [json.loads(line) for line in lines[1:]], charged


def law(kind, seconds):
    """Each host segment of `kind` at a quarter of `seconds`, per forward."""
    return {kind: {s: [seconds / 4, 0, 0] for s in HOST_SEGMENTS}}


def test_with_no_host_time_a_middle_chunk_breaks_the_one_forward_shift(streak):
    final = next(b for b in streak if b.produces_output() and b.total_seqs_num_prefill)
    middle = next(b for b in streak if not b.produces_output())
    decodes = [b for b in streak if not b.total_seqs_num_prefill][:2]
    batches = [final, middle, *decodes]
    a, b, c, _ = prices(batches)
    assert charges(batches) == pytest.approx([0, 0, a, b + c], abs=1e-12)


def test_with_no_host_time_output_forwards_charge_the_one_before(streak):
    decodes = [b for b in streak if not b.total_seqs_num_prefill]
    assert len(decodes) > 3
    assert charges(decodes) == pytest.approx([0, *prices(decodes)[:-1]])


def test_with_no_host_time_middle_chunks_charge_the_one_two_before(streak):
    middles = [b for b in streak if not b.produces_output()][:4]
    assert charges(middles) == pytest.approx([0, 0, *prices(middles)[:2]])


def test_a_host_bound_forward_charges_host_time(streak):
    decodes = [b for b in streak if not b.total_seqs_num_prefill]
    host = 10 * max(prices(decodes))
    assert charges(decodes, law("decode", host)) == pytest.approx([host] * len(decodes))
    assert charges(decodes, law("decode", host), HOST_SEGMENTS) == charges(decodes)


def test_a_device_bound_forward_charges_device_time(streak):
    decodes = [b for b in streak if not b.total_seqs_num_prefill]
    device = prices(decodes)
    host = min(device) / 10
    assert charges(decodes, law("decode", host))[2:] == pytest.approx(device[1:-1])


def test_skipping_a_segment_never_delays_a_forward(streak):
    rng = random.Random(825)
    kinds = ("decode", "prefill", "middle")
    for _ in range(20):
        laws = {
            k: {s: [rng.uniform(0, 1e-2), 0, 0] for s in HOST_SEGMENTS} for k in kinds
        }
        skip = rng.sample(HOST_SEGMENTS, rng.randint(1, 4))
        full = np.cumsum(charges(streak, laws))
        skipped = np.cumsum(charges(streak, laws, skip))
        assert (skipped <= full + 1e-12).all(), skip


def test_a_runs_runner_charges_at_the_clock_its_engine_publishes(tmp_path, monkeypatch):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    run = run_file(tmp_path, monkeypatch)
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    engine = compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    price = 8897.5e-6  # `test_run_step_grid`'s price of this batch
    # The first output forward waits for nothing; the next, called at the same
    # time, for the first one's device work.
    assert [r.forward(decode(B37)).predicted_s for _ in "ab"] == [0.0, price]
    engine.now = 1.0  # an idle wait granted after the device work ended
    assert r.forward(decode(B37)).predicted_s == 0.0


@pytest.mark.parametrize(
    "host",
    [
        {"skip": ["W_lunch"]},
        {"laws": {"decod": {}}},
        {"laws": {"decode": {"E": [0, 0]}}},
        {"law": {}},
        {"decode": {"E": [1, 0, 0]}},
        {"laws": {"E": {"decode": [1, 0, 0]}}},
        {"laws": {"decode": {"decode": [1, 0, 0]}}},
        {"skip": ["decode"]},
    ],
)
def test_a_host_entry_the_runner_would_not_read_is_refused(tmp_path, monkeypatch, host):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"bound_s": 1.0, "host": host}))
    monkeypatch.setenv(compass_run.ENV, str(path))
    with pytest.raises(ValueError, match="host is"):
        compass_run.spec()


def test_each_cost_line_reproduces_the_charge_of_its_step(
    streak, tmp_path, monkeypatch
):
    rng = random.Random(904)
    laws = {
        k: {s: [rng.uniform(0, 1e-2), rng.uniform(0, 1e-5), 0] for s in HOST_SEGMENTS}
        for k in FORWARD_KINDS
    }
    lines, charged = cost_lines(streak, tmp_path, monkeypatch, laws, ["W_launch"])

    steps = [s["step"] for s in parity.read(tmp_path)[0]]
    assert [line["step"] for line in lines] == steps == list(range(len(streak)))
    # Charged again from the lines alone: each segment a constant, and the
    # device time, each called at the clock the lines before it moved.
    now = 0.0
    host = HostDeviceRecurrence({}, (), lambda: now)
    for batch, line, charge in zip(streak, lines, charged):
        segments = {s: (line[s], 0, 0) for s in HOST_SEGMENTS}
        host.laws = dict.fromkeys(FORWARD_KINDS, segments)
        assert (
            host.charge(batch, line["device_s"], line["draft_s"])
            == line["charge_s"]
            == charge
        )
        now += charge
        assert (line["W_launch"], line["skipped"]) == (0.0, ["W_launch"])
        assert line["group_s"] == charge
        assert line["device_s"] == fold_seconds(t["seconds"] for t in line["terms"])
        provenance = {(t["method"], t["validated"], t["source"]) for t in line["terms"]}
        assert (provenance, line["refused"]) == ({("declared", False, SOURCE)}, None)
    assert all(line["E"] > 0 for line in lines)


def test_with_no_host_a_cost_line_is_the_price_and_a_rerun_replaces_the_file(
    streak, tmp_path, monkeypatch
):
    cost_lines(streak, tmp_path, monkeypatch)
    lines, charged = cost_lines(streak[:5], tmp_path, monkeypatch)
    assert [line["step"] for line in lines] == list(range(5))
    for line, charge in zip(lines, charged):
        assert line["device_s"] == line["charge_s"] == line["group_s"] == charge
        assert not {*HOST_SEGMENTS, "skipped"} & line.keys()


def test_no_cost_line_is_written_with_the_record_off_or_off_tp_rank_0(
    streak, tmp_path, monkeypatch
):
    assert cost_lines(streak[:3], tmp_path / "tp1", monkeypatch, rank=1)[0] == []
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(parity.ENV)
    r = _Runner(_config())
    for b in streak[:3]:
        r.forward(b)
    assert list(tmp_path.iterdir()) == []


def test_grading_reads_a_simulated_cells_record_beside_its_cost_lines(tmp_path):
    cells = [_cell(tmp_path / f"c{i}", _schedule(), {}) for i in range(4)]
    graded = runs.grade(cells[1:], cells[0])
    (cells[0] / "steps" / "dp0.cost.jsonl").write_text(json.dumps({"step": 0}) + "\n")
    assert runs.grade(cells[1:], cells[0]) == graded


def test_grading_refuses_a_what_if_run(tmp_path):
    cells = [str(_cell(tmp_path / f"c{i}", _schedule(), {})) for i in range(4)]
    (tmp_path / "c0" / "run.json").write_text(json.dumps({"host": {"skip": ["E"]}}))
    with pytest.raises(ValueError, match="c0 skips host segments"):
        runs.main(cells)


def test_each_kind_takes_its_own_law_on_tokens_and_seqs():
    """A final chunk, a middle chunk and two decodes, exits worked by hand."""
    laws = {
        "prefill": {
            "E": [1, 0, 0],
            "W_pre": [0, 0.01, 0],
            "W_launch": [0, 0, 1],
            "W_post": [1, 0, 0],
        },
        "middle": {"E": [2, 0, 0], "W_pre": [0, 0.005, 0], "W_launch": [1, 0, 0]},
        "decode": {
            "E": [0.5, 0, 0],
            "W_pre": [0, 0, 0.25],
            "W_launch": [1, 0, 0],
            "W_post": [0, 0, 0.5],
        },
    }

    def batch(prefill_seqs, tokens, seqs, output):
        return SimpleNamespace(
            total_seqs_num_prefill=prefill_seqs,
            total_tokens_num=tokens,
            total_seqs_num=seqs,
            produces_output=lambda: output,
        )

    clock = [0.0]
    host = HostDeviceRecurrence(laws, (), lambda: clock[0])
    exits = []
    for b, device in [
        (batch(1, 100, 2, True), 10),  # launch 0+1+1 to 4, device to 12; exit 4+1
        (batch(1, 200, 1, False), 10),  # launch 5+2+1 to 9, device 12 to 22
        # entry waits for the first forward's device end (12)
        (batch(0, 4, 4, True), 3),  # launch 13 to 14, device to 25; exit 14+2
        # entry waits for the second's (22), the output for the third's (25)
        (batch(0, 4, 4, True), 3),  # launch 23 to 24; exit 25+2
    ]:
        clock[0] += host.charge(b, device, 0.0)
        exits.append(clock[0])
    assert exits == pytest.approx([5, 9, 16, 27])


@pytest.mark.parametrize("mtp, exits", [(True, [4, 14, 28]), (False, [4, 13, 27])])
def test_with_mtp_an_output_forward_waits_for_the_target_before_its_launch(mtp, exits):
    """Three decodes, 1 s per host segment, 10 s of target and 4 of draft.

    With MTP the second forward's launch waits for the first one's target end
    (12), and its target for the first one's draft (16); its exit is 12 + 1 + 1.
    """
    seg = {**{s: [1, 0, 0] for s in HOST_SEGMENTS}, MTP_SEGMENT: [0, 0, 0]}
    clock = [0.0]
    host = HostDeviceRecurrence({"decode": seg}, (), lambda: clock[0], mtp=mtp)
    b = SimpleNamespace(
        total_seqs_num_prefill=0,
        total_tokens_num=2,
        total_seqs_num=1,
        produces_output=lambda: True,
    )
    out = []
    for _ in exits:
        clock[0] += host.charge(b, 10, 4)
        out.append(clock[0])
    assert out == pytest.approx(exits)
    assert list(host.ends) == [30, 44]
