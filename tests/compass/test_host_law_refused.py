# SPDX-License-Identifier: MIT
"""A forward whose kind lacks a host law for a segment it runs is refused as a
step no cost source answers is, naming ``<kind>.<segment>``.

Batches are ATOM's own: `test_host_device_recurrence`'s streak, where each
request prefills in three middle chunks and a final chunk, then decodes, and
`test_refused_step_continues`' driven requests, whose decodes no cost source
prices.
"""

import re
from types import SimpleNamespace

import pytest
import torch
from test_dp_step_max import runner
from test_host_device_recurrence import BACKEND, streak  # noqa: F401
from test_refused_step_continues import (
    REFUSED,
    cost_lines,
    drive,
    finish,
    no_eager_rung,
    recording,
    run_file,
)

from atom.compass.backends.resolver import CostRefused
from atom.compass.runner import overrides
from atom.compass.runner.overrides import HOST_SEGMENTS, HostDeviceRecurrence

OUTPUT = {k: dict.fromkeys(HOST_SEGMENTS, (1e-4, 0, 0)) for k in ("decode", "prefill")}
NO_MIDDLE = "host: no law for middle.E, middle.W_pre, middle.W_launch"


def host(r, laws, skip=()):
    clock = [0.0]
    r.compass_host = HostDeviceRecurrence(laws, skip, lambda: clock[0])
    return clock


def test_a_run_without_middle_laws_refuses_each_middle_chunk_by_name(
    streak, tmp_path, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    run = run_file(tmp_path, monkeypatch)
    r = runner(BACKEND)
    clock, refused = host(r, OUTPUT), []
    for b in streak:
        clock[0] += overrides._group_step_seconds(r, b, BACKEND)
        refused.append(len(r.refused_commands()))

    # Each middle chunk is refused, the streak's first forward among them, and
    # no output forward.
    middle = [not b.produces_output() for b in streak]
    assert [b - a for a, b in zip([0, *refused], refused)] == middle
    out = finish(monkeypatch, tmp_path, run, r)["schedule"]["refusals"]
    assert out["reasons"] == [[f"cost:{NO_MIDDLE}", sum(middle)]]
    assert out["fraction_of_steps"] == sum(middle) / len(streak)


def test_under_abort_the_first_middle_chunk_ends_the_run_naming_its_law(
    streak,  # noqa: F811
):
    r = runner(BACKEND)
    clock, r.compass_abort = host(r, OUTPUT), True
    with pytest.raises(CostRefused, match="middle.W_pre") as refusal:
        for b in streak:
            clock[0] += overrides._group_step_seconds(r, b, BACKEND)
    assert b is next(m for m in streak if not m.produces_output())
    assert str(refusal.value.declined[0]) == f"refused({NO_MIDDLE})"


@pytest.mark.parametrize(
    "kind, laws, skip, named",
    [
        ("decode", {"decode": {"E": [1, 0, 0]}}, HOST_SEGMENTS[1:3], "decode.W_post"),
        ("decode", {"decode": {"E": [1, 0, 0]}}, HOST_SEGMENTS[1:], None),
        ("middle", {"middle": dict.fromkeys(HOST_SEGMENTS[:3], (1, 0, 0))}, (), None),
        ("prefill", {"decode": {}}, ("E", "W_pre"), "prefill.W_launch, prefill.W_post"),
        ("prefill", {}, (), None),
    ],
)
def test_a_forward_needs_a_law_for_each_segment_it_runs_and_does_not_skip(
    kind, laws, skip, named
):
    batch = SimpleNamespace(
        total_seqs_num_prefill=kind != "decode",
        total_tokens_num=8,
        total_seqs_num=2,
        produces_output=lambda: kind != "middle",
    )
    recurrence = HostDeviceRecurrence(laws, skip, lambda: 0.0)
    if named is None:
        assert recurrence.charge(batch, 1.0) >= 0
        return
    with pytest.raises(CostRefused, match=re.escape(f"(host: no law for {named})")):
        recurrence.charge(batch, 1.0)
    assert (list(recurrence.ends), recurrence.segments) == ([-float("inf")] * 2, {})


def test_a_step_both_refuse_hands_in_one_reason_and_charges_nothing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    r = recording(runner(no_eager_rung()), tmp_path, monkeypatch)
    host(r, {"prefill": OUTPUT["prefill"]})
    drive(r)

    lines = [line for line in cost_lines(tmp_path) if line["refused"]]
    decode = ", ".join(f"decode.{s}" for s in HOST_SEGMENTS)
    both = f"{REFUSED}; host: no law for {decode}"
    assert lines and [line["refused"] for line in lines] == list(r.refused_commands())
    for line in lines:
        assert (line["refused"], line["charge_s"], line["skipped"]) == (both, 0.0, [])
        assert not set(HOST_SEGMENTS) & line.keys()
