# SPDX-License-Identifier: MIT
"""A speculative step: accepted drafts drawn from the declared acceptance, read by
ATOM's own scheduler, and the target and draft device time charged apart."""

import itertools
import json
import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import MockConfig
from test_dp_step_max import dp2, prefill, runner  # noqa: F401
from test_parity_recurrence import MS, line

from atom.compass.backends import step_labels
from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.step_stub import Coefficients, StepStubBackend
from atom.compass.run import _graph_pool
from atom.compass.runner.overrides import (
    HOST_SEGMENTS,
    MTP_SEGMENT,
    HostDeviceRecurrence,
    NonAllocatingRunner,
    RunnerRefusal,
    drafts_on_meta,
    schedulable_ladder,
)
from atom.compass.runner.step_output import DeferredTokenStream
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

REQUESTS, MAX_TOKENS = 16, 200


def _serve(rates, seed=0):
    """`REQUESTS` requests served to their end by ATOM's scheduler, each step's
    reply the stream's; the scheduler's speculative stats and the steps taken."""
    k = len(rates)
    spec = SimpleNamespace(num_speculative_tokens=k, use_dspark=lambda: False)
    scheduler = Scheduler(
        MockConfig(
            speculative_config=spec,
            max_num_seqs=REQUESTS,
            num_kvcache_blocks=4096,
            kv_cache_block_size=16,
            max_model_len=1024,
            max_num_batched_tokens=512,
        )
    )
    stream = DeferredTokenStream([7], rates=rates, seed=seed)
    Sequence.counter = itertools.count()
    params = SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True)
    seqs = [
        Sequence(list(range(5, 69)), 16, sampling_params=params)
        for _ in range(REQUESTS)
    ]
    for s in seqs:
        scheduler.add(s)
    out, steps = queue.Queue(), 0
    while scheduler.has_unfinished_requests():
        steps += 1
        assert steps < 10 * MAX_TOKENS, "the served requests do not finish"
        scheduled = scheduler.schedule()
        if scheduled is None or scheduled[0] is None:
            continue
        batch, rows = scheduled
        reply = stream.step(batch)
        assert reply["draft_token_ids"].shape == (len(reply["req_ids"]), k)
        scheduler.postprocess(
            list(rows.values()),
            ScheduledBatchOutput(**reply),
            stream_output_queue=out,
            batch=batch,
        )
    assert [s.num_completion_tokens for s in seqs] == [MAX_TOKENS] * REQUESTS
    return scheduler.engine_stats.distribution, steps


@pytest.mark.parametrize("rates", [[0.6], [0.9, 0.7, 0.4]])
def test_the_scheduler_counts_the_declared_acceptance(rates):
    """ATOM's own accepted-count histogram, P(a >= i + 1), within 4 standard
    errors of each declared rate."""
    distribution, _ = _serve(rates)
    n = sum(distribution.values())
    assert n > 1000
    for i, rate in enumerate(rates):
        seen = sum(v for a, v in distribution.items() if a > i) / n
        assert abs(seen - rate) < 4 * (rate * (1 - rate) / n) ** 0.5 + 1e-9


def test_full_acceptance_takes_a_third_of_the_decode_steps_of_none():
    """After its first token a request makes three a decode step, or one."""
    full, steps = _serve([1.0, 1.0])
    none, base = _serve([0.0, 0.0])
    assert {a for a, v in full.items() if v} == {2}
    assert {a for a, v in none.items() if v} == {0}
    decodes = MAX_TOKENS - 1
    assert base - steps == decodes - -(-decodes // 3)


def test_a_seed_fixes_the_draws_and_another_moves_them():
    assert _serve([0.5], seed=3) == _serve([0.5], seed=3) != _serve([0.5], seed=4)


class _Drafting(StepStubBackend):
    """The stub's constant step, plus a draft term of `draft` seconds."""

    def __init__(self, coefficients, draft):
        super().__init__(coefficients)
        self.draft = draft

    def price(self, batch_view):
        cost = super().price(batch_view)
        made = cost.terms[0].provenance
        return StepCost([*cost.terms, CostTerm("draft", self.draft, made, draft=True)])


def _exits(draft, tagged, inputs=1.0):
    """Four output prefills through a runner's step: 1 s per host segment,
    `inputs` after the MTP status wait, 10 s of target and `draft` of draft,
    tagged or priced as target."""
    laws = {
        "prefill": {
            **{s: [1, 0, 0] for s in HOST_SEGMENTS},
            MTP_SEGMENT: [inputs, 0, 0],
        }
    }
    if tagged:
        backend = _Drafting(Coefficients.constant(10.0, 1.0), draft)
    else:
        backend = StepStubBackend(Coefficients.constant(10.0 + draft, 1.0))
    r, clock = runner(backend), [0.0]
    r.compass_host = HostDeviceRecurrence(laws, (), lambda: clock[0], mtp=True)
    out = []
    for _ in range(4):
        clock[0] += r.forward(prefill(8)).predicted_s
        out.append(clock[0])
    return out


@pytest.mark.parametrize("draft", [4.0, 0.5])
def test_a_tagged_draft_overlaps_the_next_prepare_and_a_target_does_not(draft):
    """Device-bound, a forward waits for the last target's end, then prepares
    for 1 s while the draft runs: the period is 10 s plus the longer of the
    two. Priced as target, the draft and the prepare add up."""
    split, merged = _exits(draft, True), _exits(draft, False)
    assert np.diff(split)[1:] == pytest.approx([10 + max(draft, 1.0)] * 2)
    assert np.diff(merged)[1:] == pytest.approx([10 + draft + 1.0] * 2)


def test_under_dp_the_mtp_wait_ends_at_the_groups_latest_target(dp2):  # noqa: F811
    """dp0's target ends at 10 s and drafts 4 more; dp1's ends at 5 and drafts
    1. dp1's status wait for its next output forward releases at dp0's target
    end, not at the group's device end less its own draft."""
    ranks = [
        runner(_Drafting(Coefficients.constant(t, 1.0), d), dp_size=2, dp_rank=r)
        for r, t, d in ((0, 10.0, 4.0), (1, 5.0, 1.0))
    ]
    for r in ranks:
        r.compass_host = HostDeviceRecurrence({}, (), lambda: 0.0, mtp=True)
    dp2(*(lambda r=r: r.forward(prefill(8)) for r in ranks))
    hosts = [r.compass_host for r in ranks]
    assert [h.target_end for h in hosts] == [10.0, 5.0]
    assert [h.ends[-1] for h in hosts] == [14.0, 14.0]
    assert [h.output_end for h in hosts] == [10.0, 10.0]
    assert hosts[1].prepare_end(0.0, "decode", (1, 1, 1))[0] == 10.0


def test_an_mtp_record_labels_the_target_alone_and_the_draft_apart(tmp_path):
    """Four MTP decodes: 10 ms of target and 0.5 of draft, 0.5 ms of device gap,
    and a 0.25 ms draft copy wait inside each 1 ms post."""
    lines = []
    for n, (enter, mtp_end) in enumerate([(0, 1), (5, 12), (16, 23), (27, 34)]):
        launch = (mtp_end + 1, mtp_end + 2)
        s = line(n, [[2, 150 + n, True]], True, enter, (enter,) * 2, launch, None, 0)
        s["t_mtp_ns"] = [(enter + 1) * MS, mtp_end * MS]
        s["t_recv_ns"] = [launch[1] * MS] * 2
        s["t_draft_recv_ns"] = [launch[1] * MS, launch[1] * MS + MS // 4]
        s["t_exit_ns"] = (launch[1] + 1) * MS
        s["d_busy_ns"], s["d_target_ns"] = 10.5 * MS, 10 * MS
        s["rung"], s["d_gap_ns"] = None, MS // 2
        lines.append(s)
    record = tmp_path / "dp0.jsonl"
    record.write_text("".join(json.dumps(s) + "\n" for s in lines))
    samples, table = step_labels.label(record)
    assert [s.device_s for s in samples] == pytest.approx([0.010] * 4)
    assert [row["draft_s"] for row in table] == pytest.approx([0.0005] * 4)
    assert [s.host["W_post"] for s in samples] == pytest.approx([0.00075] * 4)


def _speculative(r, rates=(0.5,)):
    r.config.speculative_config = SimpleNamespace(synthetic_acceptance_rates=rates)
    return r


def test_a_speculative_step_priced_with_no_draft_term_is_refused_by_name():
    """The drafter runs after every target forward, so a price with no draft
    term is refused, not charged 0 s of draft; a tagged one is charged."""
    plain = _speculative(runner(StepStubBackend(Coefficients.constant(10.0, 1.0))))
    assert plain.forward(prefill(8)).predicted_s == 0.0
    assert plain.refused_commands() == ("cost:StepStubBackend: prices no draft",)
    tagged = _speculative(runner(_Drafting(Coefficients.constant(10.0, 1.0), 4.0)))
    assert tagged.forward(prefill(8)).predicted_s == pytest.approx(14.0)
    assert tagged.refused_commands() == ()


HOOKS = ("_build_draft_model", "load_model", "arm_aux_capture")


def _drafters(monkeypatch, gone=None):
    """A stand-in for ATOM's drafter module: `Drafter` defining each of `HOOKS`
    but `gone`, a direct flavor, and a flavor below it that builds its own
    model."""
    hooks = {h: lambda self, *a: None for h in HOOKS if h != gone}
    Drafter = type("Drafter", (), hooks)

    class Direct(Drafter):
        pass

    class Deeper(Direct):
        def _build_draft_model(self, model_class):
            return torch.empty(1)

        def load_model(self, target):
            raise AssertionError("read the checkpoint")

    module = SimpleNamespace(Drafter=Drafter)
    monkeypatch.setitem(sys.modules, "atom.spec_decode.drafter", module)
    return Deeper


def test_a_flavor_below_a_direct_subclass_builds_on_meta_and_reads_nothing(
    monkeypatch,
):
    deeper = _drafters(monkeypatch)
    drafts_on_meta()
    assert deeper()._build_draft_model(None).device.type == "meta"
    deeper().load_model(None)


@pytest.mark.parametrize("gone", HOOKS)
def test_a_drafter_hook_atom_renamed_is_refused_by_name(monkeypatch, gone):
    _drafters(monkeypatch, gone)
    with pytest.raises(RunnerRefusal, match=f"no longer defines {gone}"):
        drafts_on_meta()


def test_pipeline_parallel_speculation_is_refused_before_the_drafter():
    r = object.__new__(NonAllocatingRunner)
    r.config = MockConfig(
        pipeline_parallel_size=2,
        speculative_config=SimpleNamespace(synthetic_acceptance_rates=[0.5]),
    )
    with pytest.raises(RunnerRefusal, match="pipeline parallel"):
        r._build_and_load_model(object)


def test_piecewise_capture_under_a_drafter_is_refused():
    config = SimpleNamespace(enforce_eager=False, speculative_config=object())
    r = SimpleNamespace(config=config, _piecewise_cg_active=lambda: True)
    with pytest.raises(ValueError, match="draft q bucket"):
        _graph_pool(r, None, None)


@pytest.mark.parametrize("k, top", [(None, 64), (1, 32), (3, 16)])
def test_the_ladder_charges_each_sequence_its_drafts(k, top):
    spec = k and SimpleNamespace(num_speculative_tokens=k)
    config = SimpleNamespace(
        max_num_seqs=256,
        max_num_batched_tokens=64,
        capture_sizes=[1, 2, 4, 8, 16, 32, 64],
        speculative_config=spec,
    )
    assert schedulable_ladder(config)[-1] == top
