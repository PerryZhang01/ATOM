# SPDX-License-Identifier: MIT
"""Building a `BatchView` from a scheduled batch, and the rung it supplies.

The backend that prices a step is handed a projection, and two properties of
that projection are ones the type cannot hold: that there is one row per
request the scheduler scheduled, and that `capture_rung` is a width a graph was
captured at rather than a number somebody typed. Both belong to the builder, so
both are asserted here.

The batches are ATOM's own. A real `Scheduler` runs with chunked prefill
against a real block manager, and the run is driven long enough that its steps
include two-request prefill chunks, middle chunks with history behind them, and
decode. The same run is driven a second time under `pipeline_parallel_size=2`,
which is the one configuration in which the scheduler's sequences and its batch
disagree. Nothing is hand-built except the runner stand-in, which exists only
to carry a capture ladder and an eager flag, and the three-field sequence
snapshot described below.

Two things deliberately not re-derived here, because a re-derivation that
agrees with itself proves nothing: the per-request attention sums are compared
against `Scheduler.compute_detailed_aggregates`, and the rung against the
ladder search inside `ForwardMode.decide`.

**Why the sequences are snapshotted.** A `Sequence` keeps moving: the same
object that was mid-prefill at step 1 is decoding by step 9, so a test that
re-reads it at assertion time is reading a later step's state and not the one
its batch was built from. Each step therefore keeps the three fields the
projection reads, frozen at the moment the batch was built. The live objects
are kept too, and one test uses them for exactly that divergence.
"""

from itertools import count
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
from conftest import MockConfig

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.shape import (
    sum_context,
    sum_query_context,
    sum_query_square,
)
from atom.compass.runner.overrides import NonAllocatingRunner, RunnerRefusal
from atom.compass.runner.projection import capture_rung, project, request_rows
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceType
from atom.sampling_params import SamplingParams

# The two switches that make the scheduler publish its attention aggregates.
# Both are plain attributes -- the public `profile_active`, and an env flag read
# once into `_detailed_annotation_enabled` -- so they are set directly. The
# start-profile RPC flips them too, and also starts a real torch profiler on the
# runner and writes trace files, which a run with no device has no business
# doing.
PUBLISHING = SimpleNamespace(profile_active=True, _detailed_annotation_enabled=True)

CHUNK = 256
PROMPT_LONG = 2048
PROMPT_SHORT = 512
STEPS = 20

# A ladder of the shape capture leaves behind: ascending, with the eager
# fallback 0 at the bottom. Ascending is what `ForwardMode.decide` searches.
LADDER = [0, 1, 2, 4, 8]


class Step(NamedTuple):
    batch: object
    seqs: dict  # the three fields, frozen when the batch was built
    live: dict  # the same sequences, still moving
    rows: tuple  # empty when the projection refused this step
    refusal: object  # the refusal's message, or None


def runner(ladder=LADDER, enforce_eager=False, dp_size=1):
    """The three attributes a rung is read from, and nothing else."""
    return SimpleNamespace(
        capture_sizes_np=np.asarray(ladder, dtype=np.int32),
        enforce_eager=enforce_eager,
        config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=dp_size)
        ),
    )


def _captured(declared, max_num_seqs, max_num_batched_tokens):
    """A `NonAllocatingRunner` before capture: declared sizes, `[0]` held."""

    class Captured(NonAllocatingRunner):
        enforce_eager = False
        capture_sizes_np = np.asarray([0], dtype=np.int32)
        config = SimpleNamespace(
            capture_sizes=declared,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            parallel_config=SimpleNamespace(data_parallel_size=1),
        )

    return Captured()


def _frozen(seqs):
    return {
        req_id: SimpleNamespace(
            type=seq.type,
            num_tokens=seq.num_tokens,
            num_cached_tokens=seq.num_cached_tokens,
        )
        for req_id, seq in seqs.items()
    }


def _batch_like(req_ids, scheduled, context):
    """A stand-in carrying the four fields `request_rows` reads off a batch."""
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
        context_lens=np.asarray(context, dtype=np.int32),
        total_seqs_num=len(req_ids),
        num_spec_step=0,
    )


def _scheduler(chunk=CHUNK, **overrides):
    """ATOM's own scheduler, chunked prefill against a real block manager."""
    Sequence.counter = count()
    return Scheduler(
        MockConfig(
            kv_cache_block_size=16,
            num_kvcache_blocks=1024,
            enable_chunked_prefill=True,
            max_num_seqs=8,
            max_num_batched_tokens=2 * chunk,
            long_prefill_token_threshold=chunk,
            max_model_len=4096,
            **overrides,
        )
    )


def _drive(steps=STEPS, chunk=CHUNK, **overrides):
    """Run ATOM's scheduler and keep every step it produced.

    Two prompts of different lengths, a token budget two chunks wide and a
    per-request chunk cap: both prompts share the first steps, the long one is
    then issued a chunk at a time on its own, and once it is through, every
    later step is a decode of both.

    A refused step is kept rather than raised, because whether the projection
    refuses is itself the measurement under `pipeline_parallel_size=2`.
    """
    scheduler = _scheduler(chunk=chunk, **overrides)
    params = SamplingParams(max_tokens=200)
    for length in (PROMPT_LONG, PROMPT_SHORT):
        scheduler.add(Sequence(list(range(1, length + 1)), 16, sampling_params=params))
    trace = []
    for _ in range(steps):
        batch, seqs = scheduler.schedule()
        if not batch.req_ids:
            break
        Scheduler.compute_detailed_aggregates(PUBLISHING, batch, seqs)
        frozen = _frozen(seqs)
        try:
            rows, refusal = request_rows(batch, frozen), None
        except RunnerRefusal as refused:
            rows, refusal = (), str(refused)
        trace.append(Step(batch, frozen, dict(seqs), rows, refusal))
        scheduler.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[(7,) for _ in batch.req_ids],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )
    return trace


@pytest.fixture
def trace():
    return _drive()


@pytest.fixture
def pipeline_trace():
    return _drive(pipeline_parallel_size=2)


def _multi(trace):
    """The first step the scheduler put two requests in."""
    for step in trace:
        if len(step.batch.req_ids) > 1:
            return step
    raise AssertionError("no step carried more than one request")


def _decode(trace):
    for step in trace:
        if step.rows and all(row.decode for row in step.rows):
            return step
    raise AssertionError("the run produced no decode step")


class TestTheRun:
    """What the driven run contains, asserted before anything reads it.

    Empty, one-row and all-prefill are what a run that did nothing degrades to,
    and each would satisfy a row-count assertion made over whatever steps
    remained. So the shape of the run is a test of its own, and the counts here
    are the ones every test below depends on.
    """

    def test_the_run_reached_every_kind_of_step(self, trace):
        multi = [s for s in trace if len(s.batch.req_ids) > 1]
        decode = [s for s in trace if s.batch.total_seqs_num_decode]
        middle = [
            s
            for s in trace
            if any(not r.decode and r.cached_tokens > 0 for r in s.rows)
        ]
        assert len(trace) == STEPS
        assert len(multi) >= 2
        assert len(decode) >= 2
        assert len(middle) >= 2

    def test_no_step_was_empty(self, trace):
        assert all(step.rows for step in trace)

    def test_no_step_of_this_run_was_refused(self, trace):
        """The control for every count below that is stated as a refusal: at
        `pipeline_parallel_size=1` the projection declines nothing, so a
        refusal counted elsewhere is that configuration's and not the
        harness's."""
        assert [step.refusal for step in trace] == [None] * STEPS


class TestTheRowCount:
    def test_one_row_per_scheduled_request_on_every_step(self, trace):
        """Three counts, not two: the rows, the request ids the batch carries,
        and the request count the scheduler recorded separately when it built
        the batch. A projection built off the wrong one of those agrees with
        the other two and not the third."""
        for step in trace:
            assert len(step.rows) == len(step.batch.req_ids)
            assert len(step.rows) == step.batch.total_seqs_num

    def test_a_two_request_batch_does_not_project_to_one_row(self, trace):
        """Two rows, and a measurement of what one row would have said.

        A one-row `BatchView` is legal and is `tokens x history` exactly, so
        the collapse is not something the type refuses -- it is something that
        prices differently. Both chunks here are the same size, and that is the
        clean case: the cross term of one row holding both is `2q x 2c` against
        the `2qc` the two rows sum to, so it is exactly doubled."""
        step = _multi(trace)
        assert len(step.rows) == 2
        collapsed = RequestShape(
            sum(r.query_tokens for r in step.rows),
            sum(r.context_tokens for r in step.rows),
            step.rows[0].decode,
        )
        assert len({r.query_tokens for r in step.rows}) == 1
        assert sum_query_context((collapsed,)) == 2 * sum_query_context(step.rows)

    def test_a_missing_request_is_refused_rather_than_dropped(self, trace):
        """`zip` is how these two are read everywhere, and `zip` truncates.

        The second assertion is the control: paired by `zip` alone, the same
        two inputs make a one-row projection of a two-request batch."""
        step = _multi(trace)
        first = step.batch.req_ids[0]
        short = {first: step.seqs[first]}
        with pytest.raises(RunnerRefusal, match="one row per scheduled request"):
            request_rows(step.batch, short)
        assert len(list(zip(short.values(), step.batch.num_scheduled_tokens))) == 1

    def test_sequences_in_another_order_are_refused(self, trace):
        step = _multi(trace)
        backwards = {k: step.seqs[k] for k in reversed(list(step.seqs))}
        with pytest.raises(RunnerRefusal, match="one row per scheduled request"):
            request_rows(step.batch, backwards)

    def test_a_batch_whose_own_request_count_disagrees_is_refused(self, trace):
        """The third count, which neither of the two above can see.

        A scheduled batch records its request count separately from its
        request ids, so a batch carrying two ids and a count of three is one
        whose two records of the same thing came apart. Rows built off the ids
        match the ids, and say nothing."""
        step = _multi(trace)
        overstated = _batch_like(
            step.batch.req_ids,
            step.batch.num_scheduled_tokens,
            step.batch.context_lens,
        )
        overstated.total_seqs_num = len(step.batch.req_ids) + 1
        with pytest.raises(RunnerRefusal, match="rows for the"):
            request_rows(overstated, step.seqs)


class TestTheAttentionSums:
    """Against `Scheduler.compute_detailed_aggregates`, not a re-derivation.

    What the comparison catches is the row set and the query counts. The
    builder refuses when its own two readings of a row's history disagree, so
    an `N_KV` that survives that guard agrees with the scheduler's by
    construction; a query count on a decode row does not go through the guard,
    and neither does a dropped or duplicated row.

    What it cannot catch is the `decode` flag. None of the three sums reads
    it -- they are functions of `query_tokens` and `context_tokens` alone --
    while the flag is what decides which half of the cost form prices a row.
    That flag is held elsewhere: by the history check, whose two arms
    `TestBothKindsInOneBatch` drives in a single call, and by `TestTheRung`,
    where a step with no decode row answers `None`.
    """

    def test_the_three_sums_equal_atom_s_own_on_every_step(self, trace):
        for step in trace:
            assert sum_query_square(step.rows) == step.batch.detailed_sqsq
            assert sum_query_context(step.rows) == step.batch.detailed_sqsk
            assert sum_context(step.rows) == step.batch.detailed_sk

    def test_the_sums_compared_are_neither_zero_nor_absent(self, trace):
        """Zero and `None` are what publishing off, and a run that scheduled
        nothing, produce. Without this the equality above passes on a run in
        which the scheduler published nothing and the projection built
        nothing."""
        for step in trace:
            published = (
                step.batch.detailed_sqsq,
                step.batch.detailed_sqsk,
                step.batch.detailed_sk,
            )
            assert None not in published
            assert min(published) > 0

    def test_publishing_is_what_makes_those_fields_present(self, trace):
        """The control for the switch, so the comparison cannot be vacuous."""
        step = _multi(trace)
        silent = SimpleNamespace(
            profile_active=False, _detailed_annotation_enabled=True
        )
        step.batch.detailed_sqsq = None
        Scheduler.compute_detailed_aggregates(silent, step.batch, step.seqs)
        assert step.batch.detailed_sqsq is None
        Scheduler.compute_detailed_aggregates(PUBLISHING, step.batch, step.seqs)
        assert step.batch.detailed_sqsq == sum_query_square(step.rows)

    def test_a_decode_row_takes_its_query_count_from_the_batch(self, trace):
        """A decode step verifies `mtp_k + 1` tokens under speculation.

        The history check cannot see a wrong query count on a decode row --
        the sequence's own length settles that row's context whatever the
        query is -- so the count is read off the batch and the scheduler's
        `sqsq` is what would catch it being read off anything else."""
        step = _decode(trace)
        speculative = _batch_like(
            step.batch.req_ids,
            step.batch.num_scheduled_tokens * 3,
            step.batch.context_lens,
        )
        rows = request_rows(speculative, step.seqs)
        assert [r.query_tokens for r in rows] == [
            3 * int(n) for n in step.batch.num_scheduled_tokens
        ]
        assert sum_query_square(rows) == 9 * step.batch.detailed_sqsq


class TestWidthsAnInt32ProductCannotHold:
    """The two integers are cast out of `np.int32`, and these rows say so.

    `num_scheduled_tokens` and `context_lens` are `np.int32`, and every shape
    sum multiplies two of them. `Scheduler.compute_detailed_aggregates` casts
    to Python `int` for this reason and spends three comment lines on it --
    "`nq*nq` ... would overflow once a prefill/chunk exceeds ~46341 tokens
    ... silently corrupting the estimate". A wrapped product raises nothing
    anywhere downstream; it is a zero, or a negative, price.

    The third assertion is what keeps the first two from being satisfied by a
    width that never wraps: it states that this row's product is outside the
    32-bit range in the first place.
    """

    @pytest.mark.parametrize(
        "query, context",
        [(65536, 131072), (46341, 46341)],
        ids=["a-chunk-whose-square-is-2**32", "the-narrowest-chunk-that-wraps"],
    )
    def test_a_row_this_wide_sums_to_the_true_product(self, query, context):
        batch = _batch_like([5], [query], [context])
        seqs = {
            5: SimpleNamespace(
                type=SequenceType.PREFILL,
                num_tokens=context,
                num_cached_tokens=context - query,
            )
        }
        (row,) = request_rows(batch, seqs)
        assert sum_query_square((row,)) == query * query
        assert sum_query_context((row,)) == query * context
        assert min(query * query, query * context) > 2**31 - 1


def _mixed_batch():
    """One prefill chunk with history behind it, and one decoding request."""
    return _batch_like([11, 12], [CHUNK, 1], [2 * CHUNK, PROMPT_SHORT + 1])


def _mixed_seqs():
    return {
        11: SimpleNamespace(
            type=SequenceType.PREFILL,
            num_tokens=PROMPT_LONG,
            num_cached_tokens=CHUNK,
        ),
        12: SimpleNamespace(
            type=SequenceType.DECODE,
            num_tokens=PROMPT_SHORT + 1,
            num_cached_tokens=0,
        ),
    }


class TestBothKindsInOneBatch:
    """The one shape in which both arms of the history check run in one call.

    `schedule()` returns its prefill batch before it can add a decode row
    (`scheduler.py:1736`), so no batch this scheduler builds holds both kinds
    and the per-row branch is otherwise only ever exercised across steps. The
    batch here is the same stand-in the refusal tests use.

    Each sequence's *other* field is set to a number the opposite arm could
    not produce -- the prefill's `num_tokens` is its whole prompt rather than
    its chunk's history, and the decoding request's `num_cached_tokens` is 0 --
    so a row that took the wrong arm disagrees with `context_lens` and refuses
    instead of quietly agreeing. The second test is that control, stated as a
    measurement.
    """

    def test_a_prefill_row_and_a_decode_row_project_side_by_side(self):
        rows = request_rows(_mixed_batch(), _mixed_seqs())
        assert [r.decode for r in rows] == [False, True]
        assert [r.query_tokens for r in rows] == [CHUNK, 1]
        assert [r.context_tokens for r in rows] == [2 * CHUNK, PROMPT_SHORT + 1]
        assert rows[0].cached_tokens == CHUNK

    @pytest.mark.parametrize("req_id", [11, 12])
    def test_a_row_read_under_the_other_kind_s_arm_is_refused(self, req_id):
        seqs = _mixed_seqs()
        other = seqs[req_id]
        other.type = (
            SequenceType.PREFILL
            if other.type == SequenceType.DECODE
            else SequenceType.DECODE
        )
        with pytest.raises(RunnerRefusal, match="differ by history"):
            request_rows(_mixed_batch(), seqs)


class TestTheRung:
    """The width comes off the ladder, through the rule ATOM dispatches by."""

    def test_a_decode_step_takes_the_smallest_captured_width_that_holds_it(self, trace):
        step = _decode(trace)
        rung = capture_rung(step.batch, runner())
        assert rung in LADDER
        assert rung == min(g for g in LADDER if g >= len(step.rows))

    def test_the_rung_is_the_ladder_s_width_and_not_the_batch_s(self, trace):
        """A ladder with no rung at the batch's own width settles which it is.

        With the batch two requests wide and 2 taken out of the ladder, a rung
        read off the batch still answers 2 and a rung read off the ladder
        answers 4."""
        step = _decode(trace)
        assert len(step.rows) == 2
        assert capture_rung(step.batch, runner(ladder=[0, 1, 4, 8])) == 4

    def test_a_prefill_step_replays_no_graph(self, trace):
        prefill = [s for s in trace if any(not r.decode for r in s.rows)]
        assert len(prefill) >= 2
        for step in prefill:
            assert capture_rung(step.batch, runner()) is None

    def test_an_eager_runner_replays_no_graph(self, trace):
        step = _decode(trace)
        assert capture_rung(step.batch, runner(enforce_eager=True)) is None

    def test_a_batch_wider_than_the_widest_capture_replays_no_graph(self, trace):
        step = _decode(trace)
        assert len(step.rows) > 1
        assert capture_rung(step.batch, runner(ladder=[0, 1])) is None

    def test_the_eager_fallback_ladder_replays_no_graph(self, trace):
        """What a runner that captured nothing holds.

        `capture_sizes` is `[0]` from construction, and a runner that captures
        no graph never widens it, so every step of such a run comes back
        `None` here and the padding term is never charged."""
        for step in trace:
            assert capture_rung(step.batch, runner(ladder=[0])) is None

    @pytest.mark.parametrize(
        "ladder, wrong_rung",
        [([8, 4, 2, 1, 0], 8), ([0, 1, 8, 4, 2], 8)],
        ids=["descending", "one-pair-out-of-order"],
    )
    def test_a_ladder_that_is_not_ascending_is_refused(self, trace, ladder, wrong_rung):
        """`decide` resolves the rung by binary search, whose precondition is
        an ascending ladder, and this forwards whatever the runner holds.

        The second assertion is why refusing is the answer rather than a
        comment: the same widths out of order answer a rung four times the
        batch's own width, and `BatchView` accepts that -- a rung wider than
        its rows is the one direction rows cannot contradict, so the padding
        would be priced on a graph no capture ever recorded."""
        step = _decode(trace)
        assert sorted(ladder) == sorted(LADDER)
        with pytest.raises(RunnerRefusal, match="not ascending"):
            capture_rung(step.batch, runner(ladder=ladder))
        unchecked = BatchView(step.rows, capture_rung=wrong_rung)
        assert unchecked.capture_rung == wrong_rung > len(step.rows)

    @pytest.mark.parametrize(
        "declared, max_num_seqs, max_num_batched_tokens",
        [([8, 2, 4, 1], 3, 16384), ([1, 2, 4, 8], 512, 2)],
        ids=["sequence-bound", "token-bound"],
    )
    def test_a_runner_that_captured_nothing_replays_atom_s_ladder(
        self, trace, declared, max_num_seqs, max_num_batched_tokens
    ):
        """`capture_cudagraph` captures nothing and still leaves the ladder
        ATOM's capture would: declared, narrowed to the schedulable width,
        ascending. Left at the construction-time `[0]`, every decode step of a
        graphs-on run came back eager."""
        step = _decode(trace)
        assert len(step.rows) == 2
        r = _captured(declared, max_num_seqs, max_num_batched_tokens)
        assert r.capture_cudagraph() == (0.0, [], 0)
        assert r.capture_sizes_np.tolist() == [1, 2]
        assert capture_rung(step.batch, r) == 2

    def test_a_ladder_narrowed_to_nothing_is_refused(self):
        """ATOM's capture asserts on an empty narrowed ladder, so such a
        deployment never starts; pricing its decode steps as eager would
        report a run that cannot happen."""
        r = _captured([16, 32], 8, 16384)
        with pytest.raises(RunnerRefusal, match=r"\[16, 32\].*bound 8"):
            r.capture_cudagraph()

    @pytest.mark.parametrize(
        "config",
        [
            None,
            SimpleNamespace(),
            SimpleNamespace(parallel_config=SimpleNamespace()),
            SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=0)),
            SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=False)),
            SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=-1)),
        ],
        ids=[
            "no-config",
            "no-parallel-config",
            "no-data-parallel-size",
            "zero",
            "false",
            "negative",
        ],
    )
    def test_a_runner_that_states_no_parallel_size_is_refused(self, trace, config):
        """The module's only place a default could stand in for an answer.

        Each of the three lookups on the way to `data_parallel_size` is a
        place a differently shaped runner stops, and reading a stop as 1 rank
        prices a group's step as one rank's without saying so. Zero,
        `False` and a negative are not widths either, and each passes the
        `> 1` check in `capture_rung` as the single-rank case."""
        step = _decode(trace)
        bare = SimpleNamespace(
            capture_sizes_np=np.asarray(LADDER, dtype=np.int32),
            enforce_eager=False,
            config=config,
        )
        with pytest.raises(RunnerRefusal, match="states no"):
            capture_rung(step.batch, bare)


class TestTheWholeProjection:
    def test_a_decode_step_projects_its_rows_and_a_captured_width(self, trace):
        """The padding is the rung's rectangle, and a rung wider than the rows
        is what makes that visible: at a rung equal to the row count the two
        rectangles coincide, and a padding computed off the batch instead of
        off the graph would read the same."""
        step = _decode(trace)
        view = project(step.batch, step.seqs, runner(ladder=[0, 1, 4, 8]))
        assert view.requests == step.rows
        assert view.capture_rung == 4
        widest = max(r.context_tokens for r in step.rows)
        assert view.graph_padding == 4 * widest - sum_context(step.rows)
        assert view.graph_padding > len(step.rows) * widest - sum_context(step.rows)

    def test_a_prefill_step_projects_its_rows_and_no_width(self, trace):
        step = trace[0]
        view = project(step.batch, step.seqs, runner())
        assert view.requests == step.rows
        assert view.capture_rung is None
        assert view.graph_padding == 0


class TestSequencesThatHaveMovedOn:
    """The reading off the batch and the reading off the sequence must agree.

    They stop agreeing when the sequences handed in are not the ones the batch
    was built from -- and, under pipeline parallelism, when they are exactly
    those sequences. `TestPipelineParallelism` drives the second case; these
    two are the first.
    """

    def test_the_live_sequences_of_an_earlier_step_are_refused(self, trace):
        """Not a hand-built divergence: these are the same objects, nineteen
        steps later, and the run moved them."""
        step = trace[0]
        with pytest.raises(RunnerRefusal, match="differ by history"):
            request_rows(step.batch, step.live)

    def test_an_advanced_prefill_offset_is_refused(self, trace):
        step = trace[0]
        first = step.batch.req_ids[0]
        advanced = dict(step.seqs)
        advanced[first] = SimpleNamespace(
            type=SequenceType.PREFILL,
            num_tokens=step.seqs[first].num_tokens,
            num_cached_tokens=step.seqs[first].num_cached_tokens + CHUNK,
        )
        with pytest.raises(RunnerRefusal, match="differ by history"):
            request_rows(step.batch, advanced)

    def test_the_same_sequences_unmoved_are_accepted(self, trace):
        step = trace[0]
        assert request_rows(step.batch, step.seqs) == step.rows


class TestPipelineParallelism:
    """The configuration in which the scheduler's own two records disagree.

    `advance_on_schedule` is on whenever `pipeline_parallel_size > 1`
    (`scheduler.py:994`), and `_advance_prefill_on_schedule` (`:2271`) adds
    each chunk to its sequence's `num_cached_tokens` *after* the batch has
    snapshotted the pre-advance offsets (`:1730`). So the mapping `schedule()`
    returns is a chunk ahead of the batch built from it -- not a stale mapping
    and not a caller's mistake, which is why the refusal names the advance as
    well as the staleness.

    This is the same run as `trace`, one kwarg apart, and the counts below are
    read against that run: it refuses nothing.
    """

    def test_the_scheduler_advances_its_sequences_only_under_this_setting(self):
        """The switch itself, so a refusal counted below cannot be some other
        difference between the two runs."""
        assert _scheduler(pipeline_parallel_size=2).advance_on_schedule
        assert not _scheduler().advance_on_schedule

    def test_every_prefill_step_refuses_and_every_decode_step_projects(
        self, trace, pipeline_trace
    ):
        """The consequence, decomposed rather than summarised.

        `advance_on_schedule` moves `num_cached_tokens`, which is only read on
        the prefill arm; a decode row's `N_KV` is `seq.num_tokens` on both
        sides and does not move. So the projection does not merely decline to
        pick a side under pipeline parallelism -- it produces no prefill row
        at all, on step 0 of any such run."""
        refused = [s for s in pipeline_trace if s.refusal is not None]
        projected = [s for s in pipeline_trace if s.rows]
        assert len(pipeline_trace) == STEPS
        assert len(refused) == 8
        assert len(projected) == 12
        assert all(s.batch.total_seqs_num_decode == 0 for s in refused)
        assert all(
            s.batch.total_seqs_num_decode == len(s.batch.req_ids) for s in projected
        )
        assert [s.refusal for s in trace] == [None] * STEPS

    def test_the_refusal_names_the_advance_and_not_only_a_stale_mapping(
        self, pipeline_trace
    ):
        """These *are* the sequences the batch was built from. A message
        saying otherwise sends whoever hits it in a pipeline-parallel run
        looking for a plumbing bug that is not there."""
        first = pipeline_trace[0]
        assert first.refusal is not None
        assert "advance_on_schedule" in first.refusal
        assert "pipeline-parallel" in first.refusal

    def test_each_sequence_is_exactly_one_chunk_ahead_of_its_batch(
        self, pipeline_trace
    ):
        """The size of the divergence is what says it is the advance.

        Any disagreement would refuse; a disagreement of exactly this step's
        own chunk, on every request, is the advance and nothing else."""
        step = pipeline_trace[0]
        assert len(step.batch.req_ids) == 2
        for index, seq in enumerate(step.seqs.values()):
            chunk = int(step.batch.num_scheduled_tokens[index])
            settled = int(seq.num_cached_tokens) + chunk
            assert settled - int(step.batch.context_lens[index]) == chunk

    def test_atom_s_own_aggregate_is_the_reading_that_moved(self, pipeline_trace):
        """Which of the two readings is wrong, recorded rather than repaired.

        `compute_detailed_aggregates` reads the live sequence, so under this
        setting it prices history the forward does not read. The forward reads
        `context_lens`. Repairing ATOM's annotation is not this file set."""
        step = pipeline_trace[0]
        batched = sum(int(n) for n in step.batch.context_lens)
        scheduled = sum(int(n) for n in step.batch.num_scheduled_tokens)
        assert (step.batch.detailed_sk, batched, scheduled) == (1024, 512, 512)
        assert step.batch.detailed_sk - batched == scheduled


class TestRowsTheShapesAreNotDefinedOver:
    """Refusals the projection leans on, pinned where it leans on them.

    Neither has a witness elsewhere in the tree. A `RequestShape` that accepted
    these would let the projection emit a row whose shape sums mean nothing --
    a request computing no tokens, or attention reading less than the step
    writes.
    """

    @pytest.mark.parametrize(
        "scheduled, context, message",
        [(0, 100, "at least one token"), (100, 40, "fewer than")],
    )
    def test_a_row_outside_the_shapes_is_refused(self, scheduled, context, message):
        batch = _batch_like([3], [scheduled], [context])
        seqs = {
            3: SimpleNamespace(
                type=SequenceType.PREFILL,
                num_tokens=PROMPT_LONG,
                num_cached_tokens=context - scheduled,
            )
        }
        with pytest.raises(ValueError, match=message):
            request_rows(batch, seqs)


def test_an_empty_batch_has_no_shape_to_price():
    """The engine runs no forward for one, so one arriving here is a defect."""
    with pytest.raises(ValueError, match="no requests"):
        BatchView(request_rows(_batch_like([], [], []), {}))
