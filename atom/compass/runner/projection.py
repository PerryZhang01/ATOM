# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The projection a cost backend is priced from: one row per scheduled request.

`BatchView` and `RequestShape` live in `atom.compass.backends` because the scan
that forbids engine imports reads that package and nothing else. This module is
their other half and does the opposite: it reads a scheduled batch, the
sequences that batch was built from, and the runner's graph-capture ladder, and
turns them into those two types. Keeping the builder here is what lets it read
the engine at all.

Two things the type cannot check, both of which belong to whoever builds the
projection rather than to whoever prices it.

**The row count.** `BatchView` states, and measures, that a one-row batch is
`tokens x history` exactly -- the collapsed form that summing per request
exists to keep out -- and that nothing in the type ties its row count to the
number of requests a scheduler scheduled. It names this module as what closes
that. So the rows here are one per entry of
`ScheduledBatch.num_scheduled_tokens`, and a `seqs` mapping whose keys are not
the batch's own `req_ids`, in the batch's own order, is refused. That refusal
is the load-bearing line in this file rather than housekeeping: `zip` is how
both structures are read everywhere else in the tree, including by
`Scheduler.compute_detailed_aggregates`, and `zip` over a mapping one request
short drops the last row in silence -- which is a shorter batch that prices
without complaining, and at two requests is exactly the collapse.

**The rung.** The type refuses a `capture_rung` with no decode rows and one
narrower than the decode row count. It cannot refuse one that is far too wide,
because rows cannot contradict a width -- only a ladder of captured widths can
-- so a width supplied by a caller is charged at face value, and one decode row
at `capture_rung=10**9` prices 99.9999999 s of graph padding. Nothing here
takes a width from a caller. `ForwardMode.decide` is the rule ATOM dispatches a
real step by, and its answer is taken whole: `running_bs` where it says a graph
replays, `None` where it says one does not. A prefill step, an eager runner and
a batch wider than the widest captured size therefore all come back `None` from
that one call rather than from three conditions restated here. `decide` finds
the width by binary search and so has a precondition -- an ascending ladder --
which is checked here rather than assumed, because an unsorted ladder answers a
width that is not the smallest captured one holding the batch, and too wide is
the direction the type cannot refuse.

**Why each row's history is read twice.** `ScheduledBatch.context_lens` is the
`N_KV` ATOM computed when it built the batch -- a prefill chunk's cached tokens
plus the chunk, a decode's whole sequence -- and is what this builds rows from.
The sequence is read for its type and for a second derivation of that same
`N_KV`, from `Sequence.num_tokens` and `Sequence.num_cached_tokens`, which is
the one `Scheduler.compute_detailed_aggregates` takes. They must agree. A
disagreement has two causes and this module cannot tell them apart by looking
at the numbers, so the refusal names both. Either the sequences handed in are
not the ones the batch was built from -- a stale or a later step's mapping --
or they are exactly those sequences and the scheduler advanced them on
purpose: `Scheduler.advance_on_schedule` is on whenever
`pipeline_parallel_size > 1`, and `Scheduler._advance_prefill_on_schedule` adds
each chunk to its sequence's offsets after the batch has snapshotted them. Under
that configuration the two readings differ by a whole chunk of history on
every prefill row, and the projection refuses there instead of picking one of
the two readings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.runner.overrides import RunnerRefusal
from atom.model_engine.sequence import SequenceType
from atom.utils.forward_context import ForwardMode


def request_rows(batch: Any, seqs: dict[int, Any]) -> tuple[RequestShape, ...]:
    """One row per scheduled request, in the batch's own order.

    `batch.req_ids` is `list(seqs.keys())` taken when the batch was built, so
    requiring the two to be equal states both halves of the projection's
    guarantee -- the row count, and that row *i* describes the request row *i*
    of every parallel array on the batch describes.

    Both integers are cast out of `np.int32` before a row is built. The sums
    over these rows multiply two of them together, and the scheduler's own
    aggregate casts for the same reason and says so: past roughly 46341 tokens
    a 32-bit product wraps, which is not an error anywhere downstream but a
    smaller -- or negative -- price.
    """
    if list(seqs.keys()) != list(batch.req_ids):
        raise RunnerRefusal(
            f"the batch scheduled {list(batch.req_ids)} and the sequences handed "
            f"in are {list(seqs.keys())}; one row per scheduled request is what "
            "keeps a batch of two from being priced as the single collapsed row "
            "it would otherwise sum to, and every reader of these two pairs them "
            "positionally"
        )
    rows = []
    for index, (seq, scheduled) in enumerate(
        zip(seqs.values(), batch.num_scheduled_tokens)
    ):
        query = int(scheduled)
        decode = seq.type == SequenceType.DECODE
        context = int(batch.context_lens[index])
        settled = int(seq.num_tokens) if decode else int(seq.num_cached_tokens) + query
        if settled != context:
            raise RunnerRefusal(
                f"request {batch.req_ids[index]} reads {context} context tokens "
                f"off the batch and {settled} off its sequence; the two readings "
                "differ by history this step either did or did not compute. "
                "Either these are not the sequences this batch was built from, "
                "or they are and `advance_on_schedule` moved them on purpose "
                "after the batch snapshotted its offsets -- which it does on "
                "every pipeline-parallel run, and this cannot tell the two "
                "apart from the numbers"
            )
        rows.append(RequestShape(query, context, decode))
    if len(rows) != batch.total_seqs_num:
        raise RunnerRefusal(
            f"{len(rows)} rows for the {batch.total_seqs_num} requests the "
            "scheduler counted into this batch"
        )
    return tuple(rows)


def capture_rung(batch: Any, runner: Any) -> int | None:
    """The width of the graph this step replays, or `None` if it replays none.

    Asked of `ForwardMode.decide`, which is the rule the real runner dispatches
    by, rather than derived from the ladder here: deriving it is how a rung gets
    assigned to a prefill step, which replays nothing and would then be charged
    graph padding for a graph that did not run.
    """
    return _rung(forward_mode(batch, runner))


def _rung(mode: ForwardMode) -> int | None:
    return int(mode.running_bs) if mode.use_cudagraph else None


def dp_group() -> Any:
    """The DP group's CPU group: the one `ModelRunner.prepare_model` hands `decide`."""
    from aiter.dist.parallel_state import get_dp_group

    return get_dp_group().cpu_group


def forward_mode(batch: Any, runner: Any) -> ForwardMode:
    """The step's shape, settled by `ForwardMode.decide` as the real runner settles it.

    Under data parallelism this runs the step's collective, `sync_dp_metadata`,
    on the DP group, so every rank calls it once per step, as the real forward
    does; the group's rung and token counts come back from it.

    `tbo_on` and `local_tbo` are what `ModelRunner.prepare_model` passes: the
    config's `enable_tbo` and the runner's own `_local_tbo_eligibility`, which
    ride the collective, so the group agrees the two-batch-overlap split a real
    run agrees. The other arguments that do not reach the rung are passed at
    their inert values: `decide` settles a query length and a token count as
    well, and those are what `captured_tokens` and `is_block_drafter` feed. The
    width and the replay decision come from `capture_sizes`, `enforce_eager`,
    the attention builder's `step_needs_eager`, the batch and the collective
    alone.
    """
    dp_size = _data_parallel_size(runner)
    tbo_on = runner.config.enable_tbo
    ladder = np.asarray(runner.capture_sizes_np, dtype=np.int32)
    if bool(np.any(ladder[1:] < ladder[:-1])):
        raise RunnerRefusal(
            f"the capture ladder {ladder.tolist()} is not ascending, which is "
            "the precondition of the binary search `decide` resolves the rung "
            "with; out of order it answers a width that is not the narrowest "
            "captured one holding this batch, and a rung too wide is the one "
            "direction the rows cannot contradict"
        )
    return ForwardMode.decide(
        batch=batch,
        dp_size=dp_size,
        dp_group=dp_group() if dp_size > 1 else None,
        enforce_eager=bool(runner.enforce_eager),
        step_needs_eager=runner.attn_metadata_builder.step_needs_eager(batch),
        capture_sizes=ladder,
        captured_tokens=None,
        is_block_drafter=False,
        tbo_on=tbo_on,
        local_tbo=(
            runner._local_tbo_eligibility(batch) if tbo_on else (False, False, 0, 0)
        ),
        max_seqlen_q=batch.num_spec_step + 1,
    )


def _data_parallel_size(runner: Any) -> int:
    """How many ranks `decide` would settle a rung across, or a refusal.

    Absent, zero, negative or `False` is refused rather than read as one,
    the set ATOM's `ParallelConfig` refuses too. A runner shaped differently
    from the engine's own would otherwise be priced as a single rank on a
    default nothing checked; zero or a negative width would pass the `> 1`
    test in `capture_rung` as the single-rank case.
    """
    parallel = getattr(getattr(runner, "config", None), "parallel_config", None)
    size = getattr(parallel, "data_parallel_size", None)
    if size is None or size < 1:
        raise RunnerRefusal(
            "this runner states no width in "
            f"`config.parallel_config.data_parallel_size` (it reads {size!r}), "
            "and a rung is one rank's answer or a group's depending on it; "
            "reading that as 1 is the guess this would have to make to carry on"
        )
    return int(size)


def project(batch: Any, seqs: dict[int, Any], runner: Any) -> BatchView:
    """The whole projection: the batch's rows, and the rung it replayed."""
    return BatchView(
        request_rows(batch, seqs), capture_rung=capture_rung(batch, runner)
    )


def batch_view(batch: Any, mode: ForwardMode, runner: Any) -> BatchView:
    """The projection a worker builds: it holds the batch and not the sequences.

    A row's kind is the convention `ModelRunner.prepare_inputs` reads: the
    first `total_seqs_num_prefill` rows are prefill. The rung and the MoE
    gather's rows come from `mode`, the step's one `forward_mode`; a
    data-parallel runner handed a mode with no group counts is refused.
    Whether the step produces output is `ScheduledBatch.produces_output`, and
    the TBO split is the collective's `ub_tokens_across_dp`.

    The MoE gather's rows follow `running_tokens_are_unified`, as
    `FusedMoE.forward_impl_graph` does. Every rank decoding, the padded
    `all_gather` carries `running_tokens` rows from each rank, the group's
    agreed height, under `enforce_eager` too. Otherwise (a rank prefilling) the
    variable-length gather carries each rank's own count. An MTP drafter's
    first pass runs each rank's target token stream unpadded, so its gather is
    always the variable-length one (`EagleProposer.propose`); a block drafter
    pads its passes to the group and is left out.
    """
    draft_moe_rows = moe_tokens = None
    if mode.sync is not None:
        gathered = moe_tokens = int(mode.sync.num_tokens_across_dp.sum())
        if mode.running_tokens_are_unified:
            moe_rows = _data_parallel_size(runner) * mode.running_tokens
        else:
            moe_rows = gathered
        drafter = getattr(runner, "drafter", None)
        if drafter is not None and not drafter.is_block_drafter:
            draft_moe_rows = gathered
    elif _data_parallel_size(runner) > 1:
        raise RunnerRefusal(
            "this rank is one of a data-parallel group and its forward mode "
            "states no num_tokens_across_dp, so the MoE segment has no gathered "
            "row count to be priced from"
        )
    else:
        moe_rows = None
    prefill = batch.total_seqs_num_prefill
    rows = zip(batch.num_scheduled_tokens, batch.context_lens)
    return BatchView(
        tuple(
            RequestShape(int(q), int(c), i >= prefill) for i, (q, c) in enumerate(rows)
        ),
        capture_rung=_rung(mode),
        moe_rows=moe_rows,
        draft_moe_rows=draft_moe_rows,
        produces_output=batch.produces_output(),
        tbo_tokens=None if mode.sync is None else mode.sync.ub_tokens_across_dp,
        image_tokens=image_tokens(batch, runner),
        moe_tokens=moe_tokens,
    )


def image_tokens(batch: Any, runner: Any) -> int:
    """The image rows a step's forward handles, as `ModelRunner.run_model`
    handles them. With embedding spans, a payload still carrying pixel values
    encodes every span on this chunk, even one that fills none; a later chunk
    fills the spans it overlaps. Without spans, the image's embeddings sit at
    the model's image token."""
    from atom.model_engine.multimodal_runtime import embedding_indices

    media = getattr(batch, "multimodal_data", None) or {}
    count, offset = 0, 0
    for req, length, end in zip(
        batch.req_ids, batch.num_scheduled_tokens, batch.context_lens
    ):
        length, data = int(length), media.get(req)
        if data is not None and "embedding_spans" in data:
            spans = data["embedding_spans"]
            if "pixel_values" in data:
                count += sum(rows for _, rows in spans)
            else:
                count += len(embedding_indices(spans, int(end) - length, length)[0])
        elif data is not None:
            image = runner.config.multimodal_config.image_token_id
            mine = batch.scheduled_tokens[offset : offset + length]
            count += int(np.count_nonzero(mine == image))
        offset += length
    return count
