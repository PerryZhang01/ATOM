# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""What a step reports, for a runner that predicts steps instead of running them.

ATOM's runner does not hand the scheduler the tokens of the step it was just
given. Three rules govern what it hands over instead, and each fails quietly
rather than loudly when a replacement gets it wrong.

**Tokens are one step late.** `tokenIDProcessor.is_deferred_out` is true
whenever the pipeline is a single stage, which is the ordinary case. The real
runner starts an asynchronous copy of this step's sampled ids to the host and
reports the copy the *previous* step started, so the reply carries the previous
batch's request ids and the previous batch's tokens with the deferred flag set.
`Scheduler.postprocess` is written against exactly that: it appends one
placeholder token per sequence at the end of every deferring call and
overwrites that placeholder on the next one. Reporting the current batch's
tokens with the flag clear offers them to a loop that has not yet appended a
placeholder for them, and the request is then not offered again until a later
batch happens to include it.

**The unit of that lag is one output-producing step, not one step.** The real
runner returns before sampling when the whole batch is middle chunks of chunked
prefills, so the queued copy is neither read nor replaced and the tokens of the
last output-producing step stay queued across however many middle chunks
follow. A replacement that carries its tokens forward by one *step* surfaces
them early, by the number of middle chunks in between.

**A step that produces nothing still reports its request ids.** The early
return names the batch's requests with an empty token list, and nothing reads
it on any path that exists today. A single-stage scheduler
(`Scheduler.postprocess` in `atom/model_engine/scheduler.py`) skips every
request in such a batch, on `seq.is_partial_prefill`, before it reads a token,
because they are all still mid-prompt; a pipeline head drops the entry
without waiting for a reply at all. The third path is the deferred flag, and
it has two readers above that per-seq loop: one folds it into
`need_placeholder`, and the other widens the placeholder count by one.
`need_placeholder` in turn has two uses, and both sit behind a gate -- the
prefix-hash one is inside the loop, below the same skip, and the placeholder
loop at the end touches only a sequence that is RUNNING and not mid-prefill
-- while the partial-prefill update at the top flags every member of a batch
of middle chunks partial before either gate reads it. The sequences that loop
walks are the batch's own, since `Scheduler.schedule` returns the dict it
built the batch from on both its prefill and its decode path, so this holds
for any batch rather than for the ones that have been tried.
So this one is mirrored from ATOM's own early return rather than derived from
what a caller needs -- there is no caller to derive it from, and a reply that
differs from the original differs in silence until one appears.

This module holds those rules and imports nothing from the engine, so it can be
exercised where there is no driver. It builds the keyword arguments of a batch
output rather than the object itself, for the same reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def filler_token_ids(
    tokenizer: Any, eos_token_id: Any, stop_token_ids: Any
) -> list[int]:
    """The ids a predicted step may report, one picked per request.

    A predicted step has no logits, so what a request generates is not modelled
    -- only how many steps it takes to generate it. Three readers still see the
    ids. The scheduler ends a request that emits the end-of-text id or a
    configured stop id, so either would cut every request short at its first
    token. The serving path's stream detokenizer moves its window only when the
    decoded text does not end in an incomplete character, so an id that decodes
    to part of a UTF-8 sequence holds the window open and makes detokenizing a
    stream quadratic in its length. And the prefix cache hashes generated
    blocks, chained on the prompt's: one id for every request makes two
    requests with a common prompt produce identical decode blocks, and the
    second hits the first's where a real run would not.

    So an id qualifies when it is not a stop id and decodes on its own, as the
    stream detokenizer decodes it, to ASCII letters and digits. That excludes
    every special token, which decodes to nothing there. Letters and digits
    only: whitespace and punctuation are what per-request stop strings are
    made of, and this runner never sees those.
    """
    stops = {int(t) for t in (stop_token_ids or ())}
    if eos_token_id is not None:
        stops.add(int(eos_token_id))
    return [
        token_id
        for token_id in range(len(tokenizer))
        if token_id not in stops
        and (text := tokenizer.decode([token_id], skip_special_tokens=True)).isascii()
        and text.isalnum()
    ]


def reports_previous_step(pipeline_parallel_size: Any) -> bool:
    """Do this runner's replies carry the previous step's tokens?

    The condition ATOM's token processor resolves from, and the same number the
    scheduler reads in the other direction: with more than one stage it
    advances chunked-prefill progress at schedule time instead of after the
    forward, and a stage reports the step it just ran. So the two arrangements
    never both apply, and one number selects between them.
    """
    return int(pipeline_parallel_size or 1) == 1


class DeferredTokenStream:
    """Carries one batch's reported tokens to the next output-producing step.

    One instance per runner, holding the single piece of state ATOM's own token
    processor holds: the last batch that produced output. The lag is a property
    of the step sequence rather than of any request in it; only the id each
    request is reported with depends on the request.

    Under speculation, `rates` is the per-position unconditional acceptance
    ATOM's forced acceptance resolves to, one per draft. Each row of a decode
    batch that verifies drafts accepts `a` of them, drawn from a stream seeded
    by `seed`: `a >= i + 1` with probability `rates[i]`, the distribution
    ATOM's synthetic rejection kernel draws. Every row reports one draft id per
    rate.
    """

    def __init__(
        self,
        token_ids: list[int],
        deferred: bool = True,
        rates: list[float] | None = None,
        seed: int = 0,
    ) -> None:
        self.token_ids = token_ids
        self.deferred = deferred
        self.prev_batch: Any = None
        self.rates = rates
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def is_pure_middle_chunk(batch: Any) -> bool:
        """Does this batch sample nothing at all?

        The question ATOM's runner asks before it returns early: a batch with
        no decode sequence and no prefill sequence on its final chunk.
        """
        return not batch.produces_output()

    def step(self, batch: Any) -> dict[str, Any]:
        """The batch output a predicted step reports, as constructor arguments."""
        if self.is_pure_middle_chunk(batch):
            # Same requests, no tokens, and the deferred flag left clear. The
            # flag is passed rather than left to the constructor default
            # because its being clear is the load-bearing part of this reply.
            return {
                "req_ids": list(batch.req_ids),
                "token_ids": [],
                "num_rejected": None,
                "num_bonus": None,
                "draft_token_ids": None,
                "is_deferred_out": False,
            }
        if not self.deferred:
            return self._reply(batch, deferred=False)
        prev, self.prev_batch = self.prev_batch, batch
        # The first output-producing step of a run has nothing queued behind
        # it, and reports no request at all rather than a row of tokens that no
        # sequence holds a placeholder for.
        return self._reply(prev, deferred=True)

    def _reply(self, source: Any, deferred: bool) -> dict[str, Any]:
        """One token per request of *source*, which may be no batch at all."""
        req_ids = [] if source is None else list(source.req_ids)
        # Sized by the batch whose tokens these are, since that is the batch
        # the scheduler indexes them with. Zero on a row that verified nothing.
        width = 0 if source is None else int(source.total_seqs_num)
        accepted, rejected = np.zeros((2, width), dtype=np.int32)
        k = 0 if self.rates is None or source is None else source.num_spec_step
        if k and not source.total_tokens_num_prefill:  # ATOM verifies no prefill
            u = self.rng.random((width, 1))
            accepted[:] = (u < np.asarray(self.rates[:k])).sum(axis=1)
            rejected[:] = k - accepted
        # Picked by request id: requests fewer than len(ids) apart never share one.
        ids = [self.token_ids[int(r) % len(self.token_ids)] for r in req_ids]
        drafts = None
        if self.rates:
            drafts = np.tile(np.array(ids, np.int32)[:, None], (1, len(self.rates)))
        return {
            "req_ids": req_ids,
            "token_ids": [(i,) * (a + 1) for i, a in zip(ids, accepted)],
            "num_rejected": rejected,
            "num_bonus": accepted,
            "draft_token_ids": drafts,
            "is_deferred_out": deferred,
        }
