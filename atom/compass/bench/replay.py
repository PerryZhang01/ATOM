# SPDX-License-Identifier: MIT
"""A `ScheduledBatch` rebuilt from a step-record line, by a fixed, versioned rule.

A step-record line states a batch as its rows, in order, each with its query
tokens, its context tokens and whether it decodes, and whether the step
produces output. ATOM's forward also needs block tables, tokens and, for a
model that keeps per-request state, state slots. `ReplayRule.batch` supplies
them the same way for every line:

- Rows keep the line's order, and are all prefill or all decode: ATOM's
  runner runs a batch holding prefill tokens over its prefill rows alone, and
  its scheduler builds no batch that mixes the two.
- Each row is a new request, numbered by `Sequence`'s own counter, so no two
  replays share a request.
- Row i holds `ceil(context / block_size)` KV blocks, contiguous, from a fresh
  pool: row 0 starts at block 0, or the block `batch` is given, and each row
  starts where the one before ended.
- Every token of row i is the filler id `DeferredTokenStream` reports for a
  request numbered i, out of the ids `filler_token_ids` keeps for the model's
  tokenizer, as the simulated runner keeps them.
- Under a model that keeps per-request state, row i holds state slot i of a
  fresh pool, or that many past the slot `batch` is given.
- A prefill line that produces output has every row on its final chunk, one
  that does not has none: a line does not say which of its chunks ends a
  prompt, and ATOM's runner samples every row of a step that produces output.
- Every row samples with the request parameters the rule was given, ATOM's
  `SamplingParams` defaults unless stated.
- The batch carries the rule's `NAME` and `VERSION` as `replay_rule`, which a
  measurement records.

The line's `rung` is not read: the runner a batch is handed to settles its
own. The tokens are fillers, not the recorded ones, so a replayed MoE step may
route otherwise than the recorded step did.
"""

from atom.compass.runner.step_output import filler_token_ids
from atom.model_engine.llm_engine import InputOutputProcessor, _load_tokenizer
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import (
    Sequence,
    SequenceType,
    new_block_table,
)
from atom.sampling_params import SamplingParams

NAME = "fresh-contiguous"
VERSION = 1


class ReplayRule:
    """The rule for one engine configuration: its tokenizer, block size, KV
    pool, and whether its model keeps per-request state; and for the request
    parameters `sampling` states, as `SamplingParams` fields."""

    def __init__(self, config, sampling: dict | None = None) -> None:
        self.sampling = SamplingParams(**(sampling or {}))
        tokenizer = _load_tokenizer(config.model, config.trust_remote_code)
        self.fillers = filler_token_ids(
            tokenizer, config.eos_token_id, config.stop_token_ids
        )
        if not self.fillers:
            raise ValueError(
                f"the tokenizer of {config.model!r} has no id that is not a stop "
                "id and decodes on its own to ASCII letters and digits, so there "
                "is no filler token to replay a batch with"
            )
        self.block_size = config.kv_cache_block_size
        self.num_blocks = config.num_kvcache_blocks
        self.stateful = (
            config.hf_config.model_type
            in InputOutputProcessor._per_req_cache_model_types()
        )

    def batch(self, line: dict, block: int = 0, slot: int = 0) -> ScheduledBatch:
        """The `ScheduledBatch` `line`'s `rows` and `produces_output` state, its
        KV blocks from `block` on and its state slots from `slot` on."""
        rows = line["rows"]
        decode = rows[0][2]
        if any(kind != decode for _, _, kind in rows):
            raise ValueError(
                f"rows {rows} mix prefill and decode rows; ATOM's runner runs a "
                "batch holding prefill tokens over its prefill rows alone"
            )
        if decode and any(query != 1 for query, _, _ in rows):
            raise ValueError(
                f"rows {rows} decode more than one token in a row, a speculative "
                "step, which this rule does not build"
            )
        final = bool(line["produces_output"])
        seqs = {}
        for i, (query, context, _) in enumerate(rows):
            filler = self.fillers[i % len(self.fillers)]
            seq = Sequence(
                [filler] * context,
                self.block_size,
                self.sampling,
                has_per_req_cache=self.stateful,
            )
            seq.type = SequenceType.DECODE if decode else SequenceType.PREFILL
            seq.num_cached_tokens = context - query
            seq.block_table = new_block_table(range(block, block + seq.num_blocks))
            block += seq.num_blocks
            if self.stateful:
                seq.state_slots = [slot + i]
            seqs[seq.id] = seq
        if block > self.num_blocks:
            raise ValueError(
                f"rows {rows} hold {block} KV blocks of {self.block_size} tokens, "
                f"and the pool has {self.num_blocks}"
            )
        queries = [query for query, _, _ in rows]
        tokens, n = sum(queries), len(rows)
        batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=queries,
            total_tokens_num=tokens,
            total_tokens_num_prefill=0 if decode else tokens,
            total_tokens_num_decode=tokens if decode else 0,
            total_seqs_num=n,
            total_seqs_num_prefill=0 if decode else n,
            total_seqs_num_decode=n if decode else 0,
            is_final_chunk=None if decode else [final] * n,
        )
        if batch.produces_output() != final:
            raise ValueError(
                f"rows {rows} decode, so the step produces output, and the line "
                "says it does not"
            )
        batch.replay_rule = {"name": NAME, "version": VERSION}
        return batch
