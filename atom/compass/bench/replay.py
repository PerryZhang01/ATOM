# SPDX-License-Identifier: MIT
"""A `ScheduledBatch` rebuilt from a step-record line, by a fixed, versioned rule.

A step-record line states a batch as its rows, in order, each with its query
tokens, its context tokens and whether it decodes, and whether the step
produces output. ATOM's forward also needs block tables, tokens and, for a
model that keeps per-request state, state slots. `ReplayRule.batch` supplies
them the same way for every line:

- Rows keep the line's order. Prefill rows must come first, as ATOM's runner
  reads the first `total_seqs_num_prefill` rows as prefill.
- Each row is a new request, numbered by `Sequence`'s own counter, so no two
  replays share a request.
- Row i holds `ceil(context / block_size)` KV blocks, contiguous, from a fresh
  pool: row 0 starts at block 0 and each row starts where the one before ended.
- Every token of row i is the filler id `DeferredTokenStream` reports for a
  request numbered i, out of the ids `filler_token_ids` keeps for the model's
  tokenizer, as the simulated runner keeps them.
- Under a model that keeps per-request state, row i holds state slot i of a
  fresh pool.
- A line that produces output has every prefill row on its final chunk, one
  that does not has none: a line does not say which of its chunks ends a
  prompt, and ATOM's runner samples every row of a step that produces output.
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
    SequenceStatus,
    SequenceType,
    new_block_table,
)

NAME = "fresh-contiguous"
VERSION = 1


class ReplayRule:
    """The rule for one engine configuration: its tokenizer, block size, KV
    pool, and whether its model keeps per-request state."""

    def __init__(self, config) -> None:
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

    def batch(self, line: dict) -> ScheduledBatch:
        """The `ScheduledBatch` `line`'s `rows` and `produces_output` state."""
        rows = line["rows"]
        kinds = [decode for _, _, decode in rows]
        if kinds != sorted(kinds):
            raise ValueError(
                f"rows {rows} put a decode row before a prefill row, and ATOM's "
                "runner reads a batch's first rows as its prefill rows"
            )
        if any(decode and query != 1 for query, _, decode in rows):
            raise ValueError(
                f"rows {rows} decode more than one token in a row, a speculative "
                "step, which this rule does not build"
            )
        final = bool(line["produces_output"])
        seqs, block = {}, 0
        for i, (query, context, decode) in enumerate(rows):
            filler = self.fillers[i % len(self.fillers)]
            seq = Sequence(
                [filler] * context, self.block_size, has_per_req_cache=self.stateful
            )
            seq.status = SequenceStatus.RUNNING
            seq.type = SequenceType.DECODE if decode else SequenceType.PREFILL
            seq.num_cached_tokens = context - query
            seq.block_table = new_block_table(range(block, block + seq.num_blocks))
            block += seq.num_blocks
            if self.stateful:
                seq.state_slots = [i]
            seqs[seq.id] = seq
        if block > self.num_blocks:
            raise ValueError(
                f"rows {rows} hold {block} KV blocks of {self.block_size} tokens, "
                f"and the pool has {self.num_blocks}"
            )
        queries = [query for query, _, _ in rows]
        prefill = kinds.count(False)
        prefill_tokens = sum(queries[:prefill])
        batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=queries,
            total_tokens_num=sum(queries),
            total_tokens_num_prefill=prefill_tokens,
            total_tokens_num_decode=sum(queries) - prefill_tokens,
            total_seqs_num=len(rows),
            total_seqs_num_prefill=prefill,
            total_seqs_num_decode=len(rows) - prefill,
            is_final_chunk=[final] * len(rows) if prefill else None,
        )
        if batch.produces_output() != final:
            raise ValueError(
                f"rows {rows} decode, so the step produces output, and the line "
                "says it does not"
            )
        batch.replay_rule = {"name": NAME, "version": VERSION}
        return batch
