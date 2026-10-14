# SPDX-License-Identifier: MIT
"""The ids a predicted step reports, read by the three things that read them.

A request's generated ids reach ATOM's scheduler, which ends a request on a
stop id; the prefix cache, which hashes generated blocks; and the stream
detokenizer, which moves its window only past complete characters. Each run
here drives ATOM's own `Scheduler`, `BlockManager` and
`IncrementalStreamDetokenizer` against a byte-level BPE tokenizer that
`_load_tokenizer` reads from a model directory, the way the frontend and the
runner both load it.
"""

import itertools
import json

import pytest
from conftest import MockConfig
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.runner import overrides
from atom.compass.runner.overrides import (
    NonAllocatingRunner,
    RunnerRefusal,
    install_cost_backend,
)
from atom.compass.runner.step_output import DeferredTokenStream, filler_token_ids
from atom.entrypoints.openai.streaming_dispatch import IncrementalStreamDetokenizer
from atom.model_engine.llm_engine import _load_tokenizer
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

EOS, STOP = "<|endoftext|>", "<|im_end|>"
BLOCK = 16
# Two prompt blocks exactly, so every generated block holds generated ids only.
PROMPT = list(range(40, 40 + 2 * BLOCK))
CORPUS = [
    "the quick brown fox jumps over the lazy dog 12345",
    "naïve café déjà vu, über straße",
    "x = f(y); print(x)\n",
] * 20


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A byte-level BPE tokenizer, saved where a model's tokenizer is read from."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        CORPUS,
        trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=[EOS, STOP],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    directory = tmp_path_factory.mktemp("model")
    tok.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": EOS,
                "additional_special_tokens": [STOP],
            }
        )
    )
    return directory


@pytest.fixture(scope="module")
def tokenizer(model_dir):
    return _load_tokenizer(str(model_dir))


@pytest.fixture(scope="module", autouse=True)
def _unpriced():
    """Reporting is under test here, not the price."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
        yield


def _stops(tokenizer):
    """The end-of-turn id, and the ids requests 0 and 1 get when nothing is a stop."""
    unexcluded = filler_token_ids(tokenizer, tokenizer.eos_token_id, [])[:2]
    return [tokenizer.convert_tokens_to_ids(STOP), *unexcluded]


class _Runner(NonAllocatingRunner):
    def __init__(self, config, stream=None):
        self.config = config
        install_cost_backend(self, StepStubBackend())
        if stream is not None:
            self._token_stream = stream


def _drive(model_dir, tokenizer, stream=None, requests=2, max_tokens=3 * BLOCK):
    """Identical prompts through ATOM's scheduler with its stop checks live.

    Returns the sequences and, per request, the hashes the block manager gave
    its generated blocks.
    """
    config = MockConfig(
        model=str(model_dir),
        trust_remote_code=False,
        eos_token_id=tokenizer.eos_token_id,
        stop_token_ids=_stops(tokenizer),
        enable_prefix_caching=True,
        max_num_seqs=8,
        num_kvcache_blocks=256,
        kv_cache_block_size=BLOCK,
        max_model_len=1024,
        max_num_batched_tokens=256,
        pipeline_parallel_size=1,
    )
    scheduler, runner = Scheduler(config), _Runner(config, stream)
    Sequence.counter = itertools.count()
    sequences = [
        Sequence(
            list(PROMPT),
            BLOCK,
            sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=False),
        )
        for _ in range(requests)
    ]
    for sequence in sequences:
        scheduler.add(sequence)
    pool = scheduler.block_manager.kv
    generated = {s.id: set() for s in sequences}
    for _ in range(500):
        if not scheduler.has_unfinished_requests():
            break
        scheduled = scheduler.schedule()
        if scheduled is None or scheduled[0] is None or not scheduled[0].req_ids:
            continue
        batch, seqs = scheduled
        reply = runner.forward(batch)
        scheduler.postprocess(list(seqs.values()), reply, batch=batch)
        for seq in seqs.values():
            for block_id in seq.block_table[len(PROMPT) // BLOCK :]:
                if pool.block(block_id).hash != -1:
                    generated[seq.id].add(pool.block(block_id).hash)
    else:
        raise AssertionError("the driven run does not terminate")
    return sequences, generated


def test_identical_prompts_share_no_generated_block(model_dir, tokenizer):
    sequences, generated = _drive(model_dir, tokenizer)
    assert [s.leave_reason for s in sequences] == ["max_tokens"] * 2
    first, second = generated.values()
    assert len(first) == len(second) == 2
    assert not first & second


def test_one_id_for_every_request_makes_their_generated_blocks_collide(
    model_dir, tokenizer
):
    """The control: the same drive with a single filler id shares every block."""
    one = filler_token_ids(tokenizer, tokenizer.eos_token_id, _stops(tokenizer))[:1]
    _, generated = _drive(model_dir, tokenizer, stream=DeferredTokenStream(one))
    first, second = generated.values()
    assert len(first) == 2 and first == second


def test_the_stream_window_moves_past_every_generated_token(model_dir, tokenizer):
    sequences, _ = _drive(model_dir, tokenizer)
    for seq in sequences:
        stream = IncrementalStreamDetokenizer(tokenizer)
        for token_id in seq.completion_token_ids:
            assert stream.update([token_id], finished=False)
            assert stream.read_offset == len(stream.tokens)


def test_a_partial_character_holds_the_window_and_is_never_vetted(tokenizer):
    """The control for the window: a lone byte of a multi-byte character stalls it."""
    partial = next(
        i for i in range(len(tokenizer)) if tokenizer.decode([i]) == "\ufffd"
    )
    stream = IncrementalStreamDetokenizer(tokenizer)
    for _ in range(3):
        assert stream.update([partial], finished=False) == ""
    assert stream.read_offset == 0
    assert partial not in filler_token_ids(tokenizer, tokenizer.eos_token_id, [])


def test_every_filler_is_ascii_letters_or_digits_and_no_stop_id(tokenizer):
    stops = _stops(tokenizer)
    fillers = filler_token_ids(tokenizer, tokenizer.eos_token_id, stops)
    assert fillers
    assert not {tokenizer.eos_token_id, *stops} & set(fillers)
    assert not set(tokenizer.all_special_ids) & set(fillers)
    texts = [tokenizer.decode([i]) for i in fillers]
    assert all(t.isascii() and t.isalnum() for t in texts)
    # The vocabulary holds the ids this excludes: whole non-ASCII letters,
    # and ASCII whitespace and punctuation.
    vocab = [tokenizer.decode([i]) for i in range(len(tokenizer))]
    assert {"é", " ", ";", "\n"} <= set(vocab)


def test_consecutive_requests_are_reported_with_different_ids():
    class _Batch:
        req_ids, total_seqs_num = [7, 8, 9, 10], 4

    reply = DeferredTokenStream([11, 12, 13], deferred=False)._reply(_Batch, False)
    assert reply["token_ids"] == [(12,), (13,), (11,), (12,)]


def test_a_tokenizer_with_no_usable_id_is_refused(model_dir, tokenizer, monkeypatch):
    monkeypatch.setattr(overrides, "filler_token_ids", lambda *_: [])
    with pytest.raises(RunnerRefusal, match="no id"):
        _drive(model_dir, tokenizer)
