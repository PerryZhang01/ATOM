# SPDX-License-Identifier: MIT
"""The weights and buffers a simulated run installs are read off its meta build.

`NonAllocatingRunner._build_and_load_model` builds the model class on the meta
device and hands the tree to `run.runner`; the weights term in the readings it
installs is then the bytes of the tree's parameter storages, each once, and the
buffers term those of its buffer storages.
"""

import json
from types import SimpleNamespace

import torch
from test_memory_readings import CONFIG_JSON, MAX_NUM_BATCHED_TOKENS, with_readings
from test_vertical_slice import _run_file
from transformers import PretrainedConfig

from atom.compass import run as compass_run
from atom.compass.backends.provenance import Method
from atom.compass.runner.overrides import NonAllocatingRunner

VOCAB, HIDDEN = 64, 16
EXPERTS, ROWS, COLS = 4, 8, 32


class TiedPacked(torch.nn.Module):
    """A head tied two ways ATOM's models tie one, and a packed FP4-like tensor."""

    def __init__(self, config):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN, dtype=torch.bfloat16)
        # Qwen3's tie: a parameter of its own over the embedding's storage.
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias=False, dtype=torch.bfloat16)
        self.head.weight.data = self.embed.weight.data
        # Qwen3.5's tie: the embedding module itself under a second name.
        self.lm_head = self.embed
        # Two FP4 values a byte, as ATOM's MXFP4 experts are stored.
        self.experts = torch.nn.Parameter(
            torch.empty(EXPERTS, ROWS, COLS // 2, dtype=torch.uint8),
            requires_grad=False,
        )


class Buffered(torch.nn.Module):
    """Buffers held the ways ATOM's models hold them, and a table held as none."""

    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(VOCAB, HIDDEN, dtype=torch.bfloat16), requires_grad=False
        )
        # One rotary module every layer shares, as `get_rope` caches it.
        rope = torch.nn.Module()
        rope.register_buffer("cos_sin", torch.empty(ROWS, COLS, dtype=torch.bfloat16))
        self.layers = torch.nn.ModuleList(torch.nn.Module() for _ in range(2))
        for layer in self.layers:
            layer.rope = rope
        # A warmup placeholder, as DeepSeek-V4's indexer registers its cache.
        self.register_buffer(
            "kv_cache", torch.zeros(1, ROWS, HIDDEN, dtype=torch.float32)
        )
        # A buffer over a parameter's storage is resident once, as the parameter.
        self.register_buffer("row", self.weight.data[0])
        # DeepSeek-V4's rotary table: a plain attribute, so the load residue
        # holds it and the buffers do not.
        self.cos_cache = torch.empty(ROWS, COLS)


def _built(monkeypatch, tmp_path, model_class=TiedPacked):
    """A worker's runner after `_build_and_load_model` under a run file."""
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path)))
    (tmp_path / "config.json").write_text(CONFIG_JSON.read_text())
    hf = PretrainedConfig.from_dict(json.loads(CONFIG_JSON.read_text())["text_config"])
    runner = SimpleNamespace(
        config=SimpleNamespace(
            model=str(tmp_path),
            hf_config=hf,
            tensor_parallel_size=1,
            parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            enforce_eager=True,
            speculative_config=None,
            kv_transfer_config={},
        ),
    )
    with_readings(runner, tmp_path / "store")
    NonAllocatingRunner._build_and_load_model(runner, model_class)
    return runner


def test_the_weights_are_the_meta_builds_storages_each_counted_once(
    monkeypatch, tmp_path
):
    runner = _built(monkeypatch, tmp_path)
    weights = runner.compass_readings.peak_torch.terms[0]
    assert weights.name == "weights"
    assert weights.method is Method.ANALYTICAL
    assert weights.nbytes == VOCAB * HIDDEN * 2 + EXPERTS * ROWS * COLS // 2
    assert all(p.is_meta for p in runner.model.parameters())


def test_the_buffers_are_the_meta_builds_buffer_storages_each_counted_once(
    monkeypatch, tmp_path
):
    runner = _built(monkeypatch, tmp_path, Buffered)
    weights, buffers = runner.compass_readings.peak_torch.terms[:2]
    assert buffers.name == "buffers"
    assert buffers.method is Method.ANALYTICAL
    assert buffers.nbytes == ROWS * COLS * 2 + ROWS * HIDDEN * 4
    assert buffers.source == "2 buffer storages of a meta build of Buffered"
    assert weights.nbytes == VOCAB * HIDDEN * 2
