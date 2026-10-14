# SPDX-License-Identifier: MIT
"""The served tokenizer on a simulated run: ``encode`` and ``decode`` cost LP time.

`compass_run.tokenizer` is what the API server's ``main`` calls on the tokenizer
it serves. Here a run file names the machine spec of `test_memory_readings`,
whose one tokenizer entry applies to ``Qwen3ForCausalLM`` on the fast backend,
and the calls are made on the clock owner's thread, as the event loop makes the
final decode of a non-streaming completion.
"""

import ast
import copy
import hashlib
import json
import pathlib
import warnings
from types import SimpleNamespace

import pytest
from test_memory_readings import DOCUMENT, TOKENIZER

from atom.compass import run as compass_run
from atom.compass.clock import LpId, single_engine_table
from atom.compass.spec import FingerprintMismatch, SpecRefusal
from atom.utils import clock
from atom.utils.clock import LPRuntime

REPO = pathlib.Path(__file__).resolve().parents[2]
PROMPT_TOKENS, OUTPUT_TOKENS = 1000, 200


class _Tokenizer:
    is_fast = True

    def encode(self, text):
        return list(range(len(text)))

    def decode(self, ids):
        return "x" * len(ids)


class _GrantAll:
    def send(self, msg):
        self.t = msg[1]

    def recv(self):
        return self.t, {}


def _config(architecture="Qwen3ForCausalLM"):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]))


@pytest.fixture
def run_file(monkeypatch, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"bound_s": 600.0, "machine": DOCUMENT}))
    monkeypatch.setenv(compass_run.ENV, str(path))


@pytest.fixture
def rt():
    rt = LPRuntime(
        LpId("frontend"),
        single_engine_table(admission_path="serving", ipc_s=0.001, stream_s=0.002),
        _GrantAll(),
    )
    clock.install(rt)
    rt.start_run()
    yield rt
    clock.install(None)


def test_on_a_run_encode_and_decode_advance_the_lp_clock(run_file, rt):
    tok = _Tokenizer()
    compass_run.tokenizer(tok, _config())
    ids = tok.encode("p" * PROMPT_TOKENS)
    encoded = rt.now
    assert tok.decode(ids[:OUTPUT_TOKENS]) == "x" * OUTPUT_TOKENS
    decoded = rt.now - encoded
    print(f"\none request: encode {encoded!r} s, decode {decoded!r} s of LP time")
    derate = TOKENIZER["derate"]
    assert encoded == pytest.approx(
        TOKENIZER["encode_fixed_s"]
        + PROMPT_TOKENS / (TOKENIZER["encode_tokens_per_s"] * derate)
    )
    assert decoded == pytest.approx(
        TOKENIZER["decode_fixed_s"]
        + OUTPUT_TOKENS / (TOKENIZER["decode_tokens_per_s"] * derate)
    )


def test_with_no_run_file_the_tokenizer_is_left_as_it_is(monkeypatch):
    monkeypatch.delenv(compass_run.ENV, raising=False)
    tok = _Tokenizer()
    compass_run.tokenizer(tok, _config())
    assert vars(tok) == {}


def test_an_unmeasured_architecture_is_refused(run_file):
    with pytest.raises(SpecRefusal, match="no tokenizer probed for 'LlamaForCausalLM'"):
        compass_run.tokenizer(_Tokenizer(), _config("LlamaForCausalLM"))


class _LoadedFrom(_Tokenizer):
    """A fast tokenizer loaded from a model directory, as ``from_pretrained``
    records it."""

    def __init__(self, directory):
        self.name_or_path = str(directory)


def _model_dir(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text('{"version": "1.0"}')
    return model, "sha256:" + hashlib.sha256(b'{"version": "1.0"}').hexdigest()


def test_a_tokenizer_json_other_than_the_measured_one_warns(run_file, tmp_path):
    model, loaded = _model_dir(tmp_path)
    with pytest.warns(FingerprintMismatch) as caught:
        compass_run.tokenizer(_LoadedFrom(model), _config())
    message = str(caught[0].message)
    print(f"\n{message}")
    assert TOKENIZER["fingerprint"] in message and loaded in message


def test_the_measured_tokenizer_json_is_silent(monkeypatch, tmp_path):
    model, loaded = _model_dir(tmp_path)
    machine = copy.deepcopy(DOCUMENT)
    machine["host"]["tokenizers"] = [dict(TOKENIZER, fingerprint=loaded)]
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"bound_s": 600.0, "machine": machine}))
    monkeypatch.setenv(compass_run.ENV, str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compass_run.tokenizer(_LoadedFrom(model), _config())


def test_the_server_wraps_its_tokenizer_after_the_delta_reuse_probe():
    """The probe decodes at startup, before the run has traffic; a charged
    decode there waits on a clock that no request will advance."""
    tree = ast.parse((REPO / "atom/entrypoints/openai/api_server.py").read_text())
    main = next(
        f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == "main"
    )
    body = [ast.unparse(s) for s in main.body]

    def first(text):
        return next(i for i, s in enumerate(body) if text in s)

    assert first("enable_delta_reuse(") < first("compass_run.tokenizer(")
