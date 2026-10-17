# SPDX-License-Identifier: MIT
"""A chat request's images on the simulated clock.

The frontend's media preparation is a tokenizer-station job charged the run
file's ``image_preprocess_s`` per image; a remote image URL, or an image in a
run with no price, is refused by name. A step whose scheduled tokens carry
image embeddings states how many, and the fitted backends refuse it.
"""

import base64
import io
import urllib.request
from types import SimpleNamespace

import pytest
import test_tokenizer_station as station
import torch
from conftest import MockConfig
from PIL import Image
from test_backend_step_grid import replay
from test_backend_step_linear import backend
from test_backend_step_roofline import fallback, roofline
from test_dp_step_max import BLOCK, runner

from atom.compass import run as compass_run
from atom.compass.backends import BatchView, RequestShape, StepStubBackend
from atom.compass.backends.resolver import CostRefused
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import ChatMessage
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.multimodal.media import MediaLoader
from atom.sampling_params import SamplingParams
from atom.utils import clock
from atom.utils.compass_loop import Refused

loop = station.loop
IMAGE_ID = 9


def _png() -> str:
    out = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(out, "PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


class _Processor:
    """A Qwen-style processor: one image token per image."""

    def apply_chat_template(self, conversation, **_):
        return "<|image_pad|>" * len(conversation[0]["content"][:-1]) + "hi"

    def __call__(self, text, images, return_tensors):
        ids = [IMAGE_ID] * len(images) + [5, 6]
        return {
            "input_ids": torch.tensor([ids]),
            "pixel_values": torch.zeros(len(images), 3),
            "image_grid_thw": torch.ones(len(images), 3, dtype=torch.int64),
        }


def _message(*urls):
    parts = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    parts.append({"type": "text", "text": "hi"})
    return ChatMessage(role="user", content=parts)


@pytest.fixture
def priced(monkeypatch, loop):
    """The frontend bootstrap's image pricing, on `loop`'s clock, put back after."""
    monkeypatch.setattr(MediaLoader, "load_image", MediaLoader.load_image)
    monkeypatch.setattr(clock, "_installed", loop.rt)
    config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen"]))
    monkeypatch.setattr(api_server, "_get_engine_config", lambda: config)
    monkeypatch.setattr(api_server, "_get_multimodal_processor", _Processor)
    return compass_run._price_images


def _prepare(loop, *urls):
    ex, release = station._executor(loop, width=1)
    job = loop.run_in_executor(
        None, api_server._prepare_multimodal_inputs, [_message(*urls)], {}
    )
    try:
        return loop.run_until_complete(job), loop.time(), ex
    finally:
        release()


def test_media_preparation_is_a_station_job_charged_per_image(loop, priced):
    priced(0.25)
    (ids, data), done, ex = _prepare(loop, _png(), _png())
    assert ids == [IMAGE_ID, IMAGE_ID, 5, 6]
    assert data["pixel_values"].shape[0] == 2
    assert done == pytest.approx(0.5)
    assert ex.refusals == []


@pytest.mark.parametrize(
    "price, url, why",
    [
        (0.25, "https://example.com/cat.png", "an image named by a remote URL"),
        (None, _png(), "an image with no image_preprocess_s in the run file"),
    ],
    ids=["remote", "unpriced"],
)
def test_an_image_that_cannot_be_priced_is_refused_by_name(
    loop, priced, monkeypatch, price, url, why
):
    monkeypatch.setattr(urllib.request, "urlopen", None)  # never fetched
    priced(price)
    with pytest.raises(Refused) as refused:
        _prepare(loop, url)
    assert refused.value.args[0] == ("media", why)
    assert loop.rt.refusals == [f"media:{why}"]


def test_the_bootstrap_charges_each_image_once_however_often_it_runs(loop, priced):
    priced(0.25)
    priced(0.25)
    assert _prepare(loop, _png())[1] == pytest.approx(0.25)


# --- a step that carries image tokens ---------------------------------------


def _image_prefill(data, prompt):
    scheduler = Scheduler(
        MockConfig(
            num_kvcache_blocks=64,
            kv_cache_block_size=BLOCK,
            max_model_len=256,
            max_num_batched_tokens=16,
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
    )
    seq = Sequence(prompt, BLOCK, sampling_params=SamplingParams())
    seq.multimodal_data, seq.cache_seed = data, 1
    scheduler.add(seq)
    return scheduler.schedule()[0]


PIXELS = {"pixel_values": torch.zeros(1, 3), "image_grid_thw": torch.ones(1, 3)}


@pytest.mark.parametrize(
    "data, prompt, tokens",
    [
        # Qwen: the image's embeddings sit at the image token, the prompt whole.
        (PIXELS, [5, IMAGE_ID, IMAGE_ID, IMAGE_ID, 6], 3),
        # DeepSeek-V4.1: spans; the first 16-token chunk holds 6 of 10.
        ({**PIXELS, "embedding_spans": ((10, 10),)}, list(range(30, 60)), 6),
    ],
)
def test_a_steps_image_tokens_are_refused_by_every_fitted_backend(data, prompt, tokens):
    batch = _image_prefill(data, prompt)
    r = runner(backend())
    r.config.multimodal_config = SimpleNamespace(image_token_id=IMAGE_ID)
    assert r.forward(batch).predicted_s == 0.0
    why = f"{tokens} image tokens, and no vision encoder in what this prices from"
    assert r.refused_commands() == (f"cost:step-linear: {why}",)
    view = BatchView((RequestShape(16, 16, False),), image_tokens=tokens)
    for other in (replay(), roofline(), fallback()):
        with pytest.raises(CostRefused, match=why):
            other.price(view)


def test_the_step_stub_prices_a_step_with_image_tokens_as_one_without():
    rows = (RequestShape(16, 16, False),)
    stub = StepStubBackend()
    with_images = stub.price(BatchView(rows, image_tokens=6))
    assert with_images.seconds == stub.price(BatchView(rows)).seconds
