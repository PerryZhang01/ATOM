# SPDX-License-Identifier: MIT
"""A pipeline stage is priced for its own span of the stack.

`run.runner` builds each stage's `step_stub` backend from ATOM's
`get_pp_indices` span on the Qwen3.5-27B config, a hybrid whose paged layers
are one in four, so a stage's paged share is not its layer share. A stage's
`forward` then prices through `_group_step_seconds` instead of being refused.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from test_dp_step_max import prefill, runner
from test_memory_readings import CONFIG_JSON, META_8B
from test_vertical_slice import _run_file
from transformers import PretrainedConfig

from atom.compass import run as compass_run
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.runner.overrides import RunnerRefusal

STEP = BatchView(
    (
        RequestShape(512, 2048, False),
        RequestShape(1, 4000, True),
        RequestShape(1, 9000, True),
    )
)


def stage_backend(monkeypatch, tmp_path, pp, pp_rank):
    """The backend `run.runner` installs on stage `pp_rank` of `pp`."""
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path)))
    (tmp_path / "config.json").write_text(CONFIG_JSON.read_text())
    hf = PretrainedConfig.from_dict(json.loads(CONFIG_JSON.read_text())["text_config"])
    r = SimpleNamespace(
        config=SimpleNamespace(
            model=str(tmp_path),
            hf_config=hf,
            tensor_parallel_size=1,
            pipeline_parallel_size=pp,
            enable_dp_attention=False,
            enable_expert_parallel=False,
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_rank=0,
                pipeline_parallel_rank=pp_rank,
            ),
            max_num_batched_tokens=8192,
            max_num_seqs=256,
            gpu_memory_utilization=0.9,
            torch_dtype=torch.bfloat16,
            enforce_eager=True,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1]),
            capture_sizes=[1],
            kv_transfer_config={},
        ),
        _piecewise_cg_active=lambda: False,
    )
    compass_run.runner(r, META_8B)
    return r.compass_backend


def terms(backend):
    return {t.name: t.seconds for t in backend.price(STEP).terms}


def test_three_stages_sum_to_the_whole_model_and_context_follows_paged_layers(
    monkeypatch, tmp_path
):
    whole = terms(stage_backend(monkeypatch, tmp_path, 1, 0))
    stages = [stage_backend(monkeypatch, tmp_path, 3, k) for k in range(3)]

    # get_pp_indices(64, k, 3): spans of 21, 22 and 21 layers, holding 5, 5
    # and 6 of the 16 full-attention layers.
    assert [b.stack_layers for b in stages] == [21, 22, 21]
    assert [b.stage_share for b in stages] == [
        (21 / 64, 5 / 16),
        (22 / 64, 5 / 16),
        (21 / 64, 6 / 16),
    ]
    priced = [terms(b) for b in stages]
    for name, seconds in whole.items():
        assert sum(p[name] for p in priced) == pytest.approx(seconds), name
    last = priced[2]
    assert last["decode.context"] == pytest.approx(whole["decode.context"] * 6 / 16)
    assert last["prefill.tokens"] == pytest.approx(whole["prefill.tokens"] * 21 / 64)
    assert "stage share" in stages[2].describe()


def test_a_stage_backend_prices_its_forward_at_pipeline_width_two():
    whole = runner(StepStubBackend()).forward(prefill(8)).predicted_s
    stage = runner(StepStubBackend(stage_share=(0.5, 0.5)))
    stage.config.pipeline_parallel_size = 2

    assert stage.forward(prefill(8)).predicted_s == pytest.approx(whole / 2)


def test_a_backend_with_no_stage_share_is_still_refused_naming_it():
    stage = runner(StepStubBackend())
    stage.config.pipeline_parallel_size = 2
    with pytest.raises(RunnerRefusal, match="StepStubBackend prices a whole step"):
        stage.forward(prefill(8))


def test_a_share_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match="stage_share"):
        StepStubBackend(stage_share=(1.5, 0.5))
