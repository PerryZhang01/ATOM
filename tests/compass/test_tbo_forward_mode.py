# SPDX-License-Identifier: MIT
"""A simulated step settles its TBO forward mode as a real step does.

Two ranks of a gloo group, through `test_dp_step_max`'s `dp2`. The real side is
ATOM's own `ModelRunner.prepare_model`, compiled from its source with the
helpers it calls before `ForwardMode.decide` and stopped once `decide` returns;
the simulated side is `projection.forward_mode` on `CompassModelRunner` over
that same class. Importing the real runner needs a driver, which is why it is
compiled. Each side's `all_gather` payload is recorded per rank.
"""

import ast
import logging
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import MockConfig
from test_dp_step_max import dp2, prefill  # noqa: F401 -- dp2 is a fixture
from test_runner_non_allocating import ATOM_RUNNER, PACKAGE, _classes

from atom.compass.runner import projection
from atom.compass.runner.overrides import NonAllocatingRunner
from atom.model_engine.scheduler import ScheduledBatch
from atom.utils.forward_context import ForwardMode
from atom.utils.tbo.ubatching import local_tbo_precompute

#: The `ModelRunner` methods `prepare_model` runs up to and including `decide`.
REAL = {
    "prepare_model",
    "_local_tbo_eligibility",
    "_dspark_apply_q_bucket",
    "_piecewise_cg_active",
    "_producer_graph_shapes",
}


class Settled(Exception):
    """`decide` returned; the rest of `prepare_model` needs a device."""


class StopAtDecide(ForwardMode):
    @classmethod
    def decide(cls, **kwargs):
        raise Settled(ForwardMode.decide(**kwargs))


def _runner_classes():
    atom_runner = _classes(ATOM_RUNNER)["ModelRunner"]
    atom_runner.body = [
        n for n in atom_runner.body if isinstance(n, ast.FunctionDef) and n.name in REAL
    ]
    namespace = {
        "np": np,
        "logger": logging.getLogger("atom"),
        "ScheduledBatch": ScheduledBatch,
        "ForwardMode": StopAtDecide,
        "local_tbo_precompute": local_tbo_precompute,
        "get_dp_group": lambda: sys.modules["aiter.dist.parallel_state"].get_dp_group(),
        "_profile_runner_stage": lambda method: method,
        "NonAllocatingRunner": NonAllocatingRunner,
    }
    compass_runner = _classes(PACKAGE / "model_runner.py")["CompassModelRunner"]
    exec(ast.unparse(atom_runner), namespace)  # noqa: S102
    exec(ast.unparse(compass_runner), namespace)  # noqa: S102
    return namespace["ModelRunner"], namespace["CompassModelRunner"]


ModelRunner, CompassModelRunner = _runner_classes()


def runner(rank):
    r = object.__new__(CompassModelRunner)
    r.config = MockConfig(
        enable_tbo=True,
        enable_tbo_decode=False,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=2, data_parallel_rank=rank),
    )
    r.capture_sizes_np, r.enforce_eager = np.array([0], dtype=np.int32), True
    r.attn_metadata_builder = SimpleNamespace(step_needs_eager=lambda batch: False)
    r.tokenID_processor = SimpleNamespace(is_pd_producer=False)
    return r


def real(r, batch):
    with pytest.raises(Settled) as settled:
        ModelRunner.prepare_model(r, batch)
    return settled.value.args[0]


def settled(mode):
    """The mode's fields, with the token counts' tensor as a list."""
    sync = {**vars(mode.sync)}
    sync["num_tokens_across_dp"] = sync["num_tokens_across_dp"].tolist()
    return {**vars(mode), "sync": sync}


@pytest.mark.parametrize(
    "tokens, split",
    [
        # Rank 0 clears the bar and rank 1 is split with it.
        ((40, 8), ((20, 4), (20, 4))),
        # One token cannot be split, and that rank vetoes the group.
        ((40, 1), None),
    ],
)
def test_a_dp2_step_agrees_the_mode_and_payload_a_real_step_agrees(
    dp2, monkeypatch, tokens, split  # noqa: F811
):
    monkeypatch.setenv("ATOM_TBO_PREFILL_MIN_TOKENS", "32")
    log = threading.local()
    gather = torch.distributed.all_gather

    def recording(out, local, group=None):
        gather(out, local, group=group)
        log.payloads.append(torch.stack(out).tolist())

    monkeypatch.setattr(torch.distributed, "all_gather", recording)

    def on_rank(rank):
        def step():
            r, batch = runner(rank), prefill(tokens[rank])
            log.payloads = []
            real_mode = real(r, batch)
            simulated = projection.forward_mode(batch, r)
            view = projection.batch_view(batch, simulated, r)
            return real_mode, simulated, view, log.payloads

        return step

    for real_mode, simulated, view, (real_payload, payload) in dp2(
        on_rank(0), on_rank(1)
    ):
        assert len(payload[0]) == 8
        assert payload == real_payload
        assert settled(simulated) == settled(real_mode)
        assert simulated.tbo_collective_active is (split is not None)
        assert view.tbo_tokens == split
