# SPDX-License-Identifier: MIT
"""A data-parallel group's step costs the max over its ranks' own step costs.

Each rank's `forward` prices its own batch, runs ATOM's `ForwardMode.decide`,
whose `sync_dp_metadata` is the group's real collective, and exchanges one
`all_reduce(MAX)` of the seconds on the same group. Two ranks run as two
threads, each on its own gloo group, with `aiter.dist.parallel_state` stood in
to hand each thread its own. The runner is `CompassModelRunner` compiled over
an ATOM `ModelRunner` holding only its own `dummy_execution`, because importing
either runner needs a driver.
"""

import ast
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import MockConfig
from test_runner_non_allocating import ATOM_RUNNER, PACKAGE, _classes

import atom.utils.distributed.utils as dist_utils
from atom.compass.backends.shape import (
    BatchView,
    Coefficients,
    RequestShape,
    ShapeStubBackend,
)
from atom.compass.runner import projection
from atom.compass.runner.overrides import (
    NonAllocatingRunner,
    RunnerRefusal,
    install_cost_backend,
)
from atom.compass.runner.step_output import DeferredTokenStream
from atom.model_engine.scheduler import ScheduledBatch, Scheduler
from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
    SequenceType,
    new_block_table,
)
from atom.sampling_params import SamplingParams
from atom.utils import get_open_port

BLOCK = 16


def _compass_runner_class():
    atom_runner = _classes(ATOM_RUNNER)["ModelRunner"]
    atom_runner.body = [
        n
        for n in atom_runner.body
        if isinstance(n, ast.FunctionDef) and n.name == "dummy_execution"
    ]
    namespace = {
        "np": np,
        "logger": logging.getLogger("atom"),
        "Sequence": Sequence,
        "SequenceStatus": SequenceStatus,
        "SequenceType": SequenceType,
        "ScheduledBatch": ScheduledBatch,
        "new_block_table": new_block_table,
        "NonAllocatingRunner": NonAllocatingRunner,
    }
    compass_runner = _classes(PACKAGE / "model_runner.py")["CompassModelRunner"]
    exec(ast.unparse(atom_runner), namespace)  # noqa: S102
    exec(ast.unparse(compass_runner), namespace)  # noqa: S102
    return namespace["CompassModelRunner"]


CompassModelRunner = _compass_runner_class()


class Recording(ShapeStubBackend):
    """The stub, keeping each view it priced."""

    def __init__(self, coefficients=None):
        super().__init__(coefficients)
        self.views = []

    def estimate(self, batch_view):
        self.views.append(batch_view)
        return super().estimate(batch_view)


def runner(backend, dp_size=1, dp_rank=0):
    r = object.__new__(CompassModelRunner)
    r.config = MockConfig(
        pipeline_parallel_size=1,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size, data_parallel_rank=dp_rank
        ),
    )
    r.rank, r.label, r.block_size = 0, f"dp{dp_rank}", BLOCK
    r.capture_sizes_np, r.enforce_eager = np.array([0], dtype=np.int32), True
    r._token_stream = DeferredTokenStream([0])
    if backend is not None:
        install_cost_backend(r, backend)
    return r


def prefill(tokens):
    """The first batch ATOM's scheduler builds for one prompt of `tokens`."""
    scheduler = Scheduler(
        MockConfig(
            num_kvcache_blocks=64,
            kv_cache_block_size=BLOCK,
            max_model_len=256,
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
    )
    scheduler.add(
        Sequence(list(range(5, 5 + tokens)), BLOCK, sampling_params=SamplingParams())
    )
    return scheduler.schedule()[0]


def priced(*rows, coefficients=None):
    return ShapeStubBackend(coefficients).estimate(BatchView(rows)).seconds


@pytest.fixture
def dp2(monkeypatch):
    """Run one callable per rank of a two-rank gloo group; return their results."""
    monkeypatch.setattr(
        dist_utils, "_get_default_timeout", lambda _: timedelta(seconds=5)
    )
    # Gloo binds the address the hostname resolves to, one DNS lookup per rank;
    # a stalled lookup outlasts the timeout above. Loopback needs no lookup.
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    local = threading.local()
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(get_dp_group=lambda: SimpleNamespace(cpu_group=local.group)),
    )
    port = get_open_port()

    def on_rank(rank, step):
        local.group = dist_utils.stateless_init_torch_distributed_process_group(
            "127.0.0.1", port, rank, 2, backend="gloo"
        )
        try:
            return step()
        finally:
            dist_utils.stateless_destroy_torch_distributed_process_group(local.group)

    def run(*steps):
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(on_rank, r, s) for r, s in enumerate(steps)]
            return [f.result() for f in futures]

    return run


def test_one_rank_exchanges_nothing_and_reports_its_own_cost(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", None)

    reply = runner(ShapeStubBackend()).forward(prefill(40))

    assert reply.predicted_s == priced(RequestShape(40, 40, False))


def test_both_ranks_report_the_larger_cost_and_the_group_token_count(dp2):
    backends = [Recording(), Recording()]
    ranks = [runner(b, dp_size=2, dp_rank=r) for r, b in enumerate(backends)]
    big, small = prefill(40), prefill(8)

    replies = dp2(lambda: ranks[0].forward(big), lambda: ranks[1].forward(small))

    own = [priced(RequestShape(40, 40, False)), priced(RequestShape(8, 8, False))]
    assert own[0] > own[1]
    assert [r.predicted_s for r in replies] == [own[0], own[0]]
    # A rank prefills, so the variable-length gather carries both counts.
    assert [b.views[0].moe_rows for b in backends] == [48, 48]
    assert [b.views[0].requests for b in backends] == [
        (RequestShape(40, 40, False),),
        (RequestShape(8, 8, False),),
    ]


def test_an_idle_rank_prices_its_dummy_batch_into_the_max(dp2):
    # Decode dearer than prefill, so the dummy batch is the group's max.
    coefficients = Coefficients.constant(prefill_seconds=0.001, decode_seconds=0.005)
    backends = [Recording(coefficients), Recording(coefficients)]
    ranks = [runner(b, dp_size=2, dp_rank=r) for r, b in enumerate(backends)]

    busy, idle = dp2(lambda: ranks[0].forward(prefill(40)), ranks[1].dummy_execution)

    assert backends[1].views[0].requests == (RequestShape(1, 1, True),)
    assert [busy.predicted_s, idle.predicted_s] == [0.005, 0.005]
    assert priced(RequestShape(40, 40, False), coefficients=coefficients) == 0.001


def test_a_runner_with_no_backend_refuses_the_step():
    with pytest.raises(RunnerRefusal, match="install_cost_backend"):
        runner(None).forward(prefill(8))


def test_a_pipeline_stage_refuses_to_price_the_whole_model():
    stage = runner(ShapeStubBackend())
    stage.config.pipeline_parallel_size = 2
    with pytest.raises(RunnerRefusal, match="one stage of a pipeline"):
        stage.forward(prefill(8))


def test_a_dp_rank_whose_mode_states_no_group_count_is_refused():
    batch = prefill(8)
    single_rank_mode = projection.forward_mode(batch, runner(None))

    with pytest.raises(RunnerRefusal, match="states no num_tokens_across_dp"):
        projection.batch_view(batch, single_rank_mode, runner(None, dp_size=2))
