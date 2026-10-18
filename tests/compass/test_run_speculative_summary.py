# SPDX-License-Identifier: MIT
"""The run summary names a speculative engine's acceptance input, its method,
the rates it resolved to, and the accepted-draft histogram ATOM's scheduler
counted, with their mean acceptance length.

Requests are served to their end by ATOM's scheduler over the declared
acceptance's draws, the engine hands in through `engine_done`, and the
frontend's `frontend_done` writes ``summary.json``.
"""

import json
import sys
from types import SimpleNamespace

import pytest
import test_speculative_step
from test_dp_step_max import prefill, runner
from test_run_refusal_summary import STAGE1, summary
from test_runner_control_refusals import Worker

from atom.compass import run as compass_run
from atom.compass.backends import StepStubBackend
from atom.config import SpeculativeConfig
from atom.model_engine.engine_stats import EngineStats
from atom.model_engine.scheduler import Scheduler
from atom.utils import clock
from atom.utils.clock import LPRuntime

RATE = "--spec-decode-acceptance-rate"
LENGTH = "--spec-decode-acceptance-length"
RATES = "--spec-decode-acceptance-rates"


def _stated(monkeypatch, k, **knobs):
    """A speculative config of width `k` with `knobs` resolved by ATOM's own
    `SpeculativeConfig._resolve_synthetic_acceptance`; a scalar expands to its
    mean rate at every position, standing in for the kernel's module."""
    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.rejection_sampler",
        SimpleNamespace(acceptance_length_to_rates=lambda al, n: [(al - 1) / n] * n),
    )
    spec = SpeculativeConfig.__new__(SpeculativeConfig)
    unset = ("length", "rate", "rates")
    vars(spec).update({f"synthetic_acceptance_{u}": None for u in unset})
    vars(spec).update(method="mtp", num_speculative_tokens=k, **knobs)
    spec._resolve_synthetic_acceptance()
    return spec


def _served(monkeypatch, spec):
    """ATOM's scheduler after serving `test_speculative_step`'s requests over
    `spec`'s resolved rates, with `spec` as its speculative config."""
    made = []
    monkeypatch.setattr(
        test_speculative_step,
        "Scheduler",
        lambda config: made.append(Scheduler(config)) or made[-1],
    )
    test_speculative_step._serve(spec.synthetic_acceptance_rates)
    made[0].config.speculative_config = spec
    return made[0]


def _run(monkeypatch, tmp_path):
    run = {
        "admission_path": "serving",
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": "inproc:unused",
        "out_dir": str(tmp_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    return run


@pytest.mark.parametrize(
    "knobs, flag, value",
    [
        ({"synthetic_acceptance_rates": [0.9, 0.7, 0.4]}, RATES, [0.9, 0.7, 0.4]),
        ({"synthetic_acceptance_length": 2.5}, LENGTH, 2.5),
        ({"synthetic_acceptance_rate": 0.5}, RATE, 0.5),
    ],
)
def test_a_speculative_run_names_its_input_its_method_and_its_histogram(
    monkeypatch, tmp_path, knobs, flag, value
):
    spec = _stated(monkeypatch, 3, **knobs)
    scheduler = _served(monkeypatch, spec)

    out = summary(
        monkeypatch, tmp_path, StepStubBackend(STAGE1), 8, scheduler=scheduler
    )

    stats = scheduler.engine_stats
    said = out["speculative"]["engine"]
    print("speculative:", json.dumps(said))
    assert said == {
        "drafter": "mtp",
        "draft_tokens": 3,
        "input": {"flag": flag, "value": value, "method": "declared"},
        "rates": spec.synthetic_acceptance_rates,
        "distribution": {str(a): n for a, n in stats.distribution.items()},
        "mean_acceptance_length": pytest.approx(
            1 + stats.total_accepted / stats.total_steps
        ),
    }
    assert out["schema_version"] == compass_run.SUMMARY_SCHEMA_VERSION


def test_a_run_with_no_drafter_names_no_acceptance(monkeypatch, tmp_path):
    scheduler = SimpleNamespace(engine_stats=EngineStats())
    assert "speculative" not in summary(
        monkeypatch, tmp_path, StepStubBackend(STAGE1), 8, scheduler=scheduler
    )


def test_a_client_run_counts_only_the_drafts_its_own_steps_accepted(
    monkeypatch, tmp_path
):
    """Client run 1's histogram is the engine's at its end less that at client
    run 0's, as ATOM's `EngineStats` counts them."""
    run = _run(monkeypatch, tmp_path)
    a = compass_run._RecordingAuthority(run)  # it clears the out_dir as it starts
    stats = EngineStats(use_spec=True, mtp_k=1)
    spec = _stated(monkeypatch, 1, synthetic_acceptance_rates=[0.5])
    scheduler = SimpleNamespace(config=SimpleNamespace(speculative_config=spec))
    scheduler.engine_stats = stats
    r = runner(StepStubBackend(STAGE1))
    r.forward(prefill(8))
    engine = LPRuntime(compass_run.ENGINE, compass_run.channel_table(run), None)
    monkeypatch.setattr(clock, "_installed", engine)
    core = SimpleNamespace(runner_mgr=Worker(r), scheduler=scheduler)
    for tokens in (2, 2, 1):
        stats.update_spec(tokens)
    compass_run._engine_hand_in(run, core, {}, client_run=0)
    for tokens in (1, 1, 1, 2):
        stats.update_spec(tokens)
    compass_run._engine_hand_in(run, core, {}, client_run=1)

    def counted(k):
        return compass_run._read_hand_ins(
            run, tmp_path / compass_run.CLIENT_DIR.format(k), False
        )

    out = compass_run._summary(run, a, a, 0.0, counted(1), counted(0))

    said = out["speculative"]["engine"]
    assert said["distribution"] == {"0": 3, "1": 1}
    assert said["mean_acceptance_length"] == pytest.approx(1.25)
