# SPDX-License-Identifier: MIT
"""A speculative run's acceptance, read from a paired real record: the record's
engine writes its scheduler's accepted-draft histogram as SIGTERM stops it,
`compass run --acceptance-record` hands ATOM the rates it gives, and the
summary names that input ``measured``.

Each record's scheduler is ATOM's, serving `test_speculative_step`'s requests
over stated rates as `test_run_speculative_summary` serves them.
"""

import json
import os
import signal
from types import SimpleNamespace

import pytest
from test_drive_run import flags, stub, unstarted
from test_run_speculative_summary import RATES, _served, _stated

from atom.compass import parity
from atom.compass import run as compass_run
from atom.compass.drive import run as drive_run
from atom.compass.drive import runfile, serve
from atom.model_engine.engine_stats import EngineStats


def mtp(k):
    return ["--method", "mtp", "--num-speculative-tokens", str(k)]


def recorded(monkeypatch, tmp_path, rates, rank=0):
    """The record directory DP rank `rank`'s engine records into, once it has
    served requests at `rates` and taken SIGTERM inside its loop; its
    scheduler, and the signals the handler ATOM installed before the loop took."""
    steps = tmp_path / "record" / serve.STEPS
    steps.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv(compass_run.ENV, raising=False)
    monkeypatch.setenv(parity.ENV, str(steps))
    spec = _stated(monkeypatch, len(rates), synthetic_acceptance_rates=rates)
    scheduler = _served(monkeypatch, spec)
    scheduler.config.parallel_config = SimpleNamespace(data_parallel_rank=rank)
    took = []

    def atom(signum, frame):
        took.append(signum)

    before = signal.signal(signal.SIGTERM, atom)
    try:
        with compass_run.engine_loop(SimpleNamespace(scheduler=scheduler)):
            os.kill(os.getpid(), signal.SIGTERM)
        assert signal.getsignal(signal.SIGTERM) is atom
    finally:
        signal.signal(signal.SIGTERM, before)
    return steps.parent, scheduler, took


def started(monkeypatch, tmp_path, *flag, atom=()):
    """The ATOM arguments and the run file `compass run` started its server
    with, given `flag` beside a step stub."""
    seen = {}
    monkeypatch.setattr(
        serve, "serve", lambda atom_args, out, run: seen.update(atom=atom_args, run=run)
    )
    monkeypatch.setattr(drive_run, "write_grading", lambda out, report: None)
    drive_run.main(flags(tmp_path, "--step-stub", stub(tmp_path), *flag, atom=atom))
    return seen["atom"], seen["run"]


def test_a_record_s_engine_writes_its_histogram_at_sigterm_then_atom_takes_it(
    monkeypatch, tmp_path
):
    record, scheduler, took = recorded(monkeypatch, tmp_path, [0.9, 0.6], rank=1)

    held = (record / serve.STEPS / "dp1.acceptance.json").read_text()
    distribution = scheduler.engine_stats.distribution
    assert json.loads(held) == {
        "draft_tokens": 2,
        "distribution": {str(a): n for a, n in distribution.items()},
    }
    assert sum(distribution.values()) > 0 and took == [signal.SIGTERM]


def test_a_run_given_its_record_draws_from_the_rates_its_histograms_give_as_measured(
    monkeypatch, tmp_path
):
    record, first, _ = recorded(monkeypatch, tmp_path, [0.8, 0.5], rank=0)
    _, second, _ = recorded(monkeypatch, tmp_path, [0.6, 0.3], rank=1)
    counts = [
        sum(s.engine_stats.distribution[a] for s in (first, second)) for a in (0, 1, 2)
    ]
    rows = sum(counts)

    atom_args, run = started(
        monkeypatch,
        tmp_path,
        "--acceptance-record",
        str(record),
        atom=[*mtp(2), "--data-parallel-size", "2"],
    )

    rates = runfile._engine_args(atom_args).spec_decode_acceptance_rates
    assert rates == pytest.approx([(counts[1] + counts[2]) / rows, counts[2] / rows])
    assert run[compass_run.ACCEPTANCE_RECORD] == str(record.resolve())
    spec = _stated(monkeypatch, 2, synthetic_acceptance_rates=rates)
    simulated = SimpleNamespace(
        config=SimpleNamespace(speculative_config=spec),
        engine_stats=EngineStats(use_spec=True, mtp_k=2),
    )
    said = compass_run._acceptance(simulated, run)["input"]
    print("input:", said)
    assert said == {
        "flag": RATES,
        "value": rates,
        "method": "measured",
        "record": str(record.resolve()),
    }


@pytest.mark.parametrize(
    "drafted, atom, refusal",
    [
        (None, mtp(1), "holds no accepted-draft histogram"),
        ([0.7], mtp(2), "drafted 1 tokens a step, and ATOM's arguments draft 2$"),
        ([0.7], [], "drafted 1 tokens a step, and ATOM's arguments draft 0$"),
        (
            [0.7],
            [*mtp(1), "--spec-decode-acceptance-rate", "0.5"],
            "ATOM's arguments state --spec-decode-acceptance-rate too",
        ),
        (
            [0.7],
            [*mtp(1), "--spec-decode-acceptance-length", "1.5"],
            "ATOM's arguments state --spec-decode-acceptance-length too",
        ),
        (
            [0.7],
            [*mtp(1), "--spec-decode-acceptance-rates", "0.5"],
            "ATOM's arguments state --spec-decode-acceptance-rates too",
        ),
    ],
)
def test_a_record_with_no_histogram_or_of_another_width_refuses_by_name(
    monkeypatch, tmp_path, drafted, atom, refusal
):
    if drafted is None:
        record = tmp_path / "record"
        (record / serve.STEPS).mkdir(parents=True)
    else:
        record, *_ = recorded(monkeypatch, tmp_path, drafted)
    monkeypatch.setattr(serve, "serve", unstarted)

    with pytest.raises(ValueError, match=refusal):
        drive_run.main(
            flags(
                tmp_path,
                "--step-stub",
                stub(tmp_path),
                "--acceptance-record",
                str(record),
                atom=atom,
            )
        )
    assert not (tmp_path / "out").exists()


def test_a_dp_record_missing_a_rank_s_histogram_refuses_by_name(monkeypatch, tmp_path):
    record, *_ = recorded(monkeypatch, tmp_path, [0.7], rank=1)
    monkeypatch.setattr(serve, "serve", unstarted)

    with pytest.raises(ValueError, match="holds no dp0.acceptance.json, and ATOM's"):
        drive_run.main(
            flags(
                tmp_path,
                "--step-stub",
                stub(tmp_path),
                "--acceptance-record",
                str(record),
                atom=[*mtp(1), "--data-parallel-size", "2"],
            )
        )


def test_a_histogram_write_that_raises_still_hands_sigterm_to_atom(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(parity.ENV, str(tmp_path / "gone"))
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    stats = SimpleNamespace(spec_enabled=True, distribution={0: 1})
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(config=config, engine_stats=stats)
    )
    took = []

    with pytest.raises(FileNotFoundError):
        compass_run._stop_recording(
            engine, lambda signum, frame: took.append(signum), signal.SIGTERM, None
        )
    assert took == [signal.SIGTERM]
