# SPDX-License-Identifier: MIT
"""Real-clock reads in ATOM's core: each one takes the LP clock or is listed by site."""

import os
from types import SimpleNamespace

import pytest

import atom
from atom.compass.detect.clock_source import (
    CORE_ALLOW_LIST,
    CORE_ROOTS,
    ClockSourceLint,
)
from atom.entrypoints.atomesh.atom_standalone_service import SingleRequestState
from atom.entrypoints.openai.protocol import ModelCard
from atom.metrics.exporter import AtomMetricsExporter
from atom.utils import clock

REPO = os.path.dirname(os.path.dirname(atom.__file__))


def _core_reads():
    lint = ClockSourceLint()
    for root in CORE_ROOTS:
        for path in lint.modules(os.path.join(REPO, root)):
            with open(path, encoding="utf-8") as handle:
                yield from lint.scan_source(handle.read(), path)


def _seeded(module, tmp_path, seed):
    with open(os.path.join(REPO, module), encoding="utf-8") as handle:
        source = handle.read() + seed
    target = tmp_path / module
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    return target, len(source.splitlines())


class _Runtime:
    def __init__(self, t):
        self.t = t

    def read_clock(self):
        return self.t


def _no_real_read():
    raise AssertionError("a simulated run read the machine clock")


@pytest.fixture
def lp_clock():
    runtime = _Runtime(7.25)
    clock.install(runtime)
    yield runtime
    clock.install(None)


class TestTheCoreGate:
    @pytest.mark.parametrize("root", CORE_ROOTS)
    def test_no_unlisted_real_read_is_left(self, root):
        """The set of reads still waiting for substitution is empty."""
        code, report = ClockSourceLint().check(os.path.join(REPO, root))
        assert code == 0, report

    def test_the_roots_are_every_package_but_the_excluded_ones(self):
        assert "atom/kv_transfer" in CORE_ROOTS and "atom/config.py" in CORE_ROOTS
        assert not {"atom/compass", "atom/diffusion", "atom/plugin"} & set(CORE_ROOTS)

    @pytest.mark.parametrize(
        "module",
        [
            "atom/model_engine/scheduler.py",
            "atom/model_engine/model_runner.py",
            "atom/kv_transfer/offload/dense/connector.py",
        ],
    )
    def test_a_seeded_read_fails(self, module, tmp_path):
        """`model_runner.py` keeps `time.time` reads in other defs; a new def is not one of them."""
        target, line = _seeded(
            module, tmp_path, "\n\ndef _seeded():\n    return time.time()\n"
        )
        code, report = ClockSourceLint().check(str(tmp_path))
        assert code == 1
        assert report.splitlines()[:2] == [
            "clock-source lint: 1 real-clock read(s) on the simulated path:",
            f"  {target}:{line}  time.time  in _seeded",
        ]

    def test_a_new_read_beside_a_kept_one_fails(self, tmp_path):
        """Same file, def and call as a kept site: the count is what tells them apart."""
        target, line = _seeded(
            "atom/model_engine/engine_core.py",
            tmp_path,
            "\n\ndef _process_engine_step():\n    return time.perf_counter()\n",
        )
        code, report = ClockSourceLint().check(str(tmp_path))
        assert code == 1
        assert f"  {target}:{line}  time.perf_counter  in _process_engine_step" in (
            report.splitlines()
        )

    def test_every_entry_is_a_kept_class_with_a_reason(self):
        for site, (kind, why, count) in CORE_ALLOW_LIST.items():
            assert kind in ("K8", "K9"), site
            assert why.strip() and count > 0, site

    def test_every_entry_keeps_a_read_at_the_tip(self):
        """An entry whose read moved or went away excuses nothing and still reads as a decision."""
        reads = list(_core_reads())
        for site in CORE_ALLOW_LIST:
            one = ClockSourceLint(sites={site: ("", "", 0)})
            assert any(one.site(read) is not None for read in reads), site


class TestASubstitutedReadTakesTheLpClock:
    def test_the_metrics_refresh_stamp(self, lp_clock):
        exporter = AtomMetricsExporter()
        exporter.update({})
        assert exporter.read()[2] == lp_clock.t

    def test_the_model_card_stamp(self, lp_clock):
        assert ModelCard(id="m").created == int(lp_clock.t)

    def test_the_standalone_request_timings(self, lp_clock):
        """TTFT, TPOT and latency the Atomesh standalone service reports."""
        result = []
        future = SimpleNamespace(done=lambda: bool(result), set_result=result.extend)
        tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: "")
        lp_clock.t = 1.0
        state = SingleRequestState("r", tokenizer, future)
        for t, finished in ((3.0, False), (6.0, True)):
            lp_clock.t = t
            state.record(
                SimpleNamespace(
                    output_tokens=[5], finished=finished, finish_reason="stop"
                )
            )
        assert (result[0]["ttft"], result[0]["tpot"], result[0]["latency"]) == (
            2.0,
            3.0,
            5.0,
        )

    def test_the_real_clock_is_never_called(self, lp_clock):
        assert clock.now(_no_real_read) == lp_clock.t

    def test_a_real_run_reads_the_clock_it_passed(self):
        assert clock.now(lambda: 3.5) == 3.5
