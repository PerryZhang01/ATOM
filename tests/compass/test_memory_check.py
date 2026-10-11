# SPDX-License-Identifier: MIT
"""A simulated run's memory file, held against a real server's log per term.

`run.runner` writes the readings it installs into the run's out_dir, and
`python -m atom.compass.memory.check` compares them with the lines a real
ATOM server logs, through `compare` and `compare_graph_pool`. Each case
installs through `run.runner` on a Qwen3.5-27B config at TP1 on the suite's
machine document, as `test_run_graph_pool.py` does.
"""

import json

import pytest
from test_memory_readings import CAPTURE_SIZES, MAX_NUM_BATCHED_TOKENS
from test_run_graph_pool import _reserved
from test_vertical_slice import _run_file

from atom.compass import run as compass_run
from atom.compass.memory import MemoryRefusal
from atom.compass.memory.check import main

GIB = 1 << 30
#: The spec's graph-pool line at width 1, in the suite's machine document.
POOL_BASE, POOL_PER_TOKEN = 95_500_000, 318_000


def _memory(monkeypatch, tmp_path, **kwargs) -> dict:
    """The memory file `run.runner` wrote, with the reservation it installed."""
    reserved = _reserved(monkeypatch, tmp_path, **kwargs)
    saved = json.loads((tmp_path / compass_run.MEMORY_FILE.format("dp0")).read_text())
    return saved | {"reserved": reserved}


def _total(saved, name) -> int:
    return sum(row["nbytes"] for row in saved["readings"][name])


def _log(path, *, peak, non_torch, tokens=MAX_NUM_BATCHED_TOKENS, pool=None, ranks=1):
    """A real server's log: the lines ATOM writes, at the given byte counts."""
    warmup = (
        "[atom 08:31:41] Model Runner0/1: warmup_model 40.35 seconds with 1 "
        f"reqs {tokens} tokens"
    )
    budget = (
        "[atom 08:32:21] Memory budget: total_gpu=191.98GB, free=139.49GB, "
        f"utilization=0.9, budget=172.79GB, peak_torch={peak / GIB:.2f}GB, "
        f"non_torch={non_torch / GIB:.2f}GB, cudagraph_est=0.55GB, "
        "safety=3.84GB, available_for_kv=113.24GB, block_bytes=1056768, "
        "num_kvcache_blocks=77049"
    )
    lines = [warmup] + [budget] * ranks
    if pool is not None:
        lines.append(
            "[atom 08:32:24] CUDA graph capture memory: 11 graphs | "
            f"pool(reserved)={pool / GIB:.2f}GB allocated=0.57GB"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _check(capsys, tmp_path, log) -> dict[str, str]:
    """The command's output, and the first row naming each term."""
    main([str(tmp_path / compass_run.MEMORY_FILE.format("dp0")), str(log)])
    out = capsys.readouterr().out
    rows = {
        line.split()[0]: line
        for line in reversed(out.splitlines())
        if line.startswith("  ")
    }
    return {"out": out} | rows


@pytest.mark.parametrize(
    "eager,piecewise,captured",
    [
        (True, False, None),
        (False, False, sum(s for s in CAPTURE_SIZES if s <= 256)),
        (False, True, sum(s for s in CAPTURE_SIZES if s <= 256)),
    ],
    ids=["eager", "whole-graph", "piecewise"],
)
def test_a_simulated_run_writes_the_readings_it_sized_the_pool_from(
    monkeypatch, tmp_path, eager, piecewise, captured
):
    saved = _memory(monkeypatch, tmp_path, eager=eager, piecewise=piecewise)
    assert saved["tokens"] == MAX_NUM_BATCHED_TOKENS
    assert _total(saved, "cudagraph_overhead") == saved["reserved"]
    assert set(saved["readings"]) == {"peak_torch", "non_torch", "cudagraph_overhead"}
    if captured is None:
        assert saved["predicts"] is None
    else:
        pool = sum(row["nbytes"] for row in saved["predicts"])
        assert pool == POOL_BASE + POOL_PER_TOKEN * captured


@pytest.mark.parametrize("factor,verdict", [(1.0, "pass"), (2.0, "fail")])
def test_each_reading_is_one_row_with_its_own_verdict(
    monkeypatch, tmp_path, capsys, factor, verdict
):
    saved = _memory(monkeypatch, tmp_path, eager=False)
    log = _log(
        tmp_path / "real.log",
        peak=_total(saved, "peak_torch"),
        non_torch=_total(saved, "non_torch") * factor,
        pool=600_000_000,
    )
    rows = _check(capsys, tmp_path, log)
    # Within the gate, but it sums declared terms, so it is not discharged.
    assert rows["peak_torch"].endswith("not discharged")
    assert (
        "* peak_torch: within the gate, and the gate is not discharged" in rows["out"]
    )
    assert rows["non_torch"].endswith(verdict)
    # The predicted readings' own terms are printed under the table.
    assert rows["weights"].split()[2] == "declared"
    assert "graph pool against pool(reserved)" in rows["out"]
    assert rows["reserves()"].split()[-2] == f"{saved['reserved']:,}"


def test_a_real_run_at_another_warmup_shape_refuses_peak_torch_only(
    monkeypatch, tmp_path, capsys
):
    saved = _memory(monkeypatch, tmp_path, eager=True)
    log = _log(
        tmp_path / "real.log",
        peak=_total(saved, "peak_torch"),
        non_torch=_total(saved, "non_torch"),
        tokens=MAX_NUM_BATCHED_TOKENS * 2,
    )
    rows = _check(capsys, tmp_path, log)
    assert "! peak_torch: this term was taken at a shape on each side" in rows["out"]
    assert rows["non_torch"].endswith("pass")
    assert "graph pool not compared: the simulated run captures no graph and " in (
        rows["out"]
    )


def test_a_log_of_two_ranks_is_refused(monkeypatch, tmp_path):
    _memory(monkeypatch, tmp_path, eager=True)
    log = _log(tmp_path / "real.log", peak=GIB, non_torch=GIB, ranks=2)
    with pytest.raises(MemoryRefusal, match="has 2 lines matching"):
        main([str(tmp_path / compass_run.MEMORY_FILE.format("dp0")), str(log)])


def test_the_authority_removes_an_earlier_runs_memory_file(tmp_path):
    stale = tmp_path / compass_run.MEMORY_FILE.format("P.dp0")
    stale.write_text("{}")
    compass_run._RecordingAuthority(json.loads(_run_file(tmp_path).read_text()))
    assert not stale.exists()
