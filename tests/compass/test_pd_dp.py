# SPDX-License-Identifier: MIT
"""A simulated 1P1D run above DP width 1, behind each form of the router.

`scripts/compass/pd_sim.sh` starts each side at ``-tp 2 --enable-dp-attention``:
two DP ranks per side, each a member of its side's engine LP. Without
``--dp-aware`` ATOM's own DP dispatch on each side picks the rank a request
runs on; an agentic cell's router is ``--dp-aware``, the router LP, and picks
it. Decode sends each write request to the prefill rank the relayed
``dp_rank`` names, on the channel of that decode and prefill rank pair. Each
form is run twice, with different `PYTHONHASHSEED`, and `check_run` holds the
two to the scenario checks.

The run tests skip as `test_pd_slice`'s do; the rest run anywhere.
"""

import json
import math
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_dp_slice import DP_MACHINE, MAX_TOKENS, Requests
from test_kv_remote_prefill import Script, connector
from test_kv_simulated_connector import PEAKS, geometry, model_for  # noqa: F401
from test_pd_slice import NEEDS_A_PD_RUN, PD, PORTS, PdTraffic
from test_vertical_slice import CONSTANT, MODEL, TREE, WALL_S, _free_port, _run_file

from atom.compass import run as compass_run
from atom.compass.clock import NER, ClockAbort, LpId, prefill_decode_table
from atom.compass.detect.scenario import check_run
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.utils.clock import LPRuntime

DP = 2
SERVER_ARGS = "-tp 2 --enable-dp-attention --enforce-eager --max-model-len 2048"
KV = "engine-D->engine-P:kv_write_req#dp{}.dp{}"


class PdRequests(Requests):
    http, stream = PdTraffic.http, PdTraffic.stream


class RouterLpRequests(Requests):
    http, stream = "traffic->router:http", "router->traffic:stream"


# Each router form: the cell's environment and its traffic LP.
FORMS = {
    "dispatch": ({}, PdRequests),
    "router_lp": ({"BENCHMARK_KIND": "aiperf_agentic"}, RouterLpRequests),
}


def run_pd_dp(out: Path, seed: str, form: str) -> dict:
    """One run under `pd_sim.sh` at DP width 2 on each side."""
    cell, requests = FORMS[form]
    out.mkdir()
    ports = {name: _free_port() for name in PORTS}
    # The write requests take DP * DP ports from KV_WRITE_REQ_PORT; successive
    # free ports are often a step of 2 apart, so the next one may be among them.
    kv = "KV_WRITE_REQ_PORT"
    while {p for n, p in ports.items() if n != kv} & set(
        range(ports[kv], ports[kv] + DP * DP)
    ):
        ports[kv] = _free_port()
    run_file = _run_file(
        out,
        clock_endpoint=f"tcp://127.0.0.1:{ports['CLOCK_PORT']}",
        data_parallel_size=DP,
        step_stub=CONSTANT,
        machine=DP_MACHINE,
        **PD,
    )
    env = dict(
        os.environ,
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
        MODEL=MODEL,
        SERVER_ARGS=SERVER_ARGS,
        **cell,
        **{name: str(port) for name, port in ports.items()},
    )
    log = out / "pd_sim.log"
    with open(log, "w") as f:
        sim = subprocess.Popen(
            ["bash", str(TREE / "scripts/compass/pd_sim.sh"), str(run_file)],
            cwd=TREE,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    watchdog = threading.Timer(WALL_S, os.killpg, (sim.pid, signal.SIGKILL))
    watchdog.start()
    try:
        deadline = time.monotonic() + WALL_S
        while "pd_sim: ready" not in log.read_text():
            assert sim.poll() is None, f"pd_sim.sh exited:\n{log.read_text()}"
            assert time.monotonic() < deadline, f"no router in {WALL_S} wall seconds"
            time.sleep(0.5)
        traffic = requests(json.loads(run_file.read_text()), ports["ROUTER_PORT"], sim)
        traffic.run()
        code = sim.wait(timeout=120)
    finally:
        watchdog.cancel()
        if sim.poll() is None:
            os.killpg(sim.pid, signal.SIGKILL)
    assert code == 0, f"pd_sim.sh exited {code}:\n{log.read_text()[-4000:]}"
    return {
        "out": out,
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "summary": json.loads((out / compass_run.SUMMARY_FILE).read_text()),
        "handed_in": sorted(f.name for f in out.glob(compass_run.LP_FILE.format("*"))),
        "events": traffic.events,
    }


@pytest.fixture(scope="module", params=FORMS)
def runs(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"pd_dp_{request.param}")
    return [run_pd_dp(root / f"seed{s}", s, request.param) for s in ("1", "2")]


@NEEDS_A_PD_RUN
def test_every_request_meets_every_scenario_check_with_a_kv_transfer_over_its_rank_pair(
    runs,
):
    run = runs[0]
    rows = [r.split() for r in run["table"].splitlines()[1:]]
    pairs = Counter(
        r[4].rpartition("#")[2]
        for r in rows
        if r[3] == "release" and ":kv_write_req#" in r[4]
    )
    print("\nKV transfers per decode.prefill rank pair:", dict(sorted(pairs.items())))
    parity = run["out"] / "parity"
    failed = check_run(
        [r["out"] for r in runs],
        MAX_TOKENS,
        [line for _, line in run["events"]],
        {"engine-P": parity / "prefill", "engine-D": parity / "decode"},
    )
    assert not failed, "\n".join(failed)
    assert sum(pairs.values()) == len(MAX_TOKENS)
    # The requests were spread over both ranks of each side.
    assert {p.split(".")[0] for p in pairs} == {"dp0", "dp1"}
    assert {p.split(".")[1] for p in pairs} == {"dp0", "dp1"}
    assert run["handed_in"] == sorted(
        compass_run.LP_FILE.format(n)
        for n in ["frontend-P", "frontend-D"]
        + [f"engine-{s}.dp{r}" for s in "PD" for r in range(DP)]
    )


# --- without a driver ---------------------------------------------------------


def test_the_authority_releases_a_write_request_to_its_prefill_rank_only(tmp_path):
    run = json.loads(_run_file(tmp_path, data_parallel_size=DP, **PD).read_text())
    ca = compass_run._RecordingAuthority(run)
    engine_p, engine_d = LpId("engine-P"), LpId("engine-D")
    write = KV.format(1, 0)
    for lp in ("traffic", "frontend-P", "frontend-D"):
        assert ca.on_request(LpId(lp), NER, math.inf, []) == []
    for lp, member in [(engine_p, "dp0"), (engine_p, "dp1"), (engine_d, "dp0")]:
        assert ca.on_request(lp, NER, math.inf, [], member=member) == []
    assert ca.started is None
    replies = ca.on_request(engine_d, NER, math.inf, [(write, 0, 1.0)], member="dp1")
    assert ca.started is not None
    released = {to: r for to, _, r in replies if isinstance(to, tuple)}
    assert released[engine_p, "dp0"][write] == [(0, 1.0)]
    assert write not in released[engine_p, "dp1"]


def test_decode_rank_0_may_not_log_rank_1s_write_request(tmp_path):
    run = json.loads(_run_file(tmp_path, data_parallel_size=DP, **PD).read_text())
    ca = compass_run._RecordingAuthority(run)
    with pytest.raises(
        ClockAbort, match=r"dp0 logged a send on .*#dp1\.dp0, which dp1"
    ):
        ca.on_request(
            LpId("engine-D"), NER, math.inf, [(KV.format(1, 0), 0, 1.0)], member="dp0"
        )


TABLE = prefill_decode_table(
    admission_path="serving",
    ipc_s=PD["router_s"],
    stream_s=PD["router_s"],
    router_s=PD["router_s"],
    kv_write_req_s=PD["kv_write_req_s"],
    dp=DP,
)


@pytest.fixture
def ranks(geometry, monkeypatch):  # noqa: F811
    """Both prefill ranks and decode rank 1, each on its own scripted runtime."""
    model = model_for(geometry, PEAKS[0])
    address = f"inproc://kv-write-req-{uuid.uuid4().hex}:5000"

    def rank(lp, role, r):
        rt = LPRuntime(LpId(lp), TABLE, Script())
        rt.start_run()
        parallel = SimpleNamespace(data_parallel_rank=r)
        half = connector(
            rt,
            monkeypatch,
            model,
            kv_role=role,
            address=address,
            parallel_config=parallel,
        )
        return rt, half

    return SimpleNamespace(
        p=[rank("engine-P", "kv_producer", r) for r in range(DP)],
        d=rank("engine-D", "kv_consumer", 1),
    )


def _transfer(ranks, served_by: int, taken_by: int):
    """A request prefill rank `served_by` finished at 0: decode rank 1 sends its
    write request, and prefill rank `taken_by` takes it at 1.0."""
    rt_d, d = ranks.d
    params = {"do_remote_prefill": True, "transfer_id": 9, "dp_rank": served_by}
    d.update_state_after_alloc(
        SimpleNamespace(
            id=9,
            kv_transfer_params=params,
            block_table=[0, 1],
            has_per_req_cache=False,
            num_cached_tokens=0,
        )
    )
    channel, seq, arrival = rt_d.send_log[-1]
    rt_p, p = ranks.p[taken_by]
    p.request_finished(
        SimpleNamespace(
            id=9,
            kv_transfer_params={"do_remote_decode": True},
            block_table=[0, 1],
            output_tokens=[7],
        )
    )
    rt_p.conn.grants.append((1.0, {channel: [(seq, arrival)]}))
    rt_p.advance_to(1.0)
    p.process_completions(KVConnectorOutput())
    return channel, p.process_completions(KVConnectorOutput())


@pytest.mark.parametrize("prefill_rank", range(DP))
def test_decode_sends_the_write_request_to_the_prefill_rank_the_relayed_dp_rank_names(
    ranks, prefill_rank
):
    channel, output = _transfer(ranks, prefill_rank, prefill_rank)
    assert channel == KV.format(1, prefill_rank)
    assert output.finished_sending == {9}


def test_a_write_request_reaching_another_prefill_rank_is_refused_by_name(ranks):
    _, d = ranks.d
    d._socks[1] = d._socks[0]  # a decode rank wired to the wrong prefill rank
    with pytest.raises(ValueError, match="is for prefill DP rank 1 and reached rank 0"):
        _transfer(ranks, 1, 0)
