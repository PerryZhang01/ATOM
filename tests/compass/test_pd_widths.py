# SPDX-License-Identifier: MIT
"""A simulated 1P1D run whose prefill and decode sides run at different DP widths.

The run file's ``prefill_data_parallel_size`` and ``decode_data_parallel_size``
give each side its width. A side above width one is an engine LP with a member
per rank; a side at width one is an engine LP alone. Decode sends each write
request on the channel of its own rank and the prefill rank that served the
request, so the pairs are the decode ranks by the prefill ranks.

The run tests start each pair of widths under each router form, as
`test_pd_dp` does at width 2 on both sides, and skip as it does; the rest run
anywhere.
"""

import json
import math
import uuid
from collections import Counter
from types import SimpleNamespace

import pytest
from test_dp_slice import MAX_TOKENS, _deployment
from test_kv_remote_prefill import Script, connector
from test_kv_simulated_connector import PEAKS, geometry, model_for  # noqa: F401
from test_pd_dp import FORMS, run_pd_dp
from test_pd_slice import NEEDS_A_PD_RUN, PD
from test_vertical_slice import _run_file

from atom.compass import run as compass_run
from atom.compass.clock import NER, LpId
from atom.compass.detect.scenario import check_run
from atom.utils.clock import LPRuntime

#: (prefill width, decode width)
SIDES = [(2, 1), (1, 2)]
KV = "engine-D->engine-P:kv_write_req#dp{}.dp{}"


def _run(tmp_path, p, d, **more) -> dict:
    widths = {"prefill_data_parallel_size": p, "decode_data_parallel_size": d}
    return json.loads(_run_file(tmp_path, **widths, **PD, **more).read_text())


@pytest.mark.parametrize("p, d", SIDES)
def test_each_side_has_its_own_ranks_and_a_write_request_channel_per_rank_pair(
    tmp_path, p, d
):
    table = compass_run.channel_table(_run(tmp_path, p, d))
    names = {c.name for lp in table.registry for c in table.channels_from(lp)}
    for side, n in (("P", p), ("D", d)):
        assert {x for x in names if x.startswith(f"frontend-{side}->engine")} == {
            f"frontend-{side}->engine-{side}:{kind}#dp{r}"
            for kind in ("request", "control")
            for r in range(n)
        }
    assert {x for x in names if ":kv_write_req" in x} == {
        KV.format(r, q) for r in range(d) for q in range(p)
    }


@pytest.mark.parametrize("p, d", SIDES)
def test_only_the_side_above_width_one_joins_its_engine_lp_as_members(tmp_path, p, d):
    wide, narrow = ("P", "D") if p > d else ("D", "P")
    assert sorted(compass_run._hand_ins(_run(tmp_path, p, d))) == sorted(
        ["frontend-P", "frontend-D", f"engine-{narrow}"]
        + [f"engine-{wide}.dp{r}" for r in range(2)]
    )


def test_a_write_request_from_a_decode_of_width_one_reaches_its_prefill_rank_only(
    tmp_path,
):
    ca = compass_run._RecordingAuthority(_run(tmp_path, 2, 1))
    engine_p, engine_d = LpId("engine-P"), LpId("engine-D")
    write = KV.format(0, 1)
    for lp in ("traffic", "frontend-P", "frontend-D"):
        assert ca.on_request(LpId(lp), NER, math.inf, []) == []
    for member in ("dp0", "dp1"):
        assert ca.on_request(engine_p, NER, math.inf, [], member=member) == []
    assert ca.started is None
    replies = ca.on_request(engine_d, NER, math.inf, [(write, 0, 1.0)])
    assert ca.started is not None
    released = {to: r for to, _, r in replies if isinstance(to, tuple)}
    assert released[engine_p, "dp1"][write] == [(0, 1.0)]
    assert write not in released[engine_p, "dp0"]


@pytest.mark.parametrize(
    "side, deployment, key",
    [
        ("kv_producer", _deployment(tp=2, dp_attention=True), "prefill"),
        ("kv_consumer", _deployment(), "decode"),
    ],
)
def test_each_side_is_held_to_its_own_width(
    monkeypatch, tmp_path, side, deployment, key
):
    """Prefill at width 1 and decode at width 2: a prefill deployment at 2 and a
    decode one at 1 each run the other side's width, and each is refused by its
    own side's key."""
    deployment.kv_transfer_config = {"kv_connector": "compass", "kv_role": side}
    monkeypatch.setenv(compass_run.CLOCK_ENV, "tcp://127.0.0.1:1")
    run = _run_file(
        tmp_path, prefill_data_parallel_size=1, decode_data_parallel_size=2, **PD
    )
    monkeypatch.setenv(compass_run.ENV, str(run))
    width = 1 if key == "prefill" else 2
    with pytest.raises(ValueError, match=f"{key}_data_parallel_size is {width}"):
        compass_run.frontend(deployment)


def test_a_side_width_in_a_run_with_no_sides_is_refused_by_name(tmp_path):
    run = json.loads(_run_file(tmp_path, decode_data_parallel_size=2).read_text())
    with pytest.raises(ValueError, match="decode_data_parallel_size, and only"):
        compass_run.channel_table(run)


@pytest.mark.parametrize("p, d", SIDES)
def test_decode_writes_to_the_prefill_rank_that_served_it_across_two_widths(
    tmp_path, geometry, monkeypatch, p, d  # noqa: F811
):
    """Every decode rank writes to every prefill rank, each on its own pair's
    channel, and each prefill rank's connector takes only its own."""
    table = compass_run.channel_table(_run(tmp_path, p, d))
    model = model_for(geometry, PEAKS[0])
    address = f"inproc://kv-write-req-{uuid.uuid4().hex}:5000"

    def rank(lp, role, r):
        rt = LPRuntime(LpId(lp), table, Script())
        rt.start_run()
        half = connector(
            rt,
            monkeypatch,
            model,
            kv_role=role,
            address=address,
            parallel_config=SimpleNamespace(data_parallel_rank=r),
        )
        return rt, half

    prefill = [rank("engine-P", "kv_producer", r) for r in range(p)]
    sent = Counter()
    for r in range(d):
        rt_d, half = rank("engine-D", "kv_consumer", r)
        for q in range(p):
            params = {"do_remote_prefill": True, "transfer_id": q, "dp_rank": q}
            half.update_state_after_alloc(
                SimpleNamespace(
                    id=10 * r + q,
                    kv_transfer_params=params,
                    block_table=[0, 1],
                    has_per_req_cache=False,
                    num_cached_tokens=0,
                )
            )
            sent[rt_d.send_log[-1][0]] += 1
    assert sent == {KV.format(r, q): 1 for r in range(d) for q in range(p)}
    for q, (_, half) in enumerate(prefill):
        assert {peer_rank for _, peer_rank in half._socks} == set(range(d)), q


# --- a run at each pair of widths -------------------------------------------


@pytest.fixture(
    scope="module",
    params=[(form, sides) for form in FORMS for sides in SIDES],
    ids=lambda x: f"{x[0]}-P{x[1][0]}D{x[1][1]}",
)
def runs(request, tmp_path_factory):
    form, sides = request.param
    root = tmp_path_factory.mktemp(f"pd_widths_{form}_{sides[0]}{sides[1]}")
    return sides, [run_pd_dp(root / f"seed{s}", s, form, sides) for s in ("1", "2")]


@NEEDS_A_PD_RUN
def test_two_widths_meet_every_scenario_check_with_a_transfer_per_request(runs):
    (p, d), runs = runs
    run = runs[0]
    rows = [r.split() for r in run["table"].splitlines()[1:]]
    pairs = Counter(
        r[4].rpartition("#")[2]
        for r in rows
        if r[3] == "release" and ":kv_write_req#" in r[4]
    )
    print(f"\nP{p}D{d} KV transfers per decode.prefill rank pair:", dict(pairs))
    parity = run["out"] / "parity"
    failed = check_run(
        [r["out"] for r in runs],
        MAX_TOKENS,
        [line for _, line in run["events"]],
        {"engine-P": parity / "prefill", "engine-D": parity / "decode"},
    )
    assert not failed, "\n".join(failed)
    assert sum(pairs.values()) == len(MAX_TOKENS)
    assert {x.split(".")[0] for x in pairs} == {f"dp{r}" for r in range(d)}
    assert {x.split(".")[1] for x in pairs} == {f"dp{r}" for r in range(p)}
    members = {"P": p, "D": d}
    assert run["handed_in"] == sorted(
        compass_run.LP_FILE.format(n)
        for n in ["frontend-P", "frontend-D"]
        + [
            f"engine-{s}" if n == 1 else f"engine-{s}.dp{r}"
            for s, n in members.items()
            for r in range(n)
        ]
    )
