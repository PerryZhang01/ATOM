# SPDX-License-Identifier: MIT
"""A process group whose ranks belong to more than one logical process is refused.

The data-parallel group is built by `stateless_init_torch_distributed_process_group`
over DP ranks. Every DP rank is a member of the one engine logical process, so
the DP2 layout passes; a map that puts the two ranks in different logical
processes is refused before either rank blocks in the rendezvous.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

import atom.utils.distributed.utils as dist_utils
from atom.compass.clock import LpId
from atom.utils import get_open_port


@pytest.fixture(autouse=True)
def short_rendezvous(monkeypatch):
    # A lone rank of a two-rank group waits out the rendezvous timeout; bound
    # it so a missing refusal fails here instead of hanging the run.
    monkeypatch.setattr(
        dist_utils, "_get_default_timeout", lambda _: timedelta(seconds=2)
    )
    # Gloo binds the address the hostname resolves to, one DNS lookup per rank;
    # a stalled lookup outlasts the timeout above. Loopback needs no lookup.
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")


def test_a_group_over_two_lps_is_refused_naming_both(monkeypatch):
    monkeypatch.setattr(
        dist_utils, "LP_OF_RANK", {0: LpId("engine-P"), 1: LpId("engine-D")}
    )

    with pytest.raises(RuntimeError) as refused:
        dist_utils.stateless_init_torch_distributed_process_group(
            "127.0.0.1", get_open_port(), 0, 2, backend="gloo"
        )

    assert str(refused.value) == (
        "a process group of 2 ranks spans logical processes engine-D, engine-P; "
        "its ranks must all belong to one"
    )


def test_a_rank_missing_from_the_map_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(dist_utils, "LP_OF_RANK", {0: LpId("engine")})

    with pytest.raises(RuntimeError) as refused:
        dist_utils.stateless_init_torch_distributed_process_group(
            "127.0.0.1", get_open_port(), 0, 3, backend="gloo"
        )

    assert str(refused.value) == (
        "a process group of 3 ranks has ranks with no logical process: 1, 2"
    )


def test_the_dp2_group_of_one_engine_lp_is_built(monkeypatch):
    monkeypatch.setattr(
        dist_utils, "LP_OF_RANK", {0: LpId("engine"), 1: LpId("engine")}
    )
    port = get_open_port()

    with ThreadPoolExecutor(2) as pool:
        ranks = [
            pool.submit(
                dist_utils.stateless_init_torch_distributed_process_group,
                "127.0.0.1",
                port,
                r,
                2,
                backend="gloo",
            )
            for r in (0, 1)
        ]
        groups = [f.result() for f in ranks]

    assert [g.size() for g in groups] == [2, 2]
    for g in groups:
        dist_utils.stateless_destroy_torch_distributed_process_group(g)
