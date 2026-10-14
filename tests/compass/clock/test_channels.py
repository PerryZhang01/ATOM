# SPDX-License-Identifier: MIT
"""`atom.compass.clock.channels`: the channel table and the distances it gives.

A grant bounds each waiting process by ``min over j of (N[j] + D(j->i))``, so a
distance that is too large hands out time a message may still need. The
distances are checked against values worked out by hand for the shipped
tables, against a brute-force enumeration of every simple channel path, and on
the worked example of an idle process in the middle of a chain.
"""

import math

import pytest

from atom.compass.clock import (
    ChannelTable,
    LpId,
    LpRegistry,
    prefill_decode_table,
    single_engine_table,
)
from atom.compass.clock.channels import ReceiveMode

T, FP, EP, FD, ED = (
    LpId(n) for n in ("traffic", "frontend-P", "engine-P", "frontend-D", "engine-D")
)

# Distinct values for the lookaheads that have no measurement, so every entry of
# the distance matrix names the path that produced it.
IPC, STREAM, ROUTER, KV = 1.0e-4, 2.0e-3, 5.0e-4, 3.0e-4


def _pd():
    return prefill_decode_table(
        admission_path="serving",
        ipc_s=IPC,
        stream_s=STREAM,
        router_s=ROUTER,
        kv_write_req_s=KV,
    )


def _single():
    return single_engine_table(admission_path="serving", ipc_s=IPC, stream_s=STREAM)


def _table(*lps):
    registry = LpRegistry()
    for lp in lps:
        registry.register(lp)
    return ChannelTable(registry)


def _enumerated(table, source, target):
    """The least summed lookahead over every simple channel path, by brute force."""
    best = math.inf
    stack = [(source, 0.0, (source,))]
    while stack:
        lp, total, seen = stack.pop()
        if lp == target:
            best = min(best, total)
            continue
        for c in table.channels_from(lp):
            if c.target not in seen:
                stack.append((c.target, total + c.lookahead_s, seen + (c.target,)))
    return best


# --- the shipped tables ------------------------------------------------------

ADMISSION = [("serving", 1.6e-3), ("offline_batch", 13.0e-3)]


def _whole(table):
    return {
        c.name: (c.source, c.target, table.lookahead(c.name), table.recv_mode(c.name))
        for lp in table.registry
        for c in table.channels_from(lp)
    }


@pytest.mark.parametrize("path, delay", ADMISSION)
def test_the_single_engine_table_is_exactly_its_channel_list(path, delay):
    table = single_engine_table(admission_path=path, ipc_s=IPC, stream_s=STREAM)
    F, E = LpId("frontend"), LpId("engine")
    assert table.registry.ids() == (E, F, T)
    assert _whole(table) == {
        "traffic->frontend:http": (T, F, delay, "inline"),
        "frontend->traffic:stream": (F, T, STREAM, "inline"),
        "frontend->engine:request#dp0": (F, E, IPC, "thread"),
        "frontend->engine:control#dp0": (F, E, IPC, "thread"),
        "engine->frontend:output#dp0": (E, F, IPC, "thread"),
    }
    assert [c.name for c in table.channels_into(E)] == [
        "frontend->engine:control#dp0",
        "frontend->engine:request#dp0",
    ]
    assert [c.name for c in table.channels_from(F)] == [
        "frontend->engine:control#dp0",
        "frontend->engine:request#dp0",
        "frontend->traffic:stream",
    ]


@pytest.mark.parametrize("path, delay", ADMISSION)
def test_the_prefill_decode_table_is_exactly_its_channel_list(path, delay):
    table = prefill_decode_table(
        admission_path=path,
        ipc_s=IPC,
        stream_s=STREAM,
        router_s=ROUTER,
        kv_write_req_s=KV,
    )
    assert table.registry.ids() == (ED, EP, FD, FP, T)
    # Behind the router the request, the stream and the relay each carry its cost.
    assert _whole(table) == {
        "traffic->frontend-P:http": (T, FP, delay + ROUTER, "inline"),
        "frontend-D->traffic:stream": (FD, T, STREAM + ROUTER, "inline"),
        "frontend-P->frontend-D:relay": (FP, FD, ROUTER, "inline"),
        "engine-D->engine-P:kv_write_req": (ED, EP, KV, "inline"),
        "frontend-P->engine-P:request#dp0": (FP, EP, IPC, "thread"),
        "frontend-P->engine-P:control#dp0": (FP, EP, IPC, "thread"),
        "engine-P->frontend-P:output#dp0": (EP, FP, IPC, "thread"),
        "frontend-D->engine-D:request#dp0": (FD, ED, IPC, "thread"),
        "frontend-D->engine-D:control#dp0": (FD, ED, IPC, "thread"),
        "engine-D->frontend-D:output#dp0": (ED, FD, IPC, "thread"),
    }


def test_the_prefill_decode_table_above_dp_width_one_has_each_ranks_channels_and_each_pairs_write_request():
    one = _pd()
    table = prefill_decode_table(
        admission_path="serving",
        ipc_s=IPC,
        stream_s=STREAM,
        router_s=ROUTER,
        kv_write_req_s=KV,
        dp=2,
    )
    whole = _whole(table)
    pairs = [
        f"engine-D->engine-P:kv_write_req#dp{d}.dp{p}" for d in (0, 1) for p in (0, 1)
    ]
    assert {n: whole.pop(n) for n in pairs} == dict.fromkeys(
        pairs, (ED, EP, KV, "inline")
    )
    # Rank 1's channels are rank 0's, renamed; the rest are the width-one table's.
    rank0 = {n: v for n, v in _whole(one).items() if "kv_write_req" not in n}
    assert whole == rank0 | {
        n.replace("#dp0", "#dp1"): v for n, v in rank0.items() if n.endswith("#dp0")
    }
    for j in table.registry.ids():
        for i in table.registry.ids():
            assert table.distance(j, i) == one.distance(j, i), (j, i)


# --- distance ----------------------------------------------------------------


def test_the_prefill_decode_distances_match_the_hand_computed_paths():
    A, S, I, R, K = 1.6e-3 + ROUTER, STREAM + ROUTER, IPC, ROUTER, KV
    expected = {
        (T, FP): A, (T, EP): A + I, (T, FD): A + R, (T, ED): A + R + I,
        (FP, T): R + S, (FP, EP): I, (FP, FD): R, (FP, ED): R + I,
        (EP, T): I + R + S, (EP, FP): I, (EP, FD): I + R, (EP, ED): I + R + I,
        (FD, T): S, (FD, FP): min(S + A, I + K + I), (FD, EP): min(S + A + I, I + K), (FD, ED): I,
        (ED, T): I + S, (ED, FP): min(I + S + A, K + I), (ED, EP): K, (ED, FD): I,
    }  # fmt: skip
    table = _pd()
    got = {pair: table.distance(*pair) for pair in expected}
    assert got == pytest.approx(expected)
    # The shorter of two routes into prefill's engine goes through the KV write request.
    assert got[FD, EP] == pytest.approx(I + K)


def test_the_single_engine_distances_match_the_hand_computed_paths():
    table = _single()
    F, E = LpId("frontend"), LpId("engine")
    assert table.distance(T, E) == pytest.approx(1.6e-3 + IPC)
    assert table.distance(E, T) == pytest.approx(IPC + STREAM)
    assert table.distance(F, E) == IPC


@pytest.mark.parametrize(
    "table", [_pd(), _single()], ids=["prefill-decode", "single-engine"]
)
def test_distance_agrees_with_every_enumerated_path(table):
    ids = table.registry.ids()
    for j in ids:
        for i in ids:
            if j != i:
                assert table.distance(j, i) == pytest.approx(
                    _enumerated(table, j, i)
                ), (j, i)


def test_a_pair_with_no_path_is_infinitely_far_apart():
    a, b, c = LpId("a"), LpId("b"), LpId("c")
    table = _table(a, b, c)
    table.declare("a->b:x", a, b, 1.0, "inline")
    assert table.distance(a, b) == 1.0
    assert table.distance(b, a) == math.inf
    assert table.distance(c, a) == math.inf
    assert table.distance(c, c) == 0.0


def test_a_channel_declared_after_a_distance_was_read_is_counted():
    a, b, c = LpId("a"), LpId("b"), LpId("c")
    table = _table(a, b, c)
    table.declare("a->b:x", a, b, 1.0, "inline")
    table.declare("b->c:x", b, c, 1.0, "inline")
    assert table.distance(a, c) == 2.0
    table.declare("a->c:x", a, c, 0.5, "inline")
    assert table.distance(a, c) == 0.5
    # A second, shorter channel between the same two ends is the one that counts.
    table.declare("a->c:y", a, c, 0.25, "thread")
    assert table.distance(a, c) == 0.25
    table.declare("a->c:z", a, c, 0.75, "thread")
    assert table.distance(a, c) == 0.25


def test_the_idle_process_in_the_middle_of_a_chain():
    # Frontend runs at 9.0 and feeds prefill (idle, so +inf) at 0.1; prefill
    # feeds decode at 0.5. Decode has no channel from frontend, but frontend's
    # request reaches it through prefill at 9.6, and the distance says so.
    frontend, prefill, decode = LpId("frontend"), LpId("prefill"), LpId("decode")
    table = _table(frontend, prefill, decode)
    table.declare("frontend->prefill:x", frontend, prefill, 0.1, "inline")
    table.declare("prefill->decode:x", prefill, decode, 0.5, "inline")
    assert [c.source for c in table.channels_into(decode)] == [prefill]
    now = {frontend: 9.0, prefill: math.inf}
    lbts = min(now[j] + table.distance(j, decode) for j in now)
    assert lbts == 9.6


# --- refusals ------------------------------------------------------------------


def test_an_undeclared_name_is_refused_with_the_declared_ones():
    table = _single()
    with pytest.raises(
        KeyError,
        match=r"'nope' is not a declared channel; declared: engine->frontend:output#dp0, ",
    ):
        table.lookahead("nope")
    with pytest.raises(KeyError, match="is not a declared channel"):
        table.recv_mode("nope")


def test_a_duplicate_name_is_refused():
    a, b = LpId("a"), LpId("b")
    table = _table(a, b)
    table.declare("a->b:x", a, b, 1.0, "inline")
    with pytest.raises(ValueError, match="'a->b:x' is already declared"):
        table.declare("a->b:x", a, b, 2.0, "inline")
    assert table.lookahead("a->b:x") == 1.0


@pytest.mark.parametrize("bad", [-1.0e-6, math.nan, math.inf])
def test_a_lookahead_that_is_not_a_finite_non_negative_duration_is_refused(bad):
    a, b = LpId("a"), LpId("b")
    with pytest.raises(ValueError, match="must be a finite number of seconds"):
        _table(a, b).declare("a->b:x", a, b, bad, "inline")


def test_a_declared_lookahead_cannot_be_rewritten_through_a_handed_out_channel():
    a, b = LpId("a"), LpId("b")
    channel = _table(a, b).declare("a->b:x", a, b, 1.0, "inline")
    with pytest.raises(AttributeError):
        channel.lookahead_s = -1.0


def test_a_zero_lookahead_is_a_declaration():
    a, b = LpId("a"), LpId("b")
    table = _table(a, b)
    table.declare("a->b:x", a, b, 0.0, "thread")
    assert table.distance(a, b) == 0.0


def test_a_channel_whose_ends_are_not_registered_is_refused():
    a, b = LpId("a"), LpId("b")
    table = _table(a)
    with pytest.raises(KeyError, match="b is not registered; registered: a"):
        table.declare("a->b:x", a, b, 1.0, "inline")
    with pytest.raises(KeyError, match="b is not registered; registered: a"):
        table.declare("b->a:x", b, a, 1.0, "inline")
    with pytest.raises(TypeError, match="require expects an LpId"):
        table.declare("a->b:x", a, "b", 1.0, "inline")


def test_a_channel_to_itself_and_an_unknown_receive_mode_are_refused():
    a, b = LpId("a"), LpId("b")
    table = _table(a, b)
    with pytest.raises(ValueError, match="joins a to itself"):
        table.declare("a->a:x", a, a, 1.0, "inline")
    with pytest.raises(
        ValueError,
        match="receive mode of 'a->b:x' must be one of thread, inline, got 'poll'",
    ):
        table.declare("a->b:x", a, b, 1.0, "poll")


def test_a_receive_mode_given_as_a_string_is_stored_as_its_member():
    a, b = LpId("a"), LpId("b")
    channel = _table(a, b).declare("a->b:x", a, b, 1.0, "inline")
    assert channel.receive is ReceiveMode.INLINE


def test_every_query_refuses_an_unregistered_process():
    a, b = LpId("a"), LpId("b")
    table = _table(a)
    for query in (table.channels_into, table.channels_from):
        with pytest.raises(KeyError, match="b is not registered; registered: a"):
            query(b)
    with pytest.raises(KeyError, match="b is not registered; registered: a"):
        table.distance(a, b)
