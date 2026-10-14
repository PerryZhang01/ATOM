# SPDX-License-Identifier: MIT
"""One server, two client runs: each takes the grants a fresh server gives it.

The authority is built naming the traffic LP its client. When every LP waits
and none is grantable, the open client run ends: the client is granted
``+inf``, every server LP its own clock marked with the client run's index,
and nothing more is granted until the next client binds, where the last
one's latest clock left it. The second client's grants, less its start, are
the grants a fresh server gives the same trace, to rounding, since the start
is a sum of times on no binary grid; and so is its engine's step count.
"""

import collections
import dataclasses
import math

import pytest

from atom.compass.clock import NER, ClockAuthority, LpId, single_engine_table
from atom.compass.clock_transport import connect, serve
from atom.utils.clock import LPRuntime

from .deployments import DEPLOYMENTS, IPC_S, STREAM_S
from .harness import SyntheticRun, _Counted
from .participants import DESIGN_WORKLOAD, Engine, Traffic

TRAFFIC = LpId("traffic")
INF = math.inf
#: Two traces with different prompt lengths, counts and pacing. A synthetic
#: engine's steps per request are its server's, so both take the same.
TRACES = (
    dataclasses.replace(DESIGN_WORKLOAD, requests=4, decode_steps=3),
    dataclasses.replace(
        DESIGN_WORKLOAD,
        requests=3,
        decode_steps=3,
        arrival_interval_seconds=1.5,
        prompt_tokens=(2048,),
    ),
)


class Joined(Traffic):
    """A traffic LP that binds at `start` with the authority's sequence numbers:
    its trace and its scrape run from there."""

    def __init__(self, table, workload, http, scrape_on, start, seqs):
        super().__init__(table, workload, http, scrape_on)
        self.clock = start
        self.next_scrape = start + self.interval
        self.arrivals = collections.deque((start + t, *r) for t, *r in self.arrivals)
        self._seq.update(seqs)


class ClientRuns(SyntheticRun):
    """`SyntheticRun` serving one client per trace. The next client binds once
    every server LP has asked again after the end."""

    def __init__(self, deployment, traces):
        super().__init__(deployment, traces[0])
        self.clock = _Counted(self.table, client=TRAFFIC, client_runs=len(traces))
        self.traces = traces
        self.clients = [(0.0, len(self.reply_log[TRAFFIC]), self.lps[TRAFFIC])]
        self.engine_steps = []  # {engine: steps} at each client run's end
        self.between = []  # what the authority granted between client runs
        self.at_join = []  # the LP table as each next client binds

    def run(self):
        ready = {lp: person.run(0.0) for lp, person in self.lps.items()}
        joining = False
        while ready or joining:
            if not ready:
                ready[TRAFFIC] = self._join()
                joining = False
                continue
            lp = min(ready, key=self.rank.__getitem__)
            kind, t, t_daemon = ready.pop(lp)
            replies = self.clock.on_request(lp, kind, t, self._post(lp), t_daemon)
            if joining:
                self.between += replies
            for i, g, released, *ended in replies:
                messages = self._hand_over(i, g, released)
                self.reply_log[i].append((g, tuple((c, a) for a, c, _, _ in messages)))
                self._refuse_step_over(i, g)
                if i == TRAFFIC and g == INF:
                    self.engine_steps.append(
                        {
                            str(e): p.steps
                            for e, p in self.lps.items()
                            if isinstance(p, Engine)
                        }
                    )
                if g == INF:
                    self.lps[i].close()
                    joining = i == TRAFFIC and len(self.clients) < len(self.traces)
                    continue
                ready[i] = self.lps[i].on_grant(g, messages)
        return self

    def _join(self):
        self.at_join.append(self.clock.lp_table())
        start, seqs = self.clock.on_bind(TRAFFIC)
        old = self.lps[TRAFFIC]
        client = Joined(
            self.table,
            self.traces[len(self.clients)],
            old.http,
            old.scrape_on,
            start,
            seqs,
        )
        self.lps[TRAFFIC] = client
        self.granted[TRAFFIC] = start
        self.clients.append((start, len(self.reply_log[TRAFFIC]), client))
        return client.run(start)

    def client_log(self, k):
        """Client `k`'s replies, each time less its start."""
        start, first, _ = self.clients[k]
        last = self.clients[k + 1][1] if k + 1 < len(self.clients) else None
        return [
            (g - start, tuple((c, a - start) for c, a in msgs))
            for g, msgs in self.reply_log[TRAFFIC][first:last]
        ]


def _fresh(deployment, trace):
    """A fresh server's run of `trace`: the traffic LP's replies, each engine's
    steps."""
    run = SyntheticRun(deployment, trace)
    run.run()
    log = [(g, tuple((c, a) for c, _, a in msgs)) for g, msgs in run.reply_log[TRAFFIC]]
    return log, {str(e): p.steps for e, p in run.lps.items() if isinstance(p, Engine)}


def _channels(log):
    return [[c for c, _ in msgs] for _, msgs in log]


def _times(log):
    return [t for g, msgs in log for t in (g, *(a for _, a in msgs))]


@pytest.fixture(scope="module", params=sorted(DEPLOYMENTS))
def served(request):
    return request.param, ClientRuns(request.param, TRACES).run()


def test_each_client_run_takes_the_grants_a_fresh_server_gives_its_trace(served):
    deployment, run = served
    before = collections.Counter()
    for k, trace in enumerate(TRACES):
        log, steps = _fresh(deployment, trace)
        ran = dict(collections.Counter(run.engine_steps[k]) - before)
        before = collections.Counter(run.engine_steps[k])
        print(f"\n{deployment} client run {k}: {len(log)} replies, steps {ran}")
        mine = run.client_log(k)
        assert _channels(mine) == _channels(log)
        assert _times(mine) == pytest.approx(_times(log), rel=0, abs=1e-12)
        assert ran == steps
        assert all(steps.values())


def test_the_next_client_starts_where_the_last_run_ended(served):
    _, run = served
    first, second = run.clock.ended
    print(f"\nclient runs: {first.start}..{first.end}, {second.start}..{second.end}")
    assert (first.index, first.start, second.index) == (0, 0.0, 1)
    assert second.start == first.end == run.clients[1][0] > 0
    assert first.end == max(clock for _, clock in first.final_clocks)
    assert run.clock.final_clocks == second.final_clocks
    assert run.clock.recovered == 0


def test_nothing_is_granted_between_client_runs(served):
    _, run = served
    (table,) = run.at_join
    end = run.clock.ended[0].end
    print("\nat the join:\n" + "\n".join(map(repr, table)))
    assert run.between == []
    for row in table:
        if row.lp == TRAFFIC:
            assert (row.state, row.now) == ("running", end)
        else:
            # Waiting, with every daemon deadline held above the end.
            assert row.state == NER and row.n > end and not row.undelivered


def test_a_server_with_no_client_serves_one_client_run():
    table = DEPLOYMENTS["single-deployment"]()
    for client, runs in [(None, 2), (TRAFFIC, 0), (TRAFFIC, math.nan)]:
        with pytest.raises(ValueError, match="a client LP serves one or more"):
            ClockAuthority(table, client=client, client_runs=runs)


# --- over TCP: the client resigns, its socket closes, the next one binds ------


def _ask(conn, request):
    conn.send(request)
    return conn.recv()


def test_a_resigned_client_closes_and_the_next_binds_where_it_left(tmp_path):
    table = single_engine_table(
        admission_path="serving", ipc_s=IPC_S, stream_s=STREAM_S
    )
    http, stream = "traffic->frontend:http", "frontend->traffic:stream"
    L, S = table.lookahead(http), table.lookahead(stream)
    server = serve(
        ClockAuthority(table, client=TRAFFIC, client_runs=2), "tcp://127.0.0.1:0"
    )
    try:
        engine = connect(LpId("engine"), server.endpoint)
        frontend = connect(LpId("frontend"), server.endpoint)
        traffic = connect(TRAFFIC, server.endpoint)
        assert traffic.joined == (0.0, {http: 0})
        # Client run 0: one request, answered on the stream.
        engine.send((NER, INF, [], 8.0))
        frontend.send((NER, INF, [], INF))
        traffic.send((NER, INF, [(http, 0, L)], INF))
        assert frontend.recv() == (
            L,
            {http: [(0, L)], "engine->frontend:output#dp0": []},
        )
        frontend.send((NER, INF, [(stream, 0, L + S)], INF))
        assert traffic.recv() == (L + S, {stream: [(0, L + S)]})
        traffic.send((NER, INF, [], INF))
        # Its end: the client resigns, each server LP is marked at its own clock.
        assert engine.recv() == (
            0.0,
            {"frontend->engine:control#dp0": [], "frontend->engine:request#dp0": []},
            0,
        )
        assert frontend.recv()[0::2] == (L, 0)
        assert traffic.recv()[0] == INF
        traffic.close()
        # The server LPs ask again and nothing is granted; the closed socket
        # ended nothing, so the next client binds the same LP.
        engine.send((NER, INF, [], 8.0))
        frontend.send((NER, INF, [], INF))
        nxt = connect(TRAFFIC, server.endpoint)
        assert nxt.joined == (L + S, {http: 1})
        rt = LPRuntime(TRAFFIC, table, nxt)
        assert (rt.now, rt.next_seq) == (L + S, {http: 1})
        rt.start_run()
        arrival, seq = rt.stamp_send(http)
        assert (arrival, seq) == (L + S + L, 1)
        rt.ask_next_event(INF)
        assert frontend.recv()[0] == arrival
        frontend.send((NER, INF, [(stream, 1, arrival + S)], INF))
        assert rt.take_grant() == arrival + S
        # Client run 1 is the last: its end is the finish.
        rt.ask_next_event(INF)
        assert rt.take_grant() == INF
        assert engine.recv()[0] == frontend.recv()[0] == INF
    finally:
        server.close()


def test_a_marked_grant_calls_the_lp_s_hand_in_with_the_client_run(tmp_path):
    table = single_engine_table(
        admission_path="serving", ipc_s=IPC_S, stream_s=STREAM_S
    )
    endpoint = f"inproc:{tmp_path}"
    server = serve(ClockAuthority(table, client=TRAFFIC, client_runs=3), endpoint)
    try:
        rts = {
            lp: LPRuntime(LpId(lp), table, connect(LpId(lp), endpoint))
            for lp in ("engine", "frontend")
        }
        called = []
        for name, rt in rts.items():
            rt.client_run_ended = lambda k, name=name: called.append((name, k))
        for k in range(2):
            for rt in rts.values():
                rt.ask_next_event(INF)
            traffic = LPRuntime(TRAFFIC, table, connect(TRAFFIC, endpoint))
            assert traffic.next_event(INF) == INF
            assert [rt.take_grant() for rt in rts.values()] == [0.0, 0.0]
        assert called == [(n, k) for k in range(2) for n in rts]
    finally:
        server.close()
