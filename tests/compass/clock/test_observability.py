# SPDX-License-Identifier: MIT
"""`atom.compass.clock.observability` over the Clock Authority's real replies.

* **The timeline**: off by default, the same replies on and off, no log call
  when off, one record per reply in issue order, and the recovery marker on
  exactly the replies the recovery branch issued.
* **The dump** renders `lp_table()` and nothing else, with the binding term
  marked, and no term marked when none is finite.
* **The summary**: the schedule half is identical under every submit order, a
  finished run reports the clocks it finished from, refusals tally by source,
  and the record survives strict JSON.
"""

import json
import math
import re

import pytest

from atom.compass.clock import (
    NER,
    SPEED_TARGET_RATIO,
    TAR,
    ClockAuthority,
    DetectorState,
    LpId,
    RefusalTally,
    RunSummary,
    TimelineLog,
    lp_table_dump,
    single_engine_table,
)
from tests.compass.clock.test_grant_rule import (
    PICKS,
    SEEDS,
    _drive,
    _grants,
    _Recorded,
    _table,
    _two_way,
)

A, B, C = LpId("a"), LpId("b"), LpId("c")
INF = math.inf


class _Logged(_Recorded):
    """Keeps a timeline and every reply `on_request` returned, in order."""

    def __init__(self, channels):
        super().__init__(channels)
        self.timeline = TimelineLog()
        self.issued = []

    def on_request(self, lp, kind, t, log, t_daemon=INF):
        replies = super().on_request(lp, kind, t, log, t_daemon)
        self.issued += [(i, g) for i, g, _ in replies]
        return replies


# --- the timeline ------------------------------------------------------------


def test_the_timeline_is_off_unless_a_run_asks_for_it():
    assert ClockAuthority(_two_way(0.5)).timeline is None


def test_a_run_with_the_log_on_issues_exactly_the_replies_it_issues_with_it_off():
    for seed in SEEDS:
        quiet = _drive(seed, PICKS["shuffled"], cls=_Recorded, shuffle=3)[0]
        loud = _drive(seed, PICKS["shuffled"], cls=_Logged, shuffle=3)[0]
        assert quiet == loud, seed


def test_one_record_per_reply_in_issue_order_with_recovery_marked_exactly():
    recovered = strict = 0
    for seed in SEEDS:
        _, _, ca = _drive(seed, PICKS["name order"], cls=_Logged)
        records = ca.timeline.records()
        assert [(r.lp, r.time_to) for r in records] == ca.issued
        assert [(str(r.lp), r.time_to) for r in records if r.recovered] == ca.recovered
        clock = dict.fromkeys(ca.grants, 0.0)
        for r in records:
            assert r.time_from == clock[r.lp.name]
            assert r.kind in (TAR, NER)
            clock[r.lp.name] = r.time_to
        assert len(records) == sum(ca.grants.values()) + len(ca.grants)
        recovered += len(ca.recovered)
        strict += sum(ca.grants.values()) - len(ca.recovered)
    assert recovered > 0 and strict > 0


def test_a_zero_lookahead_pair_marks_the_recovery_grant_and_nothing_else():
    ca = ClockAuthority(_two_way(0.0), TimelineLog())
    ca.on_request(A, TAR, 10.3, [])
    ca.on_request(B, TAR, 10.3, [])
    ca.on_request(A, NER, INF, [("a->b:m", 0, 10.3)])
    assert ca.timeline.lines() == ("a 0.0 10.3 TAR recovery", "b 0.0 10.3 TAR -")


def test_an_ner_reply_is_recorded_as_ner():
    ca = ClockAuthority(_two_way(0.5), TimelineLog())
    ca.on_request(A, NER, 1.0, [])
    ca.on_request(B, TAR, 2.0, [])
    assert ca.timeline.lines()[0] == "a 0.0 1.0 NER -"


def _two_lps_ask(timeline):
    ca = ClockAuthority(_two_way(0.5), timeline)
    for k in range(1, 21):
        ca.on_request(A, TAR, float(k), [])
        ca.on_request(B, TAR, float(k), [])
    ca.on_request(A, NER, INF, [])
    ca.on_request(B, NER, INF, [])


def test_a_run_with_the_log_off_never_calls_it(monkeypatch):
    # A count, not a timing: the call costs about a microsecond a reply, less
    # than a loaded host adds to any wall-clock comparison.
    calls = []
    monkeypatch.setattr(TimelineLog, "record", lambda self, *args: calls.append(args))
    _two_lps_ask(None)
    assert calls == []
    _two_lps_ask(TimelineLog())
    assert calls


def test_a_record_is_five_columns_and_names_the_lp_verbatim():
    _, _, ca = _drive(7, PICKS["reverse"], cls=_Logged)
    for entry, line in zip(ca.timeline.records(), ca.timeline.lines(), strict=True):
        lp, time_from, time_to, kind, marker = line.split(" ")
        assert lp == str(entry.lp)
        assert (float(time_from), float(time_to)) == (entry.time_from, entry.time_to)
        assert kind == entry.kind
        assert marker == ("recovery" if entry.recovered else "-")


def test_a_sink_sees_every_line_as_it_is_written():
    seen = []
    ca = ClockAuthority(_two_way(0.0), TimelineLog(seen.append))
    ca.on_request(A, TAR, 1.0, [])
    assert seen == []
    ca.on_request(B, TAR, 1.0, [])
    assert seen == list(ca.timeline.lines()) == ["a 0.0 1.0 TAR recovery"]


def test_a_sink_that_is_not_callable_is_refused_where_it_is_handed_over():
    with pytest.raises(TypeError, match="sink must be callable"):
        TimelineLog("timeline.log")


def _logging_to(timeline):
    def build(table):
        ca = _Logged(table)
        ca.timeline = timeline
        return ca

    return build


def test_a_streaming_log_can_be_asked_not_to_keep_what_it_streamed():
    seen = []
    streaming = TimelineLog(seen.append, retain=False)
    kept = _drive(3, PICKS["name order"], cls=_Logged)[2].timeline
    _drive(3, PICKS["name order"], cls=_logging_to(streaming))
    assert seen == list(kept.lines())
    with pytest.raises(ValueError, match="asked not to retain"):
        streaming.records()


def test_a_log_that_would_write_nothing_anywhere_is_refused():
    with pytest.raises(ValueError, match="would write nothing"):
        TimelineLog(retain=False)


def test_the_finish_records_one_reply_per_lp_from_the_clock_it_finished_at():
    table = single_engine_table(admission_path="serving", ipc_s=1.0e-4, stream_s=2.0e-3)
    engine, frontend, traffic = table.registry.ids()
    ca = ClockAuthority(table, TimelineLog())
    ca.on_request(engine, NER, INF, [])
    ca.on_request(frontend, NER, INF, [])
    assert _grants(ca.on_request(traffic, TAR, 2.0, [])) == [("traffic", 2.0)]
    ca.on_request(traffic, NER, INF, [])
    assert ca.timeline.lines() == (
        "traffic 0.0 2.0 TAR -",
        "engine 0.0 inf NER -",
        "frontend 0.0 inf NER -",
        "traffic 2.0 inf NER -",
    )
    assert ca.final_clocks == ((engine, 0.0), (frontend, 0.0), (traffic, 2.0))
    # Every reply of the finish is +inf, and the table says so.
    assert [row.now for row in ca.lp_table()] == [INF, INF, INF]


# --- the dump ----------------------------------------------------------------


def _chain():
    """a -> b -> c, lookahead 0.5 each: a runs, b waits in TAR, c has run to 0.5."""
    ca = ClockAuthority(_table(("a", "b", "c"), [("a", "b", 0.5), ("b", "c", 0.5)]))
    ca.on_request(B, TAR, 0.25, [("b->c:m", 0, 0.5)])
    ca.on_request(C, NER, INF, [])
    ca.on_request(B, TAR, 3.0, [])
    return ca


def test_the_dump_renders_every_lp_and_nothing_else():
    dump = lp_table_dump(_chain())
    assert dump.splitlines() == [
        "a running now 0.0s target none N 0.0s",
        "    from b N+D inf",
        "    from c N+D inf",
        "b TAR now 0.25s target 3.0s N 3.0s",
        "    from a N+D 0.5s <- binds",
        "    from c N+D inf",
        "c running now 0.5s target none N 0.5s",
        "    from a N+D 1.0s <- binds",
        "    from b N+D 3.5s",
    ]


def test_the_dump_lists_undelivered_messages_and_marks_no_term_when_none_is_finite():
    # b's peers both idle in NER(inf), so every term of b's row is infinite.
    ca = ClockAuthority(_table(("a", "b", "c"), [("a", "b", 0.5)]))
    ca.on_request(A, NER, INF, [("a->b:m", 0, 4.0)])
    ca.on_request(C, NER, INF, [])
    _, row_b, _ = ca.lp_table()
    assert [term for _, term in row_b.row] == [INF, INF]
    assert row_b.binding is None
    lines = lp_table_dump(ca).splitlines()
    assert "<- binds" not in lp_table_dump(ca)
    assert lines[3:7] == [
        "b running now 0.0s target none N 0.0s",
        "    undelivered a->b:m seq 0 at 4.0s",
        "    from a N+D inf",
        "    from c N+D inf",
    ]


# --- the summary -------------------------------------------------------------


def test_the_schedule_half_is_identical_under_every_submit_order():
    distinct = []
    for seed in SEEDS:
        records = [
            json.dumps(RunSummary.of(_drive(seed, pick)[2], 0.25).schedule_record())
            for pick in PICKS.values()
        ]
        records += [
            json.dumps(
                RunSummary.of(
                    _drive(seed, PICKS["shuffled"], shuffle=s)[2], 0.25
                ).schedule_record()
            )
            for s in range(1, 5)
        ]
        assert len(dict.fromkeys(records)) == 1, seed
        distinct.append(records[0])
    assert len(dict.fromkeys(distinct)) > 1


def test_a_finished_run_reports_the_clocks_it_finished_from_not_infinity():
    replies, _, ca = _drive(11, PICKS["name order"])
    last = {
        str(i): max([g for g, _ in got if g != INF], default=0.0)
        for i, got in replies.items()
    }
    summary = RunSummary.of(ca, 0.25)
    assert dict(summary.final_clocks) == last
    assert summary.simulated_seconds == max(last.values())
    assert summary.schedule_record()["grants_total"] == sum(ca.grants.values())


def test_grants_per_lp_match_the_finite_replies_each_lp_received():
    for seed in SEEDS:
        replies, _, ca = _drive(seed, PICKS["name order"])
        finite = {str(i): sum(g != INF for g, _ in got) for i, got in replies.items()}
        assert dict(RunSummary.of(ca, 0.25).grants) == finite, seed


def test_a_run_that_has_not_finished_reports_its_clocks_as_they_stand():
    # b waits in TAR with a target above its clock; the summary reports the clock.
    ca = ClockAuthority(_two_way(0.5))
    ca.on_request(A, TAR, 0.4, [])
    ca.on_request(B, TAR, 5.0, [])
    assert RunSummary.of(ca, 0.1).final_clocks == (("a", 0.4), ("b", 0.0))


def test_the_speed_result_is_simulated_seconds_over_wall_seconds():
    ca = ClockAuthority(_two_way(0.5))
    ca.on_request(A, TAR, 0.4, [])
    fast = RunSummary.of(ca, 0.04).cost_record()
    slow = RunSummary.of(ca, 0.2).cost_record()
    assert fast["simulated_seconds"] == 0.4
    assert fast["speed_ratio"] == pytest.approx(10.0)
    assert fast["meets_speed_target"] is True and fast["speed_refused"] is None
    assert slow["speed_ratio"] == pytest.approx(2.0)
    assert slow["meets_speed_target"] is False
    assert fast["speed_target_ratio"] == SPEED_TARGET_RATIO == 5.0
    assert fast["varies_with"] == ["host", "wall-clock interleaving"]


def test_a_run_that_spent_no_wall_time_has_no_speed_result():
    record = RunSummary.of(ClockAuthority(_two_way(0.5)), 0.0).cost_record()
    assert record["speed_ratio"] is None
    assert record["meets_speed_target"] is None
    assert record["speed_refused"] == (
        "no wall time was measured, so this run has no speed result"
    )


def test_a_duration_that_is_not_a_finite_number_is_refused_where_it_enters():
    ca = ClockAuthority(_two_way(0.5))
    for bad in (INF, math.nan, -1.0):
        with pytest.raises(ValueError, match="wall_seconds must be a finite"):
            RunSummary.of(ca, bad)
        with pytest.raises(ValueError, match="lazy_trace_wall_seconds must be"):
            RunSummary.of(ca, 0.1, lazy_trace_wall_seconds=bad)
        with pytest.raises(ValueError, match="^refused_predicted_seconds must be"):
            RefusalTally.of([], 1, refused_predicted_seconds=bad)
        with pytest.raises(ValueError, match="^predicted_seconds must be"):
            RefusalTally.of([], 1, predicted_seconds=bad)


def test_the_by_value_fields_land_in_their_halves():
    summary = RunSummary.of(
        ClockAuthority(_two_way(0.5)),
        0.1,
        lazy_traces=7,
        lazy_trace_wall_seconds=1.5,
        diagnostics=2,
        detectors=DetectorState(stragglers=0, clock_lint="passed"),
    )
    cost, schedule = summary.cost_record(), summary.schedule_record()
    assert (cost["lazy_traces"], cost["lazy_trace_wall_seconds"]) == (7, 1.5)
    assert cost["diagnostics"] == 2
    assert (schedule["stragglers"], schedule["clock_lint"]) == (0, "passed")


def test_refusals_tally_under_their_source_and_the_fractions_count_cost_only():
    tally = RefusalTally.of(
        [
            "cost:no price for this shape",
            "executor:ThreadPoolExecutor.submit",
            "command:start_profiler",
            "cost:outside the measured hull",
            "cost:no price for this shape",
            "command:start_profiler",
        ],
        steps=50,
        refused_predicted_seconds=1.5,
        predicted_seconds=30.0,
    )
    record = tally.record()
    assert record["count"] == 6
    assert record["by_source"] == [["command", 2], ["cost", 3], ["executor", 1]]
    assert record["fraction_of_steps"] == pytest.approx(3 / 50)
    assert record["fraction_of_predicted_seconds"] == pytest.approx(0.05)
    assert record["reasons"] == [
        ["command:start_profiler", 2],
        ["cost:no price for this shape", 2],
        ["cost:outside the measured hull", 1],
        ["executor:ThreadPoolExecutor.submit", 1],
    ]


def test_refusal_reasons_come_out_in_one_order_whatever_order_they_arrived_in():
    forwards = RefusalTally.of(["x:b", "x:a", "y:c", "x:a"], steps=4).record()
    backwards = RefusalTally.of(["x:a", "y:c", "x:a", "x:b"], steps=4).record()
    assert forwards == backwards


@pytest.mark.parametrize("reason", ["no price", ":no price", "cost:"])
def test_a_refusal_that_does_not_name_its_source_is_refused(reason):
    with pytest.raises(ValueError, match="source:detail"):
        RefusalTally.of([reason], steps=1)


def test_an_empty_run_divides_by_nothing():
    record = RefusalTally().record()
    assert (record["fraction_of_steps"], record["fraction_of_predicted_seconds"]) == (
        0.0,
        0.0,
    )


def test_the_summary_survives_strict_json():
    _, _, ca = _drive(5, PICKS["name order"])
    refusals = RefusalTally.of(["cost:x", "command:y"], 9)
    for wall in (0.0, 0.25):
        record = RunSummary.of(ca, wall, refusals=refusals).as_record()
        assert sorted(record) == ["cost", "schedule"]
        # allow_nan=False: `Infinity` and `NaN` are not JSON.
        assert json.loads(json.dumps(record, allow_nan=False)) == record


# --- what the records may not contain ----------------------------------------

CITATION = re.compile(
    r"\bD\d+(\.\d+)?\b|\bT\d+\b|\bW\d+(\.\d+)?\b|\bP0\.\d+\b|\bprinciple \d+\b"
    r"|\bGate \d+\b",
    re.IGNORECASE,
)


def test_nothing_the_clock_emits_cites_a_design_document():
    _, _, ca = _drive(5, PICKS["name order"], cls=_Logged)
    emitted = [*ca.timeline.lines(), lp_table_dump(_chain())]
    emitted.append(json.dumps(RunSummary.of(ca, 0.0).as_record()))
    for refused in (
        lambda: TimelineLog(retain=False),
        lambda: TimelineLog(retain=False, sink=print).records(),
        lambda: RunSummary.of(ca, INF),
        lambda: RefusalTally.of(["x"], 1),
    ):
        with pytest.raises((TypeError, ValueError)) as raised:
            refused()
        emitted.append(str(raised.value))
    offenders = [text for text in emitted if CITATION.search(text)]
    assert not offenders, offenders[:3]
