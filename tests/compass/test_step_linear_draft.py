# SPDX-License-Identifier: MIT
"""A step linear law fitted on an MTP record prices each forward's draft apart,
as terms tagged `draft`, and a speculative run charges them."""

import json

import pytest
from test_backend_step_linear_fit import _fit, _record, _steps, _view
from test_dp_step_max import prefill, runner
from test_speculative_step import _speculative
from test_validate_cost import LAW, _two_ranks
from test_validate_cost import _record as _rank_record

from atom.compass.backends import (
    BatchView,
    CostTerm,
    RequestShape,
    StepCost,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_labels import label
from atom.compass.backends.step_linear_fit import fit, label_records, mark_held_out
from atom.compass.validate.cost import cost_accuracy

#: Each forward's draft, as a share of its target.
SHARE = 0.1


def _mtp_record(path, steps):
    """`steps` as an MTP record: each forward's target its price, then a draft of
    `SHARE` of it."""
    _record(path, [(rows, rung, s * (1 + SHARE)) for rows, rung, s in steps])
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line, (_, _, s) in zip(lines, steps):
        line["d_target_ns"] = round(s * 1e9)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


def test_an_mtp_record_fits_a_draft_law_that_prices_each_forward_apart(tmp_path):
    steps = _steps(3)
    law = _fit([_mtp_record(tmp_path / "dp0.jsonl", steps)])
    assert "draft" not in _fit([_record(tmp_path / "plain.jsonl", steps)])

    backend = StepLinearBackend(StepLinearLaw(law))
    for rows, rung, seconds in steps[:-1]:
        cost = backend.price(_view(rows, rung))
        assert cost.target_seconds == pytest.approx(seconds, rel=1e-5)
        assert cost.draft_seconds == pytest.approx(SHARE * seconds, rel=1e-5)
        drafts = [t.name for t in cost.terms if t.draft]
        assert drafts and all(n.startswith("draft.") for n in drafts)

    # A middle chunk's draft is its draft-KV pass, priced by the draft vector.
    row = (RequestShape(512, 512, False),)
    cost = backend.price(BatchView(row, produces_output=False))
    final = backend.price(BatchView(row, produces_output=True))
    assert cost.draft_seconds == final.draft_seconds > 0
    assert cost.target_seconds > 0


def test_an_mtp_record_s_middle_chunk_without_a_target_end_is_cut_unsplit(tmp_path):
    """Its device label holds its draft-KV pass, which the draft law prices
    again: a record from before the stamp, or a pass nothing stamps."""
    record = _mtp_record(tmp_path / "dp0.jsonl", _steps(3))
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    lines[0]["produces_output"] = False
    del lines[0]["d_target_ns"]
    record.write_text("".join(json.dumps(x) + "\n" for x in lines))

    samples = label(record)[0]
    assert [(s.step, s.cut, s.device_s) for s in samples if s.cut] == [
        (0, "unsplit", None)
    ]
    assert _fit([record])["cut"]["unsplit"] == 1


def test_a_speculative_run_charges_the_law_s_draft_and_refuses_a_law_without_one(
    tmp_path,
):
    steps = _steps(3)
    law = _fit([_mtp_record(tmp_path / "dp0.jsonl", steps)])
    priced = StepLinearBackend(StepLinearLaw(law)).price(_view([(8, 8, False)]))
    assert priced.draft_seconds > 0
    tagged = _speculative(runner(StepLinearBackend(StepLinearLaw(law))))
    assert tagged.forward(prefill(8)).predicted_s == pytest.approx(priced.seconds)
    assert tagged.refused_commands() == ()

    del law["draft"]
    plain = _speculative(runner(StepLinearBackend(StepLinearLaw(law))))
    assert plain.forward(prefill(8)).predicted_s == 0.0
    assert plain.refused_commands() == ("cost:StepLinearBackend: prices no draft",)


def test_cost_accuracy_grades_the_draft_apart_and_unvalidated(tmp_path):
    """The draft law records no held-out errors: its terms leave a step's
    validation and price to the target's, and its own grade is unvalidated."""
    records = [_mtp_record(tmp_path / f"dp0-{s}.jsonl", _steps(s)) for s in (3, 4)]
    samples = label_records(records)[0]
    law, report = fit(samples)
    mark_held_out(law, report, samples)
    (tmp_path / "steps").mkdir()
    _mtp_record(tmp_path / "steps" / "dp0.jsonl", _steps(5))
    report = cost_accuracy(tmp_path / "steps", law)
    for kind in ("prefill", "decode"):
        assert report["validated"][kind]["on_totals"] == pytest.approx(0.0, abs=1e-5)
    assert "draft" not in report["validated"]
    draft = report["unvalidated"]["draft"]
    assert draft["on_totals"] == pytest.approx(0.0, abs=1e-5)
    targets = sum(
        s["real_s"]
        for group in ("validated", "unvalidated")
        for kind, s in report[group].items()
        if kind != "draft"
    )
    assert draft["real_s"] == pytest.approx(SHARE * targets, rel=1e-5)

    del law["draft"]  # a law that prices no draft grades it as 0 s, unvalidated
    report = cost_accuracy(tmp_path / "steps", law)
    assert "draft" not in report["validated"]
    assert report["unvalidated"]["draft"]["on_totals"] == -1.0


class _DraftDearestWhenCheapest:
    """`LAW`'s prices plus a draft that is dearest on the cheapest target."""

    def __init__(self):
        self.law = StepLinearBackend(StepLinearLaw(LAW))

    def price(self, view):
        cost = self.law.price(view)
        made = cost.terms[0].provenance
        draft = CostTerm("draft", 1.0 / cost.target_seconds, made, draft=True)
        return StepCost([*cost.terms, draft], cost.treatments)

    def describe(self):
        return "planted, with a draft"


def test_a_group_step_prices_as_its_rank_of_dearest_target_not_of_dearest_step(
    tmp_path,
):
    ranks = _two_ranks()
    for r, steps in enumerate(ranks):
        _rank_record(tmp_path / "steps", steps, rank=r, group=ranks)
    report = cost_accuracy(tmp_path / "steps", _DraftDearestWhenCheapest())
    for kind in ("prefill", "decode"):
        s = report["validated"][kind]
        assert s["steps"] == 10
        assert s["on_totals"] == pytest.approx(0.0, abs=1e-9)
