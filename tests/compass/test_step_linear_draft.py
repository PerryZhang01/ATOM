# SPDX-License-Identifier: MIT
"""A step linear law fitted on an MTP record prices each forward's draft apart,
as terms tagged `draft`, and a speculative run charges them."""

import json

import pytest
from test_backend_step_linear_fit import _fit, _record, _steps, _view
from test_dp_step_max import prefill, runner
from test_speculative_step import _speculative

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
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
    (tmp_path / "steps").mkdir()
    law = _fit([_mtp_record(tmp_path / "steps" / "dp0.jsonl", steps)])
    assert "draft" not in _fit([_record(tmp_path / "plain.jsonl", steps)])

    backend = StepLinearBackend(StepLinearLaw(law))
    for rows, rung, seconds in steps[:-1]:
        cost = backend.price(_view(rows, rung))
        assert cost.target_seconds == pytest.approx(seconds, rel=1e-5)
        assert cost.draft_seconds == pytest.approx(SHARE * seconds, rel=1e-5)
        drafts = [t.name for t in cost.terms if t.draft]
        assert drafts and all(n.startswith("draft.") for n in drafts)

    middle = BatchView((RequestShape(512, 512, False),), produces_output=False)
    cost = backend.price(middle)
    assert [(t.name, t.seconds) for t in cost.terms if t.draft] == [("draft", 0.0)]
    assert cost.target_seconds > 0

    # Cost accuracy holds the target price to the target label.
    for kind in ("prefill", "decode"):
        summary = cost_accuracy(tmp_path / "steps", law)["unvalidated"][kind]
        assert summary["on_totals"] == pytest.approx(0.0, abs=1e-5)


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
