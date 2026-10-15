# SPDX-License-Identifier: MIT
"""A record split per client run by the requests its steps hold.

Records are written by `StepRecord` on `test_parity_device_window.py`'s planted
device, each group of requests driven to completion before the next, so a
group's steps hold its requests alone; a client run names its requests by the
keys the record gives them.
"""

import json

import pytest
from test_parity_device_window import BUSY, _recorded
from test_parity_record import _config, _drive, _key, _prompt

from atom.compass.parity import ENV, read
from atom.compass.parity.served import served

SAMPLING = {"temperature": 0.0, "top_p": 1.0}


def recorded(directory, monkeypatch, groups=((0, 1), (2, 3)), key=None):
    """`directory/steps`, the record of `groups` served one after another."""
    batches = []
    for g, group in enumerate(groups):
        drive = _drive(directory / f"simulated{g}", monkeypatch, {0: list(group)})
        batches += drive[0][0]
    runner_class, device = _recorded(batches[-1])
    monkeypatch.setenv(ENV, str(directory / "steps"))
    runner = runner_class(0, _config())
    runner.host_key = key
    for batch in batches:
        runner.forward(batch)
    device.tick = BUSY // 3
    runner.exit()
    return directory / "steps"


def client_run(requests, sampling=SAMPLING):
    """A client run's requests, each named by the key the record gives it."""
    return {
        _key(_prompt(i)): {"trace": f"trace-{i}", "sampling": sampling}
        for i in requests
    }


def test_each_step_belongs_to_the_client_run_whose_requests_it_holds(
    tmp_path, monkeypatch
):
    steps = recorded(tmp_path, monkeypatch)
    runs = [client_run([0, 1]), client_run([2, 3])]
    split = served(steps, runs)[0]
    whose = {_key(_prompt(i)): (i // 2, f"trace-{i}") for i in range(4)}

    assert [
        {k: v for k, v in s.items() if k not in ("client_run", "traces", "sampling")}
        for s in split
    ] == read(steps)[0]
    assert {s["client_run"] for s in split} == {0, 1}
    for step in split:
        held = [whose[row[0]] for row in step["batch"]]
        assert {k for k, _ in held} == {step["client_run"]}
        assert step["traces"] == sorted({t for _, t in held})
        assert step["sampling"] == [json.dumps(SAMPLING, sort_keys=True)]


def test_a_step_holding_no_client_runs_requests_belongs_to_none(tmp_path, monkeypatch):
    split = served(recorded(tmp_path, monkeypatch), [client_run([0, 1])])[0]
    later = [s for s in split if s["client_run"] is None]

    assert later and all(s["traces"] == s["sampling"] == [] for s in later)
    assert all(s["client_run"] == 0 for s in split[: len(split) - len(later)])


def test_a_step_holding_two_client_runs_requests_refuses_the_split(
    tmp_path, monkeypatch
):
    steps = recorded(tmp_path, monkeypatch)
    runs = [client_run([0]), client_run([1]), client_run([2, 3])]

    with pytest.raises(ValueError, match=r"holds requests of client runs \[0, 1\]"):
        served(steps, runs)


def test_a_client_run_s_export_splits_as_its_requests_do(tmp_path, monkeypatch):
    steps = recorded(tmp_path, monkeypatch)
    runs = [client_run([0, 1]), client_run([2, 3])]
    exports = []
    for k, run in enumerate(runs):
        exports.append(tmp_path / f"client-{k}.json")
        exports[-1].write_text(json.dumps({"requests": run}))

    assert served(steps, exports) == served(steps, runs)


def test_a_request_named_by_two_client_runs_refuses(tmp_path, monkeypatch):
    steps = recorded(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="is in client runs 0 and 1"):
        served(steps, [client_run([0]), client_run([0, 1])])
