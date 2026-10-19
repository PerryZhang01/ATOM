# SPDX-License-Identifier: MIT
"""`compass fit host`, run through `cli.main` on `test_fit_steps.py`'s record
directories, and the host laws it publishes handed to `run.runner` in a run
file on `test_run_step_grid.py`'s deployment."""

import json
import re

import pytest
from test_fit_steps import (  # noqa: F401
    DIGEST,
    clients,
    fit,
    member,
    planted_roots,
    record,
    served,
)
from test_memory_readings import META_8B
from test_run_step_grid import deployment, publish, run_file

from atom.compass import run as compass_run
from atom.compass.artifacts import Key, Kind
from atom.compass.backends.step_labels import digest
from atom.compass.backends.step_linear_fit import fit_host, label_records


def host_laws_key(key) -> Key:
    return Key.of(
        Kind.HOST_LAWS,
        host_cpu=key["host_cpu"],
        atom_src=key["atom"],
        model=key["model"],
        engine_config=json.dumps(key["engine"], sort_keys=True),
        machine_spec=DIGEST,
    )


def test_the_host_laws_published_are_fit_hosts_and_a_run_file_carrying_them_runs(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = compass_run.host_key(r.config)
    a = record(tmp_path / "a", 19, key=key, trace="trace-a")
    b = record(tmp_path / "b", 20, key=key, trace="trace-b")
    fit(tmp_path, "fit host", a, b, *clients(a, b))

    host = member(tmp_path / "store", host_laws_key(key), "params")
    # In run digest order, as the command reads them: a least-squares fit's
    # last bits depend on the order of its samples.
    paths = sorted((d / "steps/dp0.jsonl" for d in (a, b)), key=digest)
    laws, report = fit_host(label_records(paths)[0])
    assert host == {**laws, "key": {**laws["key"], "machine_spec": DIGEST}}
    assert member(tmp_path / "store", host_laws_key(key), "report") == {
        "provenance": served(key, {a: "trace-a", b: "trace-b"}),
        **report,
    }
    run = run_file(tmp_path, monkeypatch, host=host)
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    assert r.compass_host.laws == host["laws"]


def test_a_record_without_its_host_key_refuses(tmp_path):
    a = record(tmp_path / "a", 19, keyed=False)

    named = f"{digest(a / 'steps/dp0.jsonl')} holds steps with no host labels under a host key"
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit host", a, *clients(a))
    assert not (tmp_path / "store").exists()
