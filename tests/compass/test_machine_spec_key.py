# SPDX-License-Identifier: MIT
"""Price lists, region terms, step linear laws and host laws are keyed by the
machine-spec digest: the digest of the spec's device and software stack.

The checked-in hjbog-srdc-39 law loads under its own machine
file and refuses under node 18's, naming both digests.
"""

import copy
import json
from pathlib import Path

import pytest
from test_artifact_invalidation import LAYOUT, publish
from test_backend_step_linear import _installed
from test_backend_step_linear_fit import _steps, _timed_record
from test_memory_readings import META_8B

from atom.compass.artifacts import (
    KEY_FIELDS,
    ArtifactRefusal,
    ArtifactStore,
    Key,
    Kind,
    machine_spec_digest,
)
from atom.compass.backends import StepLinearBackend
from atom.compass.spec import MachineSpec

ROOT = Path(__file__).resolve().parents[2] / "scripts" / "compass"
LAW = json.loads(
    (ROOT / "laws" / "qwen3.8-27b-tp1-mi308x-hjbog39-661ef7661.json").read_text()
)
N18 = json.loads((ROOT / "machines" / "mi308x-hjbog18.json").read_text())
N39 = json.loads((ROOT / "machines" / "mi308x-hjbog39.json").read_text())


def digest(document: dict) -> str:
    return machine_spec_digest(MachineSpec.from_mapping(document))


def test_the_hjbog39_law_loads_under_its_own_spec_and_refuses_under_node_18s(
    monkeypatch, tmp_path
):
    assert LAW["machine_spec"] == LAW["host"]["key"]["machine_spec"] == digest(N39)
    installed = _installed(monkeypatch, tmp_path, step_linear=LAW, machine=N39)
    assert isinstance(installed, StepLinearBackend)

    with pytest.raises(ValueError) as refused:
        _installed(monkeypatch, tmp_path, step_linear=LAW, machine=N18)
    assert f"fitted on machine spec {digest(N39)}" in str(refused.value)
    assert f"this run's is {digest(N18)}" in str(refused.value)


def test_a_law_that_states_no_machine_spec_refuses(monkeypatch, tmp_path):
    unkeyed = {k: v for k, v in LAW.items() if k != "machine_spec"}
    with pytest.raises(ValueError, match="fitted on machine spec None"):
        _installed(monkeypatch, tmp_path, step_linear=unkeyed, machine=N39)


def test_the_fit_refuses_without_a_machine_spec_and_writes_no_law(tmp_path, capsys):
    from atom.compass.backends.step_linear_fit import main

    record = _timed_record(tmp_path / "dp0.jsonl", _steps(19))[0]
    with pytest.raises(SystemExit):
        main([str(record), "--out", str(tmp_path / "law.json")])
    assert "--machine-spec" in capsys.readouterr().err
    assert not (tmp_path / "law.json").exists()


def test_host_laws_from_another_spec_refuse_naming_both_digests(tmp_path, monkeypatch):
    from test_run_step_grid import deployment, run_file

    from atom.compass import run as compass_run
    from atom.compass.backends.step_linear_fit import main

    r = deployment(tmp_path, enable_dp_attention=False, enable_expert_parallel=False)
    key = compass_run.host_key(r.config)
    records = [
        str(_timed_record(tmp_path / f"dp0-{s}.jsonl", _steps(s), key=key)[0])
        for s in (19, 20)
    ]
    main(
        [
            *records,
            "--out",
            str(tmp_path / "law.json"),
            "--machine-spec",
            digest(N18),
        ]
    )
    law = json.loads((tmp_path / "law.json").read_text())
    assert law["host"]["key"]["machine_spec"] == digest(N18)

    run = run_file(tmp_path, monkeypatch, step_linear=law, host=law["host"])
    del run["step_grid"]
    ours = digest(run["machine"])
    run["step_linear"]["machine_spec"] = ours
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    with pytest.raises(ValueError) as refused:
        compass_run.runner(r, META_8B)
    assert "host laws were fitted under another ['machine_spec']" in str(refused.value)
    assert f"'machine_spec': '{ours}'" in str(refused.value)
    assert f"'machine_spec': '{digest(N18)}'" in str(refused.value)


def test_a_price_list_from_another_spec_misses_naming_both_digests(tmp_path):
    def price_list(document):
        return Key.of(
            Kind.PRICE_LIST,
            model="Qwen/Qwen3.8-27B",
            machine_spec=digest(document),
            source_root="t",
            **LAYOUT,
        )

    store = ArtifactStore(tmp_path)
    publish(store, price_list(N39))
    assert store.read(price_list(N39)).key == price_list(N39)
    with pytest.raises(ArtifactRefusal) as refused:
        store.read(price_list(N18))
    assert f"machine_spec='{digest(N18)}'" in str(refused.value)
    assert f"differs in machine_spec ('{digest(N39)}' there)" in str(refused.value)


@pytest.mark.parametrize("kind", [Kind.STEP_LINEAR_LAW, Kind.HOST_LAWS], ids=str)
def test_laws_fitted_on_another_spec_miss_naming_both_digests(tmp_path, kind):
    def law(document):
        given = {name: "x" for name in KEY_FIELDS[kind]}
        return Key.of(kind, **{**given, "machine_spec": digest(document)})

    store = ArtifactStore(tmp_path)
    publish(store, law(N39))
    assert store.read(law(N39)).key == law(N39)
    with pytest.raises(ArtifactRefusal) as refused:
        store.read(law(N18))
    assert f"machine_spec='{digest(N18)}'" in str(refused.value)
    assert f"differs in machine_spec ('{digest(N39)}' there)" in str(refused.value)


def test_region_terms_are_keyed_by_the_machine_spec_digest():
    with pytest.raises(ArtifactRefusal, match="states no machine_spec"):
        Key.of(Kind.REGION_TERMS, model="Qwen/Qwen3.8-27B", **LAYOUT)


def edited(path: str, value) -> dict:
    document = copy.deepcopy(N39)
    *blocks, leaf = path.split(".")
    node = document
    for block in blocks:
        node = node[block]
    node[leaf] = value
    return document


@pytest.mark.parametrize(
    "path, value",
    [
        ("interconnect.intra_node.link_latency_s", 3e-06),
        ("device.memory.derate", 0.8),
        ("device.memory.capacity_bytes", 1),
        ("host.cpu.cores_physical", 96),
        ("provenance.notes", "re-probed"),
    ],
)
def test_a_spec_edit_outside_device_and_stack_moves_no_key(path, value):
    assert digest(edited(path, value)) == digest(N39)


@pytest.mark.parametrize(
    "path, value",
    [
        ("device.software_pinned_to.aiter", "f4e7c7509"),
        ("device.software_pinned_to.rocm", "7.2.5"),
        ("device.arch", "gfx950"),
        ("device.name", "MI355X"),
    ],
)
def test_a_device_or_stack_edit_moves_the_key(path, value):
    assert digest(edited(path, value)) != digest(N39)
