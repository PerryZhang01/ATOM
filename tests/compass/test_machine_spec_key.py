# SPDX-License-Identifier: MIT
"""Price lists, region terms, step linear laws and host laws are keyed by the
machine-spec digest: the digest of the spec's device, software stack and PTL
state.

The checked-in PTL-off law loads under hjbog-srdc-39's machine file and refuses
under node 18's, and under one that differs from hjbog-srdc-39's in its PTL
state alone, naming both digests and this run's key. Records whose cards read
another PTL state than the spec states refuse at the fit, naming it, and so do
step grid bundles and step measurements at load.
"""

import copy
import json
import re
from pathlib import Path

import pytest
from test_artifact_invalidation import (
    BASE,
    BEFORE_HOST_CPU,
    GATES,
    KEY_OF,
    LAYOUT,
    publish,
)
from test_backend_step_linear import _installed
from test_backend_step_linear_fit import _steps, _timed_record
from test_fit_samples import edit
from test_fit_steps import clients, planted_roots, record  # noqa: F401
from test_measure_memory import a_start
from test_memory_readings import DOCUMENT, META_8B, STACK

from atom.compass import cli
from atom.compass.artifacts import (
    KEY_FIELDS,
    ArtifactRefusal,
    ArtifactStore,
    Axis,
    Key,
    Kind,
    Row,
    machine_spec_digest,
)
from atom.compass.backends import StepLinearBackend
from atom.compass.drive import identity, measure_memory
from atom.compass.drive.gpu import GpuRefusal
from atom.compass.spec import MachineSpec

ROOT = Path(__file__).resolve().parents[2] / "scripts" / "compass"
LAW = json.loads(
    (ROOT / "laws" / "qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json").read_text()
)
N18 = json.loads((ROOT / "machines" / "mi308x-hjbog18.json").read_text())
N39 = json.loads((ROOT / "machines" / "mi308x-hjbog39.json").read_text())
#: The law at the test deployment's engine config, which `_installed` states.
HERE = {k: v for k, v in LAW.items() if k != "engine_config"}


def digest(document: dict) -> str:
    return machine_spec_digest(MachineSpec.from_mapping(document))


def test_the_hjbog39_law_loads_under_its_own_spec_and_refuses_under_node_18s(
    monkeypatch, tmp_path
):
    assert LAW["machine_spec"] == LAW["host"]["key"]["machine_spec"] == digest(N39)
    assert LAW["engine_config"] == LAW["host"]["key"]["engine"]
    installed = _installed(monkeypatch, tmp_path, step_linear=HERE, machine=N39)
    assert isinstance(installed, StepLinearBackend)

    with pytest.raises(ValueError) as refused:
        _installed(monkeypatch, tmp_path, step_linear=HERE, machine=N18)
    assert f"fitted on machine spec {digest(N39)}" in str(refused.value)
    assert f"this run's is {digest(N18)}" in str(refused.value)
    assert "ptl='enabled'" in str(refused.value)


#: The spec each checked-in law was fitted under; a law missing here fails below.
LAW_SPECS = {
    "qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json": "mi308x-hjbog39.json",
    "deepseek-v4-pro-dpa8-mi355x-gpu45-prefill.json": "mi355x-gpu45.json",
    "deepseek-v4-pro-dpa8-mi355x-gpu45-decode.json": "mi355x-gpu45.json",
}


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "laws").glob("*.json")))
def test_every_checked_in_law_stamps_the_digest_of_its_spec(name):
    law = json.loads((ROOT / "laws" / name).read_text())
    spec = json.loads((ROOT / "machines" / LAW_SPECS[name]).read_text())
    assert law["machine_spec"] == law["host"]["key"]["machine_spec"] == digest(spec)


def test_the_ptl_off_law_refuses_on_cards_of_its_stack_with_ptl_on(
    monkeypatch, tmp_path
):
    on = edited("device.ptl", "enabled")
    with pytest.raises(ValueError) as refused:
        _installed(monkeypatch, tmp_path, step_linear=HERE, machine=on)
    assert f"fitted on machine spec {digest(N39)}" in str(refused.value)
    assert f"this run's is {digest(on)}" in str(refused.value)
    assert "ptl='enabled'" in str(refused.value)


def with_ptl(tmp_path, ptl: str) -> Path:
    """A spec file of `DOCUMENT` stating `ptl`."""
    document = copy.deepcopy(DOCUMENT)
    document["device"]["ptl"] = ptl
    path = tmp_path / f"spec-{ptl}.json"
    path.write_text(json.dumps(document))
    return path


def fit(tmp_path, a, spec: Path, *flags):
    store, out = tmp_path / "store", tmp_path / "out"
    tail = ["--machine", spec, "--store", store, "--out", out, *flags]
    argv = ["fit", "steps", "--backend", "step-linear", a, *clients(a), *tail]
    cli.main(list(map(str, argv)))
    return json.loads((out / "step_linear_law.json").read_text())


@pytest.mark.parametrize(
    "ran, stated",
    [("enabled", "disabled"), ("disabled", "enabled"), (None, "disabled")],
    ids=["on-under-off", "off-under-on", "unrecorded"],
)
def test_records_fit_only_under_a_spec_of_their_cards_ptl(tmp_path, ran, stated):
    a = record(tmp_path / "a", 19)
    edit(a / "run.json", lambda r: r.update(ptl=ran) if ran else r.pop("ptl"))
    told = f"ptl {ran or 'unrecorded'!r} in the records and {stated!r} in"
    with pytest.raises(SystemExit, match=re.escape(told)):
        fit(tmp_path, a, with_ptl(tmp_path, stated))
    assert not (tmp_path / "store").exists()

    if ran:
        law = fit(tmp_path, a, with_ptl(tmp_path, ran))
        spec = json.loads(with_ptl(tmp_path, ran).read_text())
        assert law["key"]["machine_spec"] == digest(spec)


def test_ptl_states_the_state_of_records_that_carry_none_and_the_report_says_so(
    tmp_path,
):
    a = record(tmp_path / "a", 19)
    edit(a / "run.json", lambda r: r.pop("ptl"))
    law = fit(tmp_path, a, with_ptl(tmp_path, "enabled"), "--ptl", "enabled")
    assert law["report"]["stated_ptl"] == "enabled"
    spec = json.loads(with_ptl(tmp_path, "enabled").read_text())
    assert law["key"]["machine_spec"] == digest(spec)


@pytest.mark.parametrize("carried", ["enabled", "disabled"])
def test_ptl_refuses_for_records_that_carry_a_ptl_state(tmp_path, carried):
    a = record(tmp_path / "a", 19)
    edit(a / "run.json", lambda r: r.update(ptl=carried))
    told = (
        "--ptl enabled states the PTL state of records that carry none, "
        f"and these carry {carried!r}"
    )
    with pytest.raises(SystemExit, match=re.escape(told)):
        fit(tmp_path, a, with_ptl(tmp_path, "enabled"), "--ptl", "enabled")
    assert not (tmp_path / "store").exists()


def test_a_bundle_measured_with_ptl_on_refuses_where_it_is_off(tmp_path):
    store, key = ArtifactStore(tmp_path), KEY_OF[Row.STEP_GRID_BUNDLE]
    publish(store, key, conditions=BASE.with_reading(Axis.PTL, "enabled"))
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(key, conditions=BASE, gates=GATES)
    assert "`step grid bundle` x `PTL`: recorded 'enabled', this run reads " + (
        "'disabled'"
    ) in str(refused.value)
    on = BASE.with_reading(Axis.PTL, "enabled")
    assert store.load(key, conditions=on, gates=GATES).key == key


@pytest.mark.parametrize(
    "row",
    [Row.PRICE_LIST, Row.REGION_TERMS, Row.MEMORY_READINGS, Row.STEP_GRID_BUNDLE],
    ids=lambda row: row.name,
)
def test_an_entry_published_before_the_ptl_axis_refuses_naming_it(tmp_path, row):
    store = ArtifactStore(tmp_path)
    entry = publish(store, KEY_OF[row])
    path = entry.directory / "entry.json"
    document = json.loads(path.read_bytes())
    for field, recorded in document["fingerprints"].items():
        recorded["cells"] = {
            name: BASE.reading(Axis[name.upper()]).as_json()
            for name in BEFORE_HOST_CPU[field]
        }
    path.chmod(0o644)
    path.write_text(json.dumps(document))

    with pytest.raises(ArtifactRefusal, match="now rows it over .*device, PTL"):
        store.load(entry.key, conditions=BASE, gates=GATES)


def test_cards_read_ptl_under_their_pci_device_and_none_reads_disabled(
    monkeypatch, tmp_path
):
    buses = {"card0": "0000:0A:00.0", "card1": "0000:80:00.0", "card2": "0001:0B:00.0"}
    listing = {card: {"PCI Bus": bus} for card, bus in buses.items()}
    monkeypatch.setattr(identity, "smi", lambda *args: json.dumps(listing))
    monkeypatch.setattr(identity, "PCI", str(tmp_path))
    for card in ("card0", "card2"):
        (tmp_path / buses[card].lower() / "ptl").mkdir(parents=True)
        (tmp_path / buses[card].lower() / "ptl" / "ptl_enable").write_text("enabled\n")

    assert identity._ptl((0, 2)) == "enabled"
    assert identity._ptl((1,)) == "disabled"
    with pytest.raises(GpuRefusal) as refused:
        identity._ptl((2, 1, 0))
    assert str(refused.value) == (
        "the cards of this run disagree on PTL: GPU 2 enabled, GPU 1 disabled, "
        "GPU 0 enabled"
    )


def test_a_memory_start_on_cards_of_another_ptl_refuses_before_it_starts(
    tmp_path, monkeypatch
):
    _, argv = a_start(tmp_path, monkeypatch, STACK)
    monkeypatch.setattr(measure_memory, "_ptl", lambda gpus: "enabled")
    with pytest.raises(ValueError) as refused:
        measure_memory.main(argv)
    assert str(refused.value) == (
        f"the cards read ptl 'enabled', and {tmp_path / 'machine.json'} states "
        "'disabled'; name the machine spec of the cards' PTL state"
    )
    assert not (tmp_path / "out").exists()


def test_a_law_that_states_no_machine_spec_refuses(monkeypatch, tmp_path):
    unkeyed = {k: v for k, v in HERE.items() if k != "machine_spec"}
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
        ("device.ptl", "enabled"),
    ],
)
def test_a_device_or_stack_edit_moves_the_key(path, value):
    assert digest(edited(path, value)) != digest(N39)


def test_a_spec_that_states_no_ptl_is_keyed_as_ptl_disabled():
    unstated = copy.deepcopy(N39)
    assert unstated["device"].pop("ptl") == "disabled"
    assert digest(unstated) == digest(N39)
