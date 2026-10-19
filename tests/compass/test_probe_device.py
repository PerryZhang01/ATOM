"""The attainable-peak probe: the cards it refuses, and the fragment its
samples make, merged into a machine spec by that machine's script."""

import importlib.util
import json
from pathlib import Path

import pytest

from atom.compass.spec import Fragment, MachineSpec, SpecRefusal
from atom.compass.spec.rules import Rule

TREE = Path(__file__).resolve().parents[2]
SCRIPTS = TREE / "scripts" / "compass"


def _load(name):
    loaded = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    return module


probe = _load("probe_device")

#: Samples as a probe writes them: GEMM seconds per shape, read, write and copy
#: seconds per size moved, copy the fastest and its rate dipping on the
#: plateau, and the banded fields' samples.
RECORD = {
    "gemm": {
        "bf16": {"4096x4096x4096": [1e-3] * 3, "8192x4096x8192": [2e-3] * 3},
        "fp8": {"8192x8192x8192": [2e-3]},
    },
    "read": {"4096": [8e-6], "8192": [8e-6], "16384": [16.4e-6]},
    "write": {"4096": [8e-6], "8192": [8e-6], "16384": [16.4e-6]},
    "copy": {"4096": [4e-6], "8192": [4e-6], "16384": [8.2e-6]},
    "link": [3.2e10, 3.0e10, 3.4e10],
    "floor": [2e-6, 1e-6, 3e-6, 4e-6, 5e-6],
    "one_hop": [1e-6] * 5,
}


def test_readings_take_the_best_shape_and_keep_the_curve_monotone():
    found = probe.readings(RECORD)
    assert found["bf16"] == 2 * 8192 * 4096 * 8192 / 2e-3  # the second shape is best
    assert [r for _, r in found["bandwidth_curve"]] == [1.024e9, 2.048e9, 2.048e9]
    assert found["link"] == 3.2e10
    assert found["device.kernel_floor_s"][0] == 3e-6


def test_the_curve_is_at_least_every_kind_at_every_size():
    record = RECORD | {
        "read": {"4096": [2e-6], "8192": [8e-6], "16384": [8e-6]},
        "write": {"4096": [8e-6], "8192": [2e-6], "16384": [5e-6]},
    }
    curve = probe.readings(record)["bandwidth_curve"]
    # read is the fastest at 4 KiB, write at 8 KiB; 16 KiB keeps 8 KiB's rate
    assert curve == [
        [4096, pytest.approx(2.048e9)],
        [8192, pytest.approx(4.096e9)],
        [16384, pytest.approx(4.096e9)],
    ]
    for kind in probe.rates(record).values():
        assert all(rate >= kind[size] for size, rate in curve)


def test_the_node18_script_publishes_the_probed_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPASS_MACHINE", "mi308x-hjbog18")
    script = _load("mi308x_spec")
    found = probe.readings(RECORD)
    monkeypatch.setattr(script, "PEAKS", ("2026-10-19", "abc", found, "f00d"))
    document = SCRIPTS / "machines" / "mi308x-hjbog18.json"
    out = tmp_path / "spec.json"
    assert script.main(["--sweep-from", str(document), "--out", str(out)]) == 0
    spec = MachineSpec.from_mapping(json.loads(out.read_text()))
    assert spec.value("device.compute.derate.fp8") == 2 * 8192**3 / 2e-3 / 6.88e14
    assert spec.value("device.memory.derate") == 2.048e9 / 5.3e12
    assert spec.value("interconnect.intra_node.derate") == 3.2e10 / 6.4e10
    floor = spec.origin("device.kernel_floor_s")
    assert floor["origin"] == "probed" and floor["record"] == "sha256:f00d"
    assert (floor["p10"], floor["p90"]) == pytest.approx((1.4e-6, 4.6e-6))


def test_a_rate_above_its_peak_is_refused():
    peaks = {
        "device.compute.bf16_flops": 1.0,
        "device.compute.fp8_flops": 1e30,
        "device.memory.bandwidth_bytes_per_s": 1e30,
        "interconnect.intra_node.link_bandwidth_bytes_per_s": 1e30,
    }
    found = probe.readings(RECORD)
    document = probe.fragment("m", found, peaks, "a", "2026-10-19", {"record": "r"})
    with pytest.raises(SpecRefusal) as refused:
        Fragment.from_mapping(document, "probe")
    assert refused.value.rule is Rule.DERATE
    assert "device.compute.derate.bf16" in str(refused.value)


def _card(drm, name, bdf, used, ptl=None, busy=0):
    device = drm / "devices" / bdf
    device.mkdir(parents=True)
    (device / "mem_info_vram_used").write_text(f"{used}\n")
    (device / "gpu_busy_percent").write_text(f"{busy}\n")
    if ptl:
        (device / "ptl").mkdir()
        (device / "ptl" / "ptl_enable").write_text(f"{ptl}\n")
    (drm / name).mkdir()
    (drm / name / "device").symlink_to(device)


def test_cards_disagreeing_with_the_spec_ptl_holding_vram_or_busy_are_refused(tmp_path):
    _card(tmp_path, "card1", "0000:0a:00.0", 3 * 10**8, "enabled")
    _card(tmp_path, "card9", "0000:80:00.0", 3 * 10**8)
    _card(tmp_path, "card17", "0000:a4:00.0", 9 * 10**9, "enabled")
    _card(tmp_path, "card25", "0000:c8:00.0", 3 * 10**8, "enabled", busy=37)
    read = probe.cards(
        ["0000:0a:00.0", "0000:80:00.0", "0000:a4:00.0", "0000:c8:00.0"], tmp_path
    )
    assert [c["ptl"] for c in read] == ["enabled", "disabled", "enabled", "enabled"]
    probe.refuse_unfit(read[:1], "enabled")
    with pytest.raises(SpecRefusal) as refused:
        probe.refuse_unfit(read, "enabled")
    assert "0000:0a" not in str(refused.value)
    assert "0000:80:00.0 (ptl disabled" in str(refused.value)
    assert "0000:a4:00.0 (ptl enabled, 9000000000 B used" in str(refused.value)
    assert "0000:c8:00.0 (ptl enabled, 300000000 B used, 37% busy)" in str(
        refused.value
    )
