# SPDX-License-Identifier: MIT
"""The fields the attainable-peak roofline reads: the bandwidth curve, the
kernel floor and one compute derate per format.

A curve that is not monotone in bytes, a single compute derate and an FP4
derate without its rate (or the reverse) are refused by name; the shipped
machine files read with every new field and its origin.
"""

import json
import pathlib

import pytest
from test_spec_schema import document, refusal_from

from atom.compass.spec import MachineSpec, explain, validate
from atom.compass.spec.rules import Rule

MACHINES = pathlib.Path(__file__).resolve().parents[2] / "scripts/compass/machines"
CURVE = [[4096, 1.1e11], [1048576, 3.0e12], [4294967296, 6.4e12]]


def roofline_document():
    """The reference spec with every new field, FP4 included."""
    written = document()
    written["device"]["memory"]["bandwidth_curve"] = CURVE
    written["device"]["kernel_floor_s"] = 2.0e-6
    written["device"]["compute"]["fp4_flops"] = 1.0e16
    written["device"]["compute"]["derate"]["fp4"] = 0.60
    return written


def test_a_spec_with_the_new_fields_validates_and_echoes_them():
    written = roofline_document()
    assert validate(written).ok
    spec = MachineSpec.from_mapping(written)
    assert spec.value("device.memory.bandwidth_curve") == tuple(CURVE)
    assert spec.value("device.kernel_floor_s") == 2.0e-6
    for fmt, derate in {"bf16": 0.70, "fp8": 0.70, "fp4": 0.60}.items():
        assert spec.value(f"device.compute.derate.{fmt}") == derate
    assert spec.echo() == written


def test_each_compute_rate_is_explained_with_its_own_formats_derate():
    spec = MachineSpec.from_mapping(roofline_document())
    rows = {row.path: row for row in explain(spec, "device.compute").contributions}
    assert rows["device.compute.fp4_flops"].derate == 0.60
    assert rows["device.compute.fp4_flops"].effective == 1.0e16 * 0.60
    assert rows["device.compute.bf16_flops"].derate == 0.70


@pytest.mark.parametrize(
    "curve",
    [
        [[4096, 2.0e11], [8192, 1.0e11]],
        [[8192, 1.0e11], [4096, 2.0e11]],
        [[4096, 1.0e11], [4096, 2.0e11]],
    ],
    ids=["falling-rate", "falling-bytes", "repeated-bytes"],
)
def test_a_curve_not_monotone_in_bytes_is_refused_by_name(curve):
    written = document()
    written["device"]["memory"]["bandwidth_curve"] = curve
    refusal = refusal_from(written)
    assert refusal.rule is Rule.SHAPE
    assert "`device.memory.bandwidth_curve` goes from" in refusal.what
    (size, rate), (more, then) = curve
    assert f"from {rate!r} B/s at {size} B to {then!r} B/s at {more} B" in refusal.what
    assert "not monotone in bytes" in refusal.what
    assert [r.rule for r in validate(written).refusals] == [Rule.SHAPE]


@pytest.mark.parametrize(
    "curve", [[], [[4096]], [[4096, 0]], [[0, 1.0e11]], "4096:1e11"]
)
def test_a_curve_that_is_not_points_is_refused(curve):
    written = document()
    written["device"]["memory"]["bandwidth_curve"] = curve
    refusal = refusal_from(written)
    assert refusal.rule is Rule.SHAPE
    assert "`device.memory.bandwidth_curve" in refusal.what


def test_a_single_compute_derate_is_refused_by_name():
    written = document()
    written["device"]["compute"]["derate"] = 0.70
    refusal = refusal_from(written)
    assert refusal.rule is Rule.SHAPE
    assert "`device.compute.derate` holds 0.7" in refusal.what
    assert not validate(written).ok


def test_a_compute_rate_without_its_formats_derate_is_refused():
    written = roofline_document()
    del written["device"]["compute"]["derate"]["fp4"]
    refusal = refusal_from(written)
    assert refusal.rule is Rule.DERATE
    assert refusal.what.startswith("`device.compute.derate.fp4` is missing")
    assert [r.rule for r in validate(written).refusals] == [Rule.DERATE]


def test_an_fp4_derate_without_an_fp4_rate_is_refused():
    written = document()
    written["device"]["compute"]["derate"]["fp4"] = 0.60
    refusal = refusal_from(written)
    assert refusal.rule is Rule.DERATE
    assert "`device.compute.fp4_flops`, the peak it derates, is not" in refusal.what
    assert [r.rule for r in validate(written).refusals] == [Rule.DERATE]


@pytest.mark.parametrize("machine", sorted(p.stem for p in MACHINES.glob("*.json")))
def test_each_machine_file_reads_the_new_fields_with_their_origins(machine):
    spec = MachineSpec.from_mapping(
        json.loads((MACHINES / f"{machine}.json").read_text())
    )
    formats = ["bf16", "fp8"]
    if "device.compute.fp4_flops" in spec.values:
        formats.append("fp4")
    paths = [f"device.compute.derate.{fmt}" for fmt in formats] + [
        "device.memory.bandwidth_curve",
        "device.kernel_floor_s",
    ]
    # Node 18's are probe_device.py's readings; the others are still assumed.
    origin = "probed" if machine == "mi308x-hjbog18" else "assumed"
    for path in paths:
        assert spec.value(path)
        assert spec.origin(path)["origin"] == origin
