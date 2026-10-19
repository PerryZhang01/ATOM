# SPDX-License-Identifier: MIT
"""The FP4 matrix-core rate and the scale-up fabric's width, on the MI455 spec.

The MI455 document is built the way its spec is authored: AMD's published
numbers as a `datasheet` fragment, and everything nobody published or measured
(the two link latencies, the derates and the host) as an `assumed` one.
"""

import copy

import pytest
from test_spec_origin import saved
from test_spec_schema import DOCUMENT, STACK, document, read, refusal_from
from test_spec_verbs import fragment

from atom.compass.spec import MachineSpec, SpecRefusal, merge, validate
from atom.compass.spec.rules import Rule

MI455 = "mi455x-72gpu"

#: The sourced fields of the MI455X, as the #504 spike read them.
DATASHEET = {
    "device": {
        "name": "MI455X",
        "arch": "gfx1250",
        "count_per_node": 4,
        "memory": {"capacity_bytes": 432e9, "bandwidth_bytes_per_s": 23.3e12},
        "compute": {
            "bf16_flops": 5.033e15,
            "fp8_flops": 20.133e15,
            "fp4_flops": 40.265e15,
        },
    },
    "interconnect": {
        "intra_node": {
            "topology": "switched",
            "domain_devices": 72,
            "link_bandwidth_bytes_per_s": 1.8e12,
        },
        "inter_node": {"link_bandwidth_bytes_per_s": 300e9},
    },
}

#: What no source states for it: never measured, so assumed.
ASSUMED = {
    "host": copy.deepcopy(DOCUMENT["host"]),
    "device": {
        "memory": {"derate": 0.86},
        "compute": {"derate": {"bf16": 0.50, "fp8": 0.50, "fp4": 0.50}},
        "software_pinned_to": dict(STACK),
    },
    "interconnect": {
        "intra_node": {"link_latency_s": 2.0e-6, "derate": 0.89},
        "inter_node": {"link_latency_s": 5.0e-6, "derate": 0.80},
        "router_relay_s": 1.5e-3,
    },
}

LATENCIES = (
    "interconnect.intra_node.link_latency_s",
    "interconnect.inter_node.link_latency_s",
)


def mi455():
    """The MI455 spec as a file holds it."""
    return saved(
        merge(
            [
                fragment("datasheet", DATASHEET, machine=MI455, origin="datasheet"),
                fragment("assumed", ASSUMED, machine=MI455, origin="assumed"),
            ]
        ).document
    )


def test_a_spec_with_an_fp4_rate_and_a_fabric_width_reads_and_echoes_them():
    written = document()
    written["device"]["compute"]["fp4_flops"] = 10.0e15
    written["device"]["compute"]["derate"]["fp4"] = 0.70
    spec = MachineSpec.from_mapping(written)
    assert spec.value("device.compute.fp4_flops") == 10.0e15
    assert spec.value("interconnect.intra_node.domain_devices") == 8
    assert spec.echo() == written
    assert validate(written).ok


def test_a_spec_without_an_fp4_rate_reads_and_declines_only_that_field():
    spec = read()
    assert validate(document()).ok
    with pytest.raises(SpecRefusal, match="declared by this schema as optional"):
        spec.value("device.compute.fp4_flops")


def test_a_fabric_narrower_than_the_node_is_refused_by_name():
    written = document()
    written["interconnect"]["intra_node"]["domain_devices"] = 7
    refusal = refusal_from(written)
    assert refusal.rule is Rule.SHAPE
    assert "`interconnect.intra_node.domain_devices` is 7" in refusal.what
    assert "`device.count_per_node` 8" in refusal.what
    assert not validate(written).ok
    written["interconnect"]["intra_node"]["domain_devices"] = 8
    assert MachineSpec.from_mapping(written)


def test_a_spec_without_a_fabric_width_is_refused_as_missing():
    written = document()
    del written["interconnect"]["intra_node"]["domain_devices"]
    refusal = refusal_from(written)
    assert refusal.what == "`interconnect.intra_node.domain_devices` is missing"


def test_the_mi455_spec_loads_with_a_fabric_wider_than_its_node():
    spec = MachineSpec.from_mapping(mi455())
    assert spec.value("device.count_per_node") == 4
    assert spec.value("interconnect.intra_node.domain_devices") == 72
    assert spec.value("device.compute.fp4_flops") == 40.265e15
    assert spec.origin("device.compute.fp4_flops") == {"origin": "datasheet"}
    for path in LATENCIES:
        assert spec.origin(path) == {"origin": "assumed"}


def test_the_mi455_latencies_stay_assumed_through_a_re_merge():
    first = mi455()
    host = {"host": {"cpu": copy.deepcopy(DOCUMENT["host"]["cpu"])}}
    again = merge(
        [
            fragment(
                "mi455.json",
                {k: v for k, v in first.items() if k != "name"},
                machine=MI455,
            ),
            fragment("probe", host, machine=MI455, date="2026-10-20"),
        ]
    )
    spec = MachineSpec.from_mapping(saved(again.document))
    assert spec.origin("host.cpu.cores_physical") == {"origin": "probed"}
    assert spec.origin("device.compute.fp4_flops") == {"origin": "datasheet"}
    for path in LATENCIES:
        assert spec.origin(path) == {"origin": "assumed"}
