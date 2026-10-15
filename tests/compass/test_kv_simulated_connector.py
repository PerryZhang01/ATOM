# SPDX-License-Identifier: MIT
"""The simulated KV connector's price, and the two halves the factory builds.

When a transfer is released is tested through real schedulers in
`test_kv_remote_prefill.py`; here is what it costs. The machine spec is the
complete document the spec tests own, so there is one valid spec in the suite
rather than two that can drift apart, and the two bandwidths below are made by
editing a copy of it -- which means they travel through the schema's
peak-and-derate rule the way a real one would.
"""

import copy
import json
import pathlib

import pytest
from conftest import atom_config_double
from test_spec_schema import DOCUMENT
from transformers import PretrainedConfig

import atom.compass.kv as compass_kv
from atom.compass.backends import KvGeometry
from atom.compass.clock import LpId, prefill_decode_table
from atom.compass.kv import (
    TRANSFER_KEY,
    WRITE_REQ_KEY,
    Scope,
    SimulatedKVConnector,
    TransferModel,
    UnboundSeam,
)
from atom.compass.kv.transfer import Link
from atom.compass.spec import MachineSpec, SpecRefusal
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import ConnectorMetadata
from atom.utils import clock
from atom.utils.clock import LPRuntime

CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
BLOCK_SIZE = 64

#: A nanosecond: the "one tick before the deadline" the release must not take.
TICK = 1.0e-9

#: Two link speeds, as spec peaks. The document's own derate turns each into
#: what a transfer actually reaches.
PEAKS = (64.0e9, 400.0e9)


@pytest.fixture(scope="module")
def geometry():
    """One worker's KV block, for a real model's shape at one rank."""
    raw = json.loads(CONFIG_JSON.read_text())
    hf = PretrainedConfig.from_dict(raw["text_config"])
    return KvGeometry.from_hf_config(hf, block_size=BLOCK_SIZE)


def spec_with(peak_bytes_per_s, *, link="intra_node"):
    """The suite's machine spec with one link speed replaced."""
    document = copy.deepcopy(DOCUMENT)
    document["interconnect"][link]["link_bandwidth_bytes_per_s"] = peak_bytes_per_s
    return MachineSpec.from_mapping(document)


def model_for(geometry, peak, scope=Scope.INTRA_NODE):
    """The transfer model a spec at that link speed gives this geometry."""
    return TransferModel.from_spec(spec_with(peak), geometry, scope)


def test_the_worker_half_is_built_by_name_and_reports_nothing():
    """Timing is the scheduler half's; the worker is polled and says nothing."""
    assert KVConnectorFactory.canonical_name("compass") == "compass"
    config = atom_config_double(kv_transfer_config={"kv_connector": "compass"})
    worker = KVConnectorFactory.create_connector(config, role="worker")
    assert isinstance(worker, SimulatedKVConnector)
    meta = ConnectorMetadata()
    meta.add_new_req_to_recv(request_id="r", local_block_ids=[0], kv_transfer_params={})
    worker.start_load_kv(meta)
    assert worker.get_finished() == (set(), set())


@pytest.mark.parametrize(
    "missing, refusal",
    [
        (TRANSFER_KEY, TRANSFER_KEY),
        (WRITE_REQ_KEY, WRITE_REQ_KEY),
        (None, "LP runtime"),
    ],
)
def test_the_scheduler_half_refuses_what_it_cannot_invent(
    geometry, monkeypatch, missing, refusal
):
    """No price, no address or no LP clock: refused by name at construction."""
    kv = {
        "kv_connector": "compass",
        TRANSFER_KEY: model_for(geometry, PEAKS[0]),
        WRITE_REQ_KEY: "inproc://unused",
    }
    kv.pop(missing, None)
    if missing is not None:
        table = prefill_decode_table(
            admission_path="serving",
            ipc_s=0.0,
            stream_s=0.0,
            router_s=0.0,
            kv_write_req_s=0.0,
        )
        monkeypatch.setattr(
            clock, "_installed", LPRuntime(LpId("engine-D"), table, None)
        )
    config = atom_config_double(kv_transfer_config=kv)
    with pytest.raises(UnboundSeam, match=refusal):
        KVConnectorFactory.create_connector(config, role="scheduler")


def test_the_connector_names_no_clock_of_its_own():
    """The strongest form of "it reads the clock it is given": there is no other.

    Asserted against the source, because a wall-clock fallback reached only on
    a path no test drives would pass every timing test here and still silently
    put a real run on the wall clock. The walk is recursive and the list of
    spellings is wider than the module needs today, so a file added to this
    package later is scanned by this test rather than exempt from it.
    """
    package = pathlib.Path(compass_kv.__file__).parent
    scanned = sorted(package.rglob("*.py"))
    assert scanned, "the package was not found, so nothing was checked"
    for module in scanned:
        source = module.read_text()
        for forbidden in (
            "import time",
            "monotonic",
            "perf_counter",
            "time.time",
            "datetime",
            "os.times",
            "get_event_loop",
            "clock_gettime",
        ):
            assert forbidden not in source, f"{module.name} names {forbidden}"


def test_the_derate_is_spent_and_not_the_spec_peak(geometry):
    """A datasheet number is not an achievable one, and is not charged as one."""
    spec = spec_with(PEAKS[0])
    model = TransferModel.from_spec(spec, geometry, Scope.INTRA_NODE)
    derate = spec.value("interconnect.intra_node.derate")
    assert 0 < derate < 1, "this document states no haircut, so nothing is proven"
    assert model.bandwidth_bytes_per_s == PEAKS[0] * derate


def test_the_named_side_of_the_node_boundary_is_the_one_priced(geometry):
    """No transfer is quietly priced on the other side's link."""
    spec = spec_with(PEAKS[0])
    intra = TransferModel.from_spec(spec, geometry, Scope.INTRA_NODE)
    inter = TransferModel.from_spec(spec, geometry, Scope.INTER_NODE)
    assert intra.latency_s == spec.value("interconnect.intra_node.link_latency_s")
    assert inter.latency_s == spec.value("interconnect.inter_node.link_latency_s")
    assert intra.duration_s(512) < inter.duration_s(512)


def test_a_spec_that_cannot_answer_for_the_link_refuses(geometry):
    """The spec's refusal travels out; the other link is not substituted.

    The refusal has to name the field it could not answer for, and must name
    nothing from the other side of the node boundary -- a message mentioning
    the intra-node link here would be a fallback announcing itself. The rule's
    own sentence is deliberately not pinned: a spec built by hand this way is
    declined as "not a field of this schema", which is inaccurate for a field
    the schema does declare, and freezing that wording here would make a
    correction to the spec package fail this test.
    """
    spec = spec_with(PEAKS[0])
    without = MachineSpec(
        values={
            path: value
            for path, value in spec.values.items()
            if not path.startswith("interconnect.inter_node")
        },
        tokenizers=spec.tokenizers,
    )
    TransferModel.from_spec(without, geometry, Scope.INTRA_NODE)
    with pytest.raises(SpecRefusal) as refusal:
        TransferModel.from_spec(without, geometry, Scope.INTER_NODE)
    message = str(refusal.value)
    assert "interconnect.inter_node.link_" in message, message
    assert "intra_node" not in message, message


def test_a_transfer_of_no_blocks_still_costs_the_latency(geometry):
    """Both ends still have to agree that there was nothing to move."""
    assert model_for(geometry, PEAKS[0]).duration_s(0) == pytest.approx(
        model_for(geometry, PEAKS[0]).latency_s
    )


def test_a_negative_block_count_is_refused(geometry):
    """A transfer that finished before it started is not a fast transfer."""
    with pytest.raises(ValueError, match="not a transfer"):
        model_for(geometry, PEAKS[0]).duration_s(-1)


# -- Writes sharing one link ----------------------------------------------

#: Binary-exact numbers, so completion times compare with `==`.
EXACT = TransferModel(latency_s=0.5, bandwidth_bytes_per_s=4.0, bytes_per_block=2)


def test_a_lone_write_completes_at_its_duration():
    link = Link(EXACT)
    link.start("x", 1.0, 3)
    done = 1.0 + EXACT.duration_s(3)
    assert link.finished(done - TICK) == {}
    assert link.finished(done) == {"x": done}
    assert not link.busy()


def test_writes_in_flight_together_share_the_bandwidth():
    """x and y share 4 B/s from 0, so x's 2 bytes are through at 1, when z
    arrives; z and y then share it until z's 2 bytes are through at 2."""
    link = Link(EXACT)
    link.start("x", 0.0, 1)
    link.start("y", 0.0, 3)
    link.start("z", 1.0, 1)
    assert link.finished(2.0) == {"x": 1.5}
    assert link.finished(10.0, notify=0.25) == {"z": 2.5, "y": 3.0}


def test_two_ends_asking_at_different_times_compute_the_same_times(geometry):
    """One end polls between every pair of arrivals, the other only at the end."""
    model = model_for(geometry, PEAKS[0])
    arrivals = [(k, 0.37 * (k // 3) + 1e-3 * k, 1 + (7 * k) % 5) for k in range(12)]
    asked, once = Link(model), Link(model)
    seen = {}
    for i, (key, a, blocks) in enumerate(arrivals):
        asked.start(key, a, blocks)
        once.start(key, a, blocks)
        if i + 1 < len(arrivals):
            seen |= asked.finished((a + arrivals[i + 1][1]) / 2)
    seen |= asked.finished(float("inf"))
    assert seen == once.finished(float("inf"))
    assert len(seen) == len(arrivals)


def test_an_arrival_behind_an_event_the_link_has_passed_is_refused():
    link = Link(EXACT)
    link.start("x", 0.0, 1)
    link.finished(10.0)
    with pytest.raises(ValueError, match="arrives at 0.25, after this link"):
        link.start("y", 0.25, 1)
