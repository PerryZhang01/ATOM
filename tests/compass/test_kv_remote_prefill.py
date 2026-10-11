# SPDX-License-Identifier: MIT
"""A disaggregated prefill through two real schedulers, and the blob that starts one.

**The blob's field set is not asserted against a list typed into this file.**
It is parsed out of Mooncake, the backend the simulated connector stands in
for, so a field added there, or renamed, turns this red.

**The transfer is not asserted by calling the connector.** A prefill engine
and a decode engine, each a real `Scheduler` with the connector's scheduler
half on its own `LPRuntime`, are stepped in `EngineCore`'s order: schedule,
the forward, the KV poll, postprocess. The clock authority is a script: every
clock request is granted the next time the test names, releasing the write
requests it names.
"""

from __future__ import annotations

import ast
import ipaddress
import itertools
import json
import pathlib
import socket
import sys
import time
from types import SimpleNamespace

import numpy
import pytest
import torch
from conftest import MockConfig, atom_config_double
from test_kv_simulated_connector import BLOCK_SIZE, CONFIG_JSON, PEAKS, TICK, model_for
from test_runner_non_allocating import ATOM_RUNNER, PACKAGE, _classes
from transformers import PretrainedConfig

from atom.compass.backends import KvGeometry
from atom.compass.clock import LpId, prefill_decode_table
from atom.compass.kv import TRANSFER_KEY, WRITE_REQ_KEY
from atom.compass.kv.connector import SimulatedKVConnector
from atom.compass.kv.handoff import (
    SIMULATED_ENGINE_ID,
    SIMULATED_HOST,
    SIMULATED_PORT,
    transfer_params,
)
from atom.compass.runner.overrides import NonAllocatingRunner, RunnerRefusal
from atom.kv_transfer import disaggregation
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import ConnectorMetadata, KVConnectorOutput
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams
from atom.utils import clock, forward_context
from atom.utils.clock import LPRuntime

#: The backend whose blob shape the router and the consumer were built around.
MOONCAKE_SOURCE = (
    pathlib.Path(disaggregation.__file__).parent / "mooncake" / "mooncake_connector.py"
)

#: The first sampled token the producer hands back; the consumer resumes on it.
FIRST_TOKEN_ID = 7

#: A prompt long enough to occupy several blocks of the pool below.
PROMPT = list(range(40))

#: The write request's declared latency, which is also the notify latency.
L = 2.0**-10
KV = "engine-D->engine-P:kv_write_req"
TABLE = prefill_decode_table(
    admission_path="serving",
    ipc_s=L,
    stream_s=L,
    router_s=L,
    kv_write_req_s=L,
)
_PAIRS = itertools.count()


@pytest.fixture(scope="module")
def geometry():
    """One worker's KV block, for a real model's shape at one rank."""
    raw = json.loads(CONFIG_JSON.read_text())
    hf = PretrainedConfig.from_dict(raw["text_config"])
    return KvGeometry.from_hf_config(hf, block_size=BLOCK_SIZE)


class Script:
    """The clock authority: each clock request gets the next scripted grant."""

    def __init__(self):
        self.grants = []

    def send(self, request):
        pass

    def recv(self):
        return self.grants.pop(0)


def runtime(lp):
    rt = LPRuntime(LpId(lp), TABLE, Script())
    rt.start_run()
    return rt


def at(rt, t, *released):
    """Move `rt`'s clock to `t`, releasing the write requests `(seq, arrival)`."""
    rt.conn.grants.append((t, {KV: list(released)} if released else {}))
    rt.advance_to(t)


def connector(rt, monkeypatch, model, *, kv_role, address="inproc://unused", **cfg):
    """The scheduler half, built the way the engine builds it, on `rt`."""
    monkeypatch.setattr(clock, "_installed", rt)
    kv = {
        "kv_connector": "compass",
        "kv_role": kv_role,
        TRANSFER_KEY: model,
        WRITE_REQ_KEY: address,
    }
    half = KVConnectorFactory.create_connector(
        atom_config_double(kv_transfer_config=kv, **cfg), role="scheduler"
    )
    monkeypatch.setattr(clock, "_installed", None)
    return half


@pytest.fixture
def pd(geometry, monkeypatch):
    """A prefill and a decode engine whose connectors share one write-request pair."""
    model = model_for(geometry, PEAKS[0])
    address = f"inproc://kv-write-req-{next(_PAIRS)}"
    rt_p, rt_d = runtime("engine-P"), runtime("engine-D")
    p = connector(rt_p, monkeypatch, model, kv_role="kv_producer", address=address)
    d = connector(rt_d, monkeypatch, model, kv_role="kv_consumer", address=address)
    return SimpleNamespace(
        model=model,
        address=address,
        rt_p=rt_p,
        rt_d=rt_d,
        p=scheduler_with(p),
        d=scheduler_with(d),
    )


def scheduler_with(connector_half, **config):
    """A real scheduler, sized by the caller, carrying this connector."""
    settings = {
        "num_kvcache_blocks": 100,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 1000,
    }
    settings.update(config)
    engine = Scheduler(MockConfig(**settings))
    engine.kv_connector = connector_half
    return engine


def engine_step(engine, rt, t, *released):
    """`EngineCore`'s order: schedule, the forward ending at `t`, the KV poll,
    postprocess. With nothing scheduled it is the idle drain: the poll only."""
    result = engine.schedule()
    at(rt, t, *released)
    engine._update_from_kv_xfer_finished(KVConnectorOutput())
    batch = None if result is None else result[0]
    if batch is not None and batch.req_ids:
        engine.postprocess(
            list(result[1].values()),
            ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[(FIRST_TOKEN_ID,)] * len(batch.req_ids),
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )


def prefilled(pd, n, r):
    """`n` requests prefilled on engine-P by a forward that ends at `r`."""
    seqs = [
        Sequence(PROMPT, 4, sampling_params=SamplingParams(max_tokens=1))
        for _ in range(n)
    ]
    for seq in seqs:
        pd.p.add(seq)
    engine_step(pd.p, pd.rt_p, r)
    return seqs


def admitted(pd, seqs, seq_factory, t):
    """Decode requests for the prefilled `seqs`, admitted by engine-D at `t`."""
    at(pd.rt_d, t)
    out = [
        seq_factory(PROMPT, kv_transfer_params=dict(seq.kv_transfer_params_output))
        for seq in seqs
    ]
    for seq in out:
        pd.d.add(seq)
    pd.d.schedule()
    return out


def free_blocks(engine):
    return engine.block_manager.kv.num_free


# -- The transfer --------------------------------------------------------

#: Prefill's forward ends at R; decode is busy until SEND, so admits late.
R, SEND = 1.0, 2.0


def test_prefill_frees_at_max_a_r_plus_t_and_decode_is_ready_at_a_t_notify(
    pd, seq_factory
):
    """One request whose decode admission is delayed past prefill's finish."""
    total = free_blocks(pd.p)
    (x,) = prefilled(pd, 1, R)
    assert x.id in pd.p.deferred_free_blocks
    (y,) = admitted(pd, [x], seq_factory, SEND)
    assert y.status is SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert y.kv_transfer_params["do_remote_prefill"] is False, "Mooncake clears it"
    assert pd.d.kv_connector.has_pending_work() is True, "an idle decode stops polling"

    a = SEND + L
    T = pd.model.duration_s(len(y.block_table))
    e, ready = max(a, R) + T, a + T + L
    assert pd.rt_d.send_log == [(KV, 0, a)], "the write request's arrival"

    engine_step(pd.p, pd.rt_p, a, (0, a))
    engine_step(pd.p, pd.rt_p, e - TICK)
    assert x.id in pd.p.deferred_free_blocks, "freed before the write was done"
    engine_step(pd.p, pd.rt_p, e)
    assert pd.p.deferred_free_blocks == {}
    assert free_blocks(pd.p) == total

    engine_step(pd.d, pd.rt_d, ready - TICK)
    pd.d.schedule()
    assert y.status is SequenceStatus.WAITING_FOR_REMOTE_KVS, "ready early"
    engine_step(pd.d, pd.rt_d, ready)
    pd.d.schedule()
    assert y.status is SequenceStatus.RUNNING
    assert y.token_ids[y.num_prompt_tokens] == FIRST_TOKEN_ID
    assert pd.d.finished_recving_kv_req_ids == [], "an id with no reachable pop"
    assert pd.d.kv_connector.has_pending_work() is False


def test_a_write_request_for_an_unfinished_prefill_is_refused(pd, seq_factory):
    """r > a: the write request arrived before prefill had the data."""
    x = SimpleNamespace(
        kv_transfer_params_output={"do_remote_prefill": True, "transfer_id": 999}
    )
    admitted(pd, [x], seq_factory, SEND)
    with pytest.raises(ValueError, match="before its prefill finished"):
        engine_step(pd.p, pd.rt_p, SEND + L, (0, SEND + L))


#: The decode pool's block size (`MockConfig`'s), and a different one.
HBS = 4


@pytest.mark.parametrize(
    "blob_hbs, priced",
    [(HBS, 1), (4 * HBS, len(PROMPT) // HBS)],
    ids=["matching", "different"],
)
def test_a_decode_side_prefix_hit_is_written_only_when_both_hash_alike(
    pd, monkeypatch, seq_factory, blob_hbs, priced
):
    """Mooncake's rule: the blocks decode already caches are not written,
    unless prefill hashed its blocks at another size."""
    half = connector(
        pd.rt_d,
        monkeypatch,
        pd.model,
        kv_role="kv_consumer",
        address=pd.address,
        kv_cache_block_size=HBS,
    )
    pd.d = scheduler_with(half, enable_prefix_caching=True)
    warm = seq_factory(PROMPT)
    pd.d.block_manager.allocate(warm)
    pd.d.block_manager.hash_blocks(warm, warm.num_tokens)
    pd.d.block_manager.deallocate(warm)

    blob = {"do_remote_prefill": True, "transfer_id": 0, "hash_block_size": blob_hbs}
    (y,) = admitted(
        pd, [SimpleNamespace(kv_transfer_params_output=blob)], seq_factory, SEND
    )
    assert y.status is SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert y.num_cached_tokens == len(PROMPT) - HBS, "no decode-side hit"

    ready = SEND + L + pd.model.duration_s(priced) + L
    engine_step(pd.d, pd.rt_d, ready - TICK)
    pd.d.schedule()
    assert y.status is SequenceStatus.WAITING_FOR_REMOTE_KVS, "ready early"
    engine_step(pd.d, pd.rt_d, ready)
    pd.d.schedule()
    assert y.status is SequenceStatus.RUNNING


def test_pipeline_parallelism_is_refused_by_name(geometry, monkeypatch):
    """A PP head never hands the connector an empty poll, so nothing completes."""
    with pytest.raises(
        ValueError,
        match="pipeline_parallel_size is 2.*process_completions.*MSG_RELEASE",
    ):
        connector(
            runtime("engine-P"),
            monkeypatch,
            model_for(geometry, PEAKS[0]),
            kv_role="kv_producer",
            pipeline_parallel_size=2,
        )


@pytest.mark.parametrize("follower", [False, True], ids=["idle", "follower"])
def test_an_idle_producer_frees_every_finished_prefill(pd, seq_factory, follower):
    """N prefills, then nothing (or one more request): all freed on their times."""
    total = free_blocks(pd.p)
    seqs = prefilled(pd, 3, R)
    if follower:
        seqs += prefilled(pd, 1, R + 0.5)
    held = sum(len(seq.block_table) for seq in seqs)
    assert len(pd.p.deferred_free_blocks) == len(seqs)
    assert free_blocks(pd.p) == total - held

    admitted(pd, seqs, seq_factory, SEND)
    a = SEND + L
    engine_step(pd.p, pd.rt_p, a, *[(k, a) for k in range(len(seqs))])
    T = pd.model.duration_s(max(len(seq.block_table) for seq in seqs))
    engine_step(pd.p, pd.rt_p, a + T)
    assert pd.p.deferred_free_blocks == {}
    assert free_blocks(pd.p) == total


def test_a_request_the_cap_bounced_parks_on_a_later_step(pd, seq_factory):
    """The claim is not spent on the asking: the flag is cleared at allocation.

    So a request bounced before allocation asks again and parks, and each
    parked request sends exactly one write request.
    """
    pd.d.max_num_seqs = 1
    blob = {"do_remote_prefill": True, "transfer_id": 0}
    first = seq_factory(PROMPT, kv_transfer_params=dict(blob))
    bounced = seq_factory(PROMPT, kv_transfer_params=dict(blob, transfer_id=1))
    pd.d.add(first)
    pd.d.add(bounced)
    pd.d.schedule()
    assert first.status is SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert bounced.status is SequenceStatus.WAITING, "the cap did not bounce it"
    pd.d.max_num_seqs = 4
    pd.d.schedule()
    assert bounced.status is SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert [seq for _, seq, _ in pd.rt_d.send_log] == [0, 1]


@pytest.mark.parametrize("side", ["p", "d"])
def test_an_ordinary_request_is_neither_claimed_nor_sent(pd, seq_factory, side):
    seq = seq_factory(PROMPT)
    half = getattr(pd, side).kv_connector
    assert half.get_num_new_matched_tokens(seq) == (0, False)
    seq.block_table = [0, 1]
    half.update_state_after_alloc(seq)
    assert getattr(pd, f"rt_{side}").send_log == []


def test_the_producing_side_refuses_a_remote_fill(pd, seq_factory):
    """A deployment cannot wait for a transfer it is the source of."""
    seq = seq_factory(PROMPT, kv_transfer_params={"do_remote_prefill": True})
    with pytest.raises(ValueError, match="producer side"):
        pd.p.kv_connector.update_state_after_alloc(seq)


def test_a_parked_request_is_not_counted_as_admittable_work(pd, seq_factory):
    """The engine's guards read the suspended state; the request must have it."""
    seq = seq_factory(
        PROMPT, kv_transfer_params={"do_remote_prefill": True, "transfer_id": 0}
    )
    pd.d.add(seq)
    pd.d.schedule()
    assert seq.status is SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert pd.d._waiting_new_token_count() == 0
    assert pd.d._can_admit_head_prefill() is False
    assert pd.d._oldest_waiting_prefill_age_ms() == 0.0

    # The positive control: an ordinary request queued beside it is counted.
    queued = seq_factory(list(range(24)))
    queued.arrive_time = time.time() - 1.0
    pd.d.add(queued)
    assert pd.d._waiting_new_token_count() == 24
    assert pd.d._can_admit_head_prefill() is True
    assert 1000.0 <= pd.d._oldest_waiting_prefill_age_ms() < 60_000.0


# -- The blob ------------------------------------------------------------


def relayed_fields() -> frozenset:
    """The keys of Mooncake's own transfer blob, read from its source.

    Parsed rather than imported: the module reaches for an RDMA library this
    tier does not have. Exactly one assignment is expected.
    """
    tree = ast.parse(MOONCAKE_SOURCE.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(
            isinstance(target, ast.Attribute)
            and target.attr == "kv_transfer_params_output"
            for target in node.targets
        ):
            continue
        assert all(
            isinstance(key, ast.Constant) and isinstance(key.value, str)
            for key in node.value.keys
        ), f"{MOONCAKE_SOURCE.name} builds its blob with computed keys"
        found.append(frozenset(key.value for key in node.value.keys))
    assert len(found) == 1, f"expected one blob assignment, found {len(found)}"
    return found[0]


def assert_relays_every_field(blob) -> None:
    """The check the field-set test makes, and the one the drop test breaks."""
    missing = sorted(relayed_fields() - set(blob))
    assert not missing, f"the relayed blob would not carry {missing}"
    extra = sorted(set(blob) - relayed_fields())
    assert not extra, f"the relayed blob would carry {extra}, which nothing reads"


#: What a finished request holds when the producer hands it back.
FINISHED_ID = 11
FINISHED_BLOCKS = [4, 5, 6]
FINISHED_HIT_TOKENS = 128


def finished_sequence():
    """A request the way `request_finished` finds one: done, with a table."""
    return SimpleNamespace(
        id=FINISHED_ID,
        block_table=list(FINISHED_BLOCKS),
        output_tokens=[FIRST_TOKEN_ID, 8, 9],
        spec_token_ids=[],
        prefix_cache_hit_tokens=FINISHED_HIT_TOKENS,
        kv_transfer_params_output=None,
    )


def simulated_blob(tp_size=4, dp_rank=2) -> dict:
    return transfer_params(
        finished_sequence(),
        tp_size=tp_size,
        dp_rank=dp_rank,
        pp_size=1,
        hash_block_size=16,
    )


def assert_could_not_be_dialled(value, field) -> None:
    """Refuse a value anything downstream could turn into a connection.

    Asserted as a property of whatever the field holds, not as an identity
    with the constant beside it, because a defect that moves the constant
    moves both sides of such an identity and is invisible to it.
    """
    host = str(value).split(":")[0]
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{field} carries the address {host!r}")
    own = socket.gethostname()
    labels = {part.lower() for part in str(value).replace(":", ".").split(".")}
    borrowed = labels & {own.lower(), own.lower().split(".")[0]}
    assert not borrowed, (
        f"{field} carries {sorted(borrowed)}, taken from the name of the "
        "machine this run is on"
    )


def test_the_blob_carries_the_field_set_the_router_relays():
    """The emitted set equals Mooncake's, re-derived."""
    assert set(simulated_blob()) == relayed_fields()
    assert_relays_every_field(simulated_blob())


@pytest.mark.parametrize("field", sorted(relayed_fields()))
def test_dropping_one_field_from_the_blob_is_refused_by_name(field):
    """Every field is load-bearing, and the check says which one went."""
    short = dict(simulated_blob())
    del short[field]
    with pytest.raises(AssertionError) as refusal:
        assert_relays_every_field(short)
    assert field in str(refusal.value)


def test_no_field_of_the_blob_names_a_machine_the_simulation_runs_on():
    """A simulated peer has no address, and must not borrow a real one.

    The identities say the blob is built from the constants; the properties
    say what those constants may hold, and survive the constants changing.
    """
    blob = simulated_blob()
    assert blob["remote_host"] == SIMULATED_HOST
    assert blob["remote_port"] == SIMULATED_PORT == 0
    assert blob["remote_handshake_port"] == SIMULATED_PORT
    assert blob["remote_engine_id"] == SIMULATED_ENGINE_ID

    host = blob["remote_host"]
    assert host.endswith(".invalid"), "a name that could resolve"
    assert host.count(".") == 1, (
        f"{host!r} is a real name with the reserved suffix pinned on the end; "
        "the whole point is that there is one label and it was invented here"
    )
    assert ":" not in blob["remote_engine_id"], "an identity, not host:port"
    for field in ("remote_host", "remote_engine_id"):
        assert_could_not_be_dialled(blob[field], field)


def test_the_engine_id_is_the_label_the_host_invents_and_nothing_else():
    """The engine id is one invented label, the one the host reserves."""
    blob = simulated_blob()
    engine_id = blob["remote_engine_id"]
    assert "." not in engine_id, (
        f"remote_engine_id {engine_id!r} is a dotted name, which a resolver "
        "can route; it must be the single invented label"
    )
    assert blob["remote_host"] == f"{engine_id}.invalid", (
        f"remote_engine_id {engine_id!r} is not the label remote_host "
        f"{blob['remote_host']!r} reserves under .invalid"
    )


def test_the_ranks_the_router_reads_are_numbers(pd, geometry, monkeypatch):
    """The router drops `dp_rank` unless it is a number, and says nothing.

    Driven through the connector from a config carrying the widths as floats
    with no fractional part: each is converted to an `int` on the way in.
    """
    half = connector(
        pd.rt_p,
        monkeypatch,
        pd.model,
        kv_role="kv_producer",
        address=f"inproc://kv-write-req-{next(_PAIRS)}",
        tensor_parallel_size=8.0,
        parallel_config=SimpleNamespace(data_parallel_rank=3.0),
    )
    seq = finished_sequence()
    half.request_finished(seq)
    relayed = seq.kv_transfer_params_output
    assert isinstance(relayed["tp_size"], int) and relayed["tp_size"] == 8
    assert isinstance(relayed["dp_rank"], int) and relayed["dp_rank"] == 3


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("tp_size", 8.5, id="tp_size-float"),
        pytest.param("tp_size", "8.5", id="tp_size-text"),
        pytest.param("tp_size", "eight", id="tp_size-word"),
        pytest.param("dp_rank", 2.5, id="dp_rank-float"),
        pytest.param("dp_rank", "2.5", id="dp_rank-text"),
        pytest.param("tp_size", "8.0", id="tp_size-decimal-text"),
        pytest.param("tp_size", "1e1", id="tp_size-exponent-text"),
        pytest.param("tp_size", True, id="tp_size-true"),
        pytest.param("tp_size", False, id="tp_size-false"),
        pytest.param("dp_rank", True, id="dp_rank-true"),
        pytest.param("dp_rank", False, id="dp_rank-false"),
        pytest.param("tp_size", numpy.bool_(True), id="tp_size-numpy-true"),
        pytest.param("tp_size", numpy.bool_(False), id="tp_size-numpy-false"),
        pytest.param("tp_size", torch.tensor(True), id="tp_size-torch-true"),
        pytest.param("tp_size", "8", id="tp_size-integer-text"),
        pytest.param("dp_rank", "3", id="dp_rank-integer-text"),
        pytest.param("tp_size", " 8 ", id="tp_size-padded-text"),
    ],
)
def test_a_width_that_is_not_a_whole_number_is_refused_by_name(
    geometry, monkeypatch, field, value
):
    """A width that is not exactly an integer refuses the connector by name.

    Refused when the connector is built, so a malformed config never serves a
    request: 8.5 cast to 8, or a boolean sent as 1 or 0, would name a
    deployment that was never launched.
    """
    widths = {"tp_size": 8, "dp_rank": 3, field: value}
    with pytest.raises(ValueError, match=f"^{field} is .*not a whole number"):
        connector(
            runtime("engine-P"),
            monkeypatch,
            model_for(geometry, PEAKS[0]),
            kv_role="kv_producer",
            tensor_parallel_size=widths["tp_size"],
            parallel_config=SimpleNamespace(data_parallel_rank=widths["dp_rank"]),
        )


def test_the_blob_carries_the_request_and_not_a_template():
    """What is specific to the request comes off the request."""
    blob = simulated_blob()
    assert blob["do_remote_prefill"] is True
    assert blob["do_remote_decode"] is False
    assert blob["remote_block_ids"] == FINISHED_BLOCKS
    assert blob["transfer_id"] == FINISHED_ID
    assert blob["first_token_id"] == FIRST_TOKEN_ID
    assert blob["draft_token_ids"] == []
    assert blob["prefix_cache_hit_tokens"] == FINISHED_HIT_TOKENS


def test_a_finished_request_carries_the_blob_out(pd):
    """The connector attaches it where the API layer looks for it."""
    seq = finished_sequence()
    pd.p.kv_connector.request_finished(seq)
    assert seq.kv_transfer_params_output == transfer_params(
        seq, tp_size=1, dp_rank=0, pp_size=1, hash_block_size=16
    )


def test_the_emitted_blob_is_one_a_consumer_can_actually_consume():
    """The blob goes back in where a consumer reads it, through `ReqMeta`."""
    blob = simulated_blob(tp_size=4, dp_rank=2)
    meta = ConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id=FINISHED_ID, local_block_ids=[0, 1], kv_transfer_params=blob
    )
    req = meta.reqs_to_recv[FINISHED_ID]

    assert req.remote_block_ids == FINISHED_BLOCKS
    assert req.remote_host == SIMULATED_HOST
    assert req.remote_port == SIMULATED_PORT
    assert req.remote_handshake_port == SIMULATED_PORT
    assert req.remote_engine_id == SIMULATED_ENGINE_ID
    assert req.tp_size == 4
    assert req.transfer_id == FINISHED_ID


# -- The runner ----------------------------------------------------------


def test_the_compass_runner_builds_the_worker_half(monkeypatch):
    """The worker the engine polls is built by the runner and reports nothing.

    Importing either runner needs a driver, so `CompassModelRunner`'s class
    statement is compiled from source over the overrides and an ATOM
    `ModelRunner` holding only its `process_kvconnector_output`, and the TP
    group `ModelRunner.__init__` opens is stood in at rank 0.
    """
    atom_runner = _classes(ATOM_RUNNER)["ModelRunner"]
    (method,) = [
        n
        for n in atom_runner.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_kvconnector_output"
    ]
    atom_runner.body = [method]
    namespace = {
        "torch": torch,
        "get_kvconnector": forward_context.get_kvconnector,
        "NonAllocatingRunner": NonAllocatingRunner,
    }
    compass_runner = _classes(PACKAGE / "model_runner.py")["CompassModelRunner"]
    exec(ast.unparse(atom_runner), namespace)  # noqa: S102
    exec(ast.unparse(compass_runner), namespace)  # noqa: S102
    group = SimpleNamespace(get_tp_group=lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", group)
    monkeypatch.setattr(forward_context, "_global_kvconnector", None)

    runner = object.__new__(namespace["CompassModelRunner"])
    runner.config = atom_config_double(
        kv_transfer_config={"kv_connector": "compass", "kv_role": "kv_consumer"}
    )
    assert runner.allocate_kv_cache(100) is True
    worker = forward_context.get_kvconnector()
    assert isinstance(worker, SimulatedKVConnector)
    meta = ConnectorMetadata()
    meta.add_new_req_to_recv(request_id=0, local_block_ids=[0], kv_transfer_params={})
    runner.process_kvconnector_output(meta)
    assert worker.get_finished() == (set(), set())


@pytest.mark.parametrize(
    "kv", [{"kv_connector": "mooncake"}, {"kv_connector": "moriio"}, {}]
)
def test_the_compass_runner_refuses_a_real_transfer_backend(kv, monkeypatch):
    """Refused by name before anything is built; `{}` is the factory's moriio."""
    calls = []
    monkeypatch.setattr(
        forward_context, "set_kv_cache_data", lambda *a, **k: calls.append(a)
    )
    runner = object.__new__(NonAllocatingRunner)
    runner.config = atom_config_double(
        kv_transfer_config={"kv_role": "kv_consumer", **kv}
    )
    name = kv.get("kv_connector", "moriio")
    with pytest.raises(RunnerRefusal, match=f"kv_connector '{name}' is a real"):
        runner.allocate_kv_cache(100)
    assert calls == []


def test_the_compass_runner_accepts_no_transfer_config(monkeypatch):
    """`{}` is ATOM's default: every run that sets no kv_transfer_config takes this."""
    calls = []
    monkeypatch.setattr(
        forward_context, "set_kv_cache_data", lambda *a, **k: calls.append(a)
    )
    runner = object.__new__(NonAllocatingRunner)
    runner.config = atom_config_double(kv_transfer_config={})
    assert runner.allocate_kv_cache(100) is True
    assert len(calls) == 1


def test_the_compass_runner_refuses_a_config_that_lacks_the_field(monkeypatch):
    """A config with no `kv_transfer_config` is refused, not read as unset."""
    calls = []
    monkeypatch.setattr(
        forward_context, "set_kv_cache_data", lambda *a, **k: calls.append(a)
    )
    runner = object.__new__(NonAllocatingRunner)
    runner.config = SimpleNamespace(kv_transfer_config_renamed={})
    with pytest.raises(RunnerRefusal, match="no field 'kv_transfer_config'"):
        runner.allocate_kv_cache(100)
    assert calls == []
