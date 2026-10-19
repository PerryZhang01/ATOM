# SPDX-License-Identifier: MIT
"""`compass measure ops`, the parts that run with no device.

Graphs are built by `trace ops`'s own folding (`capture.graph.build`); the
stub runs in a stand-in runner whose forward sets ATOM's forward context and
calls the model, as ATOM's runner does.
"""

import dataclasses
import json
import pathlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_bench_replay import _line, rule  # noqa: F401  the rule fixture
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture
from test_parity_record import _unpriced  # noqa: F401  an autouse fixture

from atom.compass import cli, ir
from atom.compass.artifacts import ArtifactStore, Kind, machine_spec_digest
from atom.compass.bench import ops
from atom.compass.capture.forward import decode
from atom.compass.capture.graph import Traced, build, to_json
from atom.compass.spec import MachineSpec
from atom.compass.spec.fields import PINNED
from atom.utils import block_convert
from atom.utils import forward_context as fctx

F32 = "torch.float32"
KEY = "model.layers.{L}.attn"
TREE = pathlib.Path(__file__).resolve().parents[2]


def _t(shape, role, stride=None, dtype=F32):
    stride = stride or [s for s in torch.empty(shape).stride()]
    return ir.Tensor(tuple(shape), dtype, tuple(stride), role)


def _leaf(name, *args, key=None):
    ref = key and ir.ContextRef(key)
    return ir.Op(name, "opaque_leaf", args, (), context_ref=ref)


def _graph(*layer_ops, layers=4):
    """`layer_ops` in each of `layers` layers, as JSON."""
    items = [(i, op) for i in range(layers) for op in layer_ops]
    return json.loads(to_json(build(items, Traced("x tp=1 decode eager", "B=2"))))


def test_each_leaf_holds_every_instance_s_context_key_and_module_path():
    attn = _leaf(
        "aiter.attention",
        ("q", _t([2, 8], "activation")),
        ("layer", ir.ContextRef(KEY)),
        ("rope", ir.ModuleRef("model.rope")),
        key=KEY,
    )
    gemm = _leaf("aiter.gemm", ("x", _t([2, 8], "activation")))

    found = ops.leaves(_graph(attn, gemm)["region"])

    by_op = {op["op"]: instances for op, instances in found.values()}
    assert by_op == {
        "aiter.attention": [
            {"layer": f"model.layers.{i}.attn", "rope": "model.rope"} for i in range(4)
        ],
        "aiter.gemm": [{}],
    }


LIVE = {
    "kv_indptr": torch.arange(9, dtype=torch.int32),
    # Its own shape and stride, with more room than `kv_indptr`.
    "cu_seqlens_q": torch.arange(12, dtype=torch.int32)[:3],
    "slots": torch.arange(64, dtype=torch.int32)[8:],
}


@pytest.mark.parametrize(
    "slot, shape, picked",
    [
        ("kv_indptr_convert", [3], "kv_indptr"),  # a name the slot holds
        ("self", [3], "cu_seqlens_q"),  # then its own shape and stride
        ("self", [5], "kv_indptr"),  # then the tightest with room
        ("self", [40], "slots"),
    ],
)
def test_a_metadata_argument_is_a_view_of_live_metadata(slot, shape, picked):
    spec = {"shape": shape, "dtype": "torch.int32", "stride": [1], "role": "metadata"}

    name, view = ops._meta(spec, slot, LIVE)

    assert name == picked
    assert (
        view.untyped_storage().data_ptr() == LIVE[picked].untyped_storage().data_ptr()
    )
    assert torch.equal(view, LIVE[picked][: shape[0]])


def test_metadata_with_no_room_refuses_by_slot():
    spec = {"shape": [99], "dtype": "torch.int32", "stride": [1], "role": "metadata"}

    with pytest.raises(ValueError, match=r"no room for indices \[99\] torch.int32"):
        ops._meta(spec, "indices", LIVE)


@dataclasses.dataclass
class _Meta:
    tables: torch.Tensor
    lens: dict
    also: torch.Tensor
    owner: torch.nn.Module


def test_a_snapshot_clones_every_tensor_it_reaches_and_keeps_aliases():
    base = torch.arange(24.0)
    tables = base.view(4, 6)[:, 1:3]
    meta = _Meta(tables, {"lens": [base[2:5]]}, tables, torch.nn.Linear(2, 2))

    twin = ops._deep(meta, {})

    assert twin.also is twin.tables and twin.owner is meta.owner
    for old, new in [(tables, twin.tables), (base[2:5], twin.lens["lens"][0])]:
        assert new.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
        assert new.stride() == old.stride() and torch.equal(new, old)


def test_a_module_copy_holds_tensors_of_its_own():
    module = torch.nn.Linear(2, 2)
    module.register_buffer("cache", torch.ones(3))
    module.cos_cache = torch.full((4,), 2.0)

    twin = ops._module_copy(module)

    for name in ("weight", "bias", "cache", "cos_cache"):
        old, new = getattr(module, name), getattr(twin, name)
        assert torch.equal(old, new) and old.data_ptr() != new.data_ptr()
    assert sum(t.numel() for t in ops._own(module)) == 4 + 2 + 3 + 4


class _Runner:
    """ATOM's runner as the stub sees it: a model, its persistent buffers, and
    a forward that writes a batch's blocks and slots into them, sets the
    forward context on them and calls the model."""

    device = torch.device("cpu")

    def __init__(self, contexts=()):
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 3))
        self.forward_vars = {"block_tables": torch.zeros(4, 8, dtype=torch.int32)}
        self.positions = torch.zeros(2, dtype=torch.int64)
        text = SimpleNamespace(hidden_size=3)
        self.config = SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=dict(contexts)),
            num_kvcache_blocks=64,
            hf_config=SimpleNamespace(get_text_config=lambda: text),
            torch_dtype=torch.float32,
        )
        self.batches = []

    def forward(self, batch):
        self.batches.append(batch)
        block, slot = batch
        self.forward_vars["block_tables"].fill_(block)
        self.positions.fill_(slot)
        meta = fctx.AttentionMetaData(block_tables=self.forward_vars["block_tables"])
        context = fctx.Context(positions=self.positions)
        fctx._forward_context = fctx.ForwardContext(attn_metadata=meta, context=context)
        self.model(torch.zeros(2, dtype=torch.int32), context.positions)


def test_a_point_places_its_batch_for_its_context_keys_and_refuses_by_name(
    monkeypatch,
):
    # Two layers of 4 blocks' KV, 256 bytes a block: 1056 bytes a call, so
    # 10 copies past twice a 5000-byte cache, 5 placements of 4 blocks and 2 slots.
    layers = {f"model.layers.{i}.attn": SimpleNamespace(layer_num=i) for i in (0, 1)}
    kv = {f"layer_{i}": SimpleNamespace(k_cache=torch.zeros(64, 16, 4)) for i in (0, 1)}
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", kv)
    runner = _Runner(layers)
    found = ops.leaves(
        _graph(
            _leaf(
                "aten.nowhere.default",
                ("q", _t([2, 4], "activation")),
                ("layer", ir.ContextRef(KEY)),
                key=KEY,
            ),
            _leaf(
                "aten.mm.default",
                ("self", _t([2, 3], "activation")),
                ("mat2", _t([5, 5], "weight")),
            ),
            _leaf("aten.add.Tensor", ("self", _t([2, 3], "kv"))),
            _leaf(
                "aten.add_.Tensor",
                ("self", _t([2, 8], "metadata", dtype="torch.int32")),
            ),
            layers=2,
        )["region"]
    )
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )
    stub = ops.Stub(runner, 5_000)

    with patch.object(runner.model, "forward", stub):
        got = stub.point(placed, decode(2), 20, found)

    assert runner.batches == [(4 * p, 2 * p) for p in range(5)]
    why = {r["op"]: r["refused"] for r in got}
    assert "'nowhere'" in why["aten.nowhere.default"]
    assert (
        "no tensor of the model is a weight [5, 5] torch.float32"
        in why["aten.mm.default"]
    )
    assert "a kv tensor argument [2, 3]" in why["aten.add.Tensor"]
    [meta] = [r for r in got if r["op"] == "aten.add_.Tensor"]
    assert meta["metadata"] == {"self": "block_tables"}
    assert {r["placements"] for r in got} == {5}


def test_each_placement_keeps_its_own_forward_context_and_metadata():
    runner = _Runner()
    stub = ops.Stub(runner, 10_000)
    meta = _t([2, 8], "metadata", dtype="torch.int32")
    stub.leaves = ops.leaves(
        _graph(_leaf("aten.add_.Tensor", ("self", meta)), layers=1)["region"]
    )

    with patch.object(runner.model, "forward", stub):
        for p in range(3):
            runner.forward((4 * p, 2 * p))

    for p, (snap, views) in enumerate(stub.placed):
        [(name, view)] = views.values()
        assert name == "block_tables"
        assert set(snap.attn_metadata.block_tables.flatten().tolist()) == {4 * p}
        assert set(view.flatten().tolist()) == {4 * p}
        assert snap.context.positions.tolist() == [2 * p] * 2


def test_a_shape_the_pool_cannot_place_refuses_its_leaves_and_the_next_is_measured(
    monkeypatch,
):
    runner = _Runner()
    runner.exit = lambda: runner.batches.append("exit")
    monkeypatch.setattr(ops, "_runner", lambda config: runner)
    monkeypatch.setattr("atom.compass.bench.replay.ReplayRule", lambda config: None)

    def point(self, _rule, shape, context, found):
        if shape.seqs == 2:
            raise ValueError("no room for 2 rows")
        return [{"shape": str(shape), "leaf": sig} for sig in found]

    monkeypatch.setattr(ops.Stub, "point", point)
    x = ("self", _t([2, 3], "activation"))
    found = ops.leaves(
        _graph(_leaf("aten.add.Tensor", x), _leaf("aten.mul.Tensor", x))["region"]
    )

    got = ops.measure(None, [(decode(2), 20, found), (decode(4), 20, found)], 10_000)

    assert [(r["op"], r["refused"]) for r in got[:2]] == [
        ("aten.add.Tensor", "ValueError: no room for 2 rows"),
        ("aten.mul.Tensor", "ValueError: no room for 2 rows"),
    ]
    assert {r["shape"] for r in got[:2]} == {str(decode(2))}
    assert got[2:] == [{"shape": str(decode(4)), "leaf": sig} for sig in found]
    assert runner.batches == ["exit"]


class _Launches:
    """A Triton kernel that records the grid of each launch and runs nothing."""

    def __init__(self):
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)
        return lambda *args, **kwargs: None


def test_a_decode_point_takes_an_integer_affine_in_the_context_at_its_own(
    monkeypatch,
    rule,  # noqa: F811
):
    kernel = _Launches()
    monkeypatch.setattr(block_convert, "kv_indices_generate_kernel", kernel)
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})

    def call(stub, op, sig, instances, n):
        kwargs, _ = stub._copy(op, sig, instances, 0)
        ops._function(op["op"])(**kwargs)
        return {}

    monkeypatch.setattr(ops.Stub, "_time", call)
    runner = _Runner()
    stub = ops.Stub(runner, 10_000)
    ints = [_t(shape, "metadata", dtype="torch.int32") for shape in ([2, 8], [16], [3])]
    leaf = _leaf(
        "atom.utils.block_convert:kv_indices_generate_triton",
        *zip(("block_tables", "kv_indices_convert", "kv_indptr_convert"), ints),
        ("ratio", 1),
        ("max_ctx_len", ir.Affine(1, 0)),
    )
    found = ops.leaves(_graph(leaf, layers=1)["region"])
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )

    with patch.object(runner.model, "forward", stub):
        [record] = stub.point(placed, decode(2), 8192, found)

    # ATOM's own step at decode B = 2, context 8192 passes its rows' longest
    # context.
    batch = rule().batch(_line([[1, 8192, True]] * 2))
    tables = torch.zeros(2, 8, dtype=torch.int32)
    block_convert.kv_indices_generate_triton(
        tables, tables, tables, 1, batch.context_lens.max()
    )
    assert "refused" not in record
    assert kernel.grids == [(2, 2), (2, 2)]


def test_an_embedding_reads_the_token_ids_the_forward_was_given():
    runner = _Runner()
    # An index buffer whose name holds the slot's, which the name rule picks.
    runner.forward_vars["spec_state_indices"] = torch.zeros(8, dtype=torch.int32)
    stub = ops.Stub(runner, 10_000)
    embedding = _leaf(
        "aten.embedding.default",
        ("weight", _t([16, 3], "weight")),
        ("indices", _t([2], "metadata", dtype="torch.int32")),
    )
    stub.leaves = ops.leaves(_graph(embedding, layers=1)["region"])

    with patch.object(runner.model, "forward", stub):
        runner.forward((0, 0))

    [(_, views)] = stub.placed
    assert [name for name, _ in views.values()] == ["input_ids"]


def test_a_call_counts_its_module_s_tensors_and_cycles_over_every_layer():
    layers = {f"model.layers.{i}.attn": SimpleNamespace(layer_num=i) for i in (0, 1)}
    kv = {f"layer_{i}": SimpleNamespace(k_cache=torch.zeros(64, 16, 4)) for i in (0, 1)}
    stub = ops.Stub(_Runner(layers), 50)
    attn = {"op": "x", "args": {"layer": {"context": None}}}
    rope = {"op": "x", "args": {"rope": {"module": None}}}
    pairs = [{"layer": key} for key in layers]

    # 4 blocks of 256 bytes is past twice a 50-byte cache, and both layers still
    # cycle.
    assert stub._plan(attn, pairs, kv, 4, 2) == (2, 1024, 1)
    # A Linear(3, 3) holds 12 floats, 48 bytes: 3 copies pass twice 50, on one
    # placement.
    assert stub._plan(rope, [{"rope": "0"}], kv, 4, 2) == (3, 48, 1)


def test_copy_c_rotates_weights_and_instances_and_clones_past_the_first_round():
    runner = _Runner()
    stub = ops.Stub(runner, 10_000)
    op = {
        "op": "aten.mm.default",
        "args": {
            "self": {
                "shape": [2, 3],
                "dtype": F32,
                "stride": [3, 1],
                "role": "metadata",
            },
            "mat2": {"shape": [3, 3], "dtype": F32, "stride": [3, 1], "role": "weight"},
            "rope": {"module": None},
            "layer": {"context": None},
            "out_dtype": "torch.bfloat16",
        },
    }
    live = torch.ones(2, 3)
    stub.placed = [(f"snap{p}", {("sig", "self"): ("x", live + p)}) for p in (0, 1)]
    instances = [{"rope": "0", "layer": f"model.layers.{i}.attn"} for i in (0, 1)]

    copies = [stub._copy(op, "sig", instances, c) for c in range(6)]

    weights = [m.weight for m in runner.model]
    assert [kw["layer"] for kw, _ in copies] == [
        "model.layers.0.attn",
        "model.layers.1.attn",
    ] * 3
    assert [snap for _, snap in copies] == [
        "snap0",
        "snap0",
        "snap1",
        "snap1",
        "snap0",
        "snap0",
    ]
    assert copies[0][0]["mat2"] is weights[0] and copies[1][0]["mat2"] is weights[1]
    assert torch.equal(copies[2][0]["mat2"], weights[0])
    assert copies[2][0]["mat2"].data_ptr() != weights[0].data_ptr()
    assert copies[3][0]["self"] is stub.placed[1][1][("sig", "self")][1]
    assert copies[4][0]["self"].data_ptr() != copies[0][0]["self"].data_ptr()
    assert copies[3][0]["rope"] is runner.model[0]
    assert copies[4][0]["rope"] is not runner.model[0]
    assert copies[0][0]["out_dtype"] is torch.bfloat16


@pytest.mark.parametrize("copies", [1, 8])
def test_every_cold_and_warm_graph_holds_calls_calls_at_any_copies(copies):
    # A replay's fixed cost is spread over a graph's calls: a count that moved
    # with the copies would move each price by its copies.
    stub = ops.Stub(_Runner(), 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]
    graphs = []

    def replay(fn, calls):
        graphs.append(len(calls))
        return [1.0] * ops.REPEATS

    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=["k"]),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        record = stub._time({"op": "x", "args": {"n": 1}}, "sig", [{}], copies)

    assert graphs == [ops.CALLS] * (2 if copies > 1 else 1)
    assert (record["copies"], record["calls"]) == (copies, ops.CALLS)


def test_a_batch_takes_the_blocks_and_slots_it_is_placed_at(rule):  # noqa: F811
    line = _line([[1, 40, True], [1, 17, True]])

    batch = rule("qwen3_next").batch(line, block=10, slot=3)

    assert [list(table) for table in batch.block_tables] == [[10, 11, 12], [13, 14]]
    assert batch.state_slots == [[3], [4]]
    with pytest.raises(ValueError, match="hold 15 KV blocks .* the pool has 14"):
        rule(blocks=14).batch(line, block=10)


@pytest.mark.parametrize(
    "argv, refusal",
    [
        (["--width", "4", "--context", "9", "--gpu", "0,1"], "names 2 GPUs"),
        (["--context", "9", "--gpu", "0"], "name a shape"),
        (["--width", "4", "--gpu", "0"], "a decode shape needs --context"),
    ],
)
def test_measure_ops_refuses_before_it_touches_a_gpu(tmp_path, argv, refusal):
    argv += ["--llc-mib", "256", "--store", str(tmp_path), "--out", str(tmp_path / "o")]

    with pytest.raises(SystemExit, match=f"^compass measure ops: .*{refusal}"):
        cli.main(["measure", "ops", *argv])
    assert not (tmp_path / "o").exists()


def test_a_measured_point_is_keyed_by_the_spec_of_the_platform_it_ran_on(
    tmp_path, monkeypatch
):
    spec = MachineSpec.from_mapping(
        json.loads((TREE / "scripts/compass/machines/mi308x-hjbog39.json").read_text())
    )
    stack = {c: spec.value(f"device.software_pinned_to.{c}") for c in PINNED}
    name, arch = spec.value("device.name"), spec.value("device.arch")
    run = {
        "device": [{"index": 6, "name": name, "arch": arch}],
        "stack": stack,
        "ptl": spec.values.get("device.ptl", "disabled"),
        "host_key": {"model": "sha256:" + "0" * 64, "host_cpu": "cpu"},
        "source": {"revision_kind": "tree", "revision": "a" * 40},
    }
    layout = {"dp": 1, "pp": 1, "pcp": 1, "tp": 1, "decode_context_parallel_size": 1}
    layout |= {"enable_expert_parallel": False, "enable_dp_attention": False}
    monkeypatch.setattr("atom.compass.run.memory.key_fields", lambda c: layout)
    monkeypatch.setattr("atom.compass.run.step_grid.keyed", lambda c: {"level": 0})
    store = ArtifactStore(str(tmp_path / "store"))
    key = ops.keys(None, run)
    measured = {"shape": "decode B=2", "context": 9, "leaf": "a", "cold_ns": 1.0}
    refused = measured | {"leaf": "b", "refused": "no"}

    ops.publish(
        store, key, None, run, "compass measure ops", [measured, refused], tmp_path
    )

    at = key("decode B=2", 9, "a")
    assert at.kind is Kind.OP_MEASUREMENT
    assert at.values["machine_spec"] == machine_spec_digest(spec)
    assert at.values["source_root"] == "tree-" + "a" * 40
    assert store.directory_for(at).exists()
    assert not store.directory_for(key("decode B=2", 9, "b")).exists()
    lines = [json.loads(x) for x in (tmp_path / "ops.jsonl").read_text().splitlines()]
    assert [x["token"] for x in lines] == [
        at.token(),
        key("decode B=2", 9, "b").token(),
    ]
