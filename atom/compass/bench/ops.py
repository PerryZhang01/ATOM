# SPDX-License-Identifier: MIT
"""`compass measure ops`: each leaf of an `op_graph` timed alone on one device, cold.

    compass measure ops (--width W | --prefill T B)... [--context C] --llc-mib M
        --gpu G --store S --out DIR -- <ATOM args>
    compass measure ops --plan RECORD [--plan RECORD]... --llc-mib M --gpu G
        --store S --out DIR -- <ATOM args>
    compass measure ops --coverage RECORD [--coverage RECORD]... --store S
        --out DIR -- <ATOM args>

Each shape names the `op_graph` `compass trace ops` published in S for the
model and TP width ATOM's arguments resolve. A point is one leaf of that graph,
its instances grouped (`leaves`), at the shape: a decode step's rows each hold
`--context` tokens, and a prefill's rows are fresh, split as the capture split
them.

ATOM's runner is built on the one GPU from ATOM's arguments, its weights a
dummy, through ATOM's warmup, and a `Stub` takes the place of the model's
forward. Each point reaches the runner as a batch (`ReplayRule`), so ATOM
builds its metadata and forward context, and the stub calls each leaf from its
contract: an activation is random at its shape, dtype and stride; a weight is
a tensor of the model of that shape, dtype and stride; a metadata tensor is a
view of ATOM's live metadata (`meta`), whose name the point records, or of
the token ids ATOM passes the forward (`TOKEN_IDS`); a module or context key
is the one each instance names. A leaf whose operator cannot run so refuses
by name, a KV tensor argument among them.

Cold, by copies. A call's bytes are its tensor arguments' spans, the tensors
of each module it names and, for a leaf that names a context key, the KV and
per-request state its layer holds for the batch. One graph runs the leaf K =
`CALLS` times, cycling over N copies, N the least whose bytes pass `COLD`
times the last-level cache, at least the model's weights of each weight
argument's spec and, for a context key, at least its I instances (at most K).
Copy c takes fresh activations,
the c-th of the model's weights of that spec (a clone past the last), and
instance c mod I at placement (c div I) mod P, its metadata and module
tensors cloned past the first round: a placement is the point's batch on KV
blocks and state slots of its own, for which ATOM builds the forward context
again. A second graph runs copy 0 K times, warm; with one copy there is none.
Each graph is replayed `WARMUPS` times, then `REPEATS` times timed;
a repeat is the replay's device time over K, and a graph that captured no
kernel times 0. The kernels one eager call of copy 0 launches are the point's
kernel identity. A batch the KV pool cannot place refuses its leaves.

DIR gets run.json, the run's identity, and ops.jsonl, one line per point. S
gets each measured point as an `op_measurement`, keyed by the model, the
parallel layout, the digest of the machine spec of the device, stack and PTL
state run.json reads, the source root and the point.

`--plan` measures, in rounds, the points the steps of record directories call
for (`record`'s or `trace steps`' output), capturing each `op_graph` S lacks:
every leaf at every rung of ATOM's decode ladder at the steps' lowest mean and
highest max row context; then each leaf of a kind those show depends on
context (`_dependent`) over a ladder (`_ladder`) of the steps' row contexts,
and each prefill family (`_family`) over a ladder of their token counts, chunk
remainders among them; then the point halfway between two neighbours of a
family that launched different kernels, until they are adjacent. DIR gets each
round's ops.jsonl in `round<n>`, and every point held in plan.jsonl.

`--coverage` counts the steps of record directories that DIR's plan.jsonl
covers (`coverage`), and prints the count by phase and why each leaf kind
left the rest out.
"""

import argparse
import bisect
import copy
import dataclasses
import datetime
import hashlib
import importlib
import itertools
import json
import multiprocessing
import os
import re
import statistics
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch

from atom.compass import ir
from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Provenance,
    Topology,
    member_name,
    roots_for,
)
from atom.compass.capture.forward import _weights
from atom.compass.drive.gpu import add_gpu_flag

PROG = "compass measure ops"
#: The timing protocol's version; it changes whenever a count below does.
PROTOCOL = 4
WARMUPS = 3
REPEATS = 20
#: Calls every graph holds, and so the copies it cycles over at most. One count
#: for every leaf: a replay's fixed cost adds the same share to every price.
CALLS = 256
#: Times over the last-level cache the copies' bytes pass: just past it, part of
#: them stays there, and an MI308X GEMM read up to 7% fast.
COLD = 2
#: Metadata arguments, by operator and slot, that are the token ids ATOM passes
#: the model's forward; `_meta`'s name rule binds an embedding's `indices` to
#: another index buffer.
TOKEN_IDS = {("aten.embedding.default", "indices")}
#: The rank every member is written at: the bench is one rank.
RANK = Topology().ranks()[0]


def main(argv: list[str]) -> None:
    from atom.compass.bench.measure import atom_config
    from atom.compass.capture.forward import decode, prefill
    from atom.compass.drive.gpu import hip_visible, quiet
    from atom.compass.drive.identity import identity
    from atom.compass.run.step_grid import keyed

    cut = argv.index("--") if "--" in argv else len(argv)
    if "--coverage" in argv[:cut]:
        return print(json.dumps(_coverage_main(argv[:cut], argv[cut + 1 :]), indent=1))
    args = _parser().parse_args(argv[:cut])
    if len(args.gpu) != 1:
        raise ValueError(f"--gpu names {len(args.gpu)} GPUs; the op bench uses one")
    shapes = [decode(w) for w in args.width] + [prefill(*s) for s in args.prefill]
    if not shapes and not args.plan:
        raise ValueError("name a shape: --width or --prefill, or --plan")
    if shapes and args.plan:
        raise ValueError("--plan places every point; name no --width or --prefill")
    if args.context is None and any(s.phase == "decode" for s in shapes):
        raise ValueError("a decode shape needs --context, each row's tokens")
    os.environ["HIP_VISIBLE_DEVICES"] = hip_visible(args.gpu)
    config = atom_config(argv[cut + 1 :])
    if config.tp_world_size > 1:
        raise ValueError(
            f"ATOM's arguments resolve {config.tp_world_size} ranks; the op bench "
            "starts one"
        )
    if config.compilation_config.level:
        raise ValueError(
            f"--level {config.compilation_config.level} compiles the model, whose "
            "forward the bench stubs; measure at --level 0"
        )
    store = ArtifactStore(args.store)
    todo = [
        (s, args.context, leaves(graph(store, config, s)["region"])) for s in shapes
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "engine.json").write_text(json.dumps(keyed(config)))
    command = [*PROG.split(), *argv]
    with quiet(args.gpu, out):
        run = identity(out / "engine.json", config.model, command, args.gpu)
        (out / "run.json").write_text(json.dumps(run))
        key, held = keys(config, run), 0
        if args.plan:
            return plan(store, key, config, run, args, " ".join(command), out)
        for shape, context, found in todo:
            for sig in list(found):
                if store.directory_for(key(shape, context, sig)).exists():
                    del found[sig]
                    held += 1
        print(f"{held} points already in the store", flush=True)
        todo = [point for point in todo if point[2]]
        results = _child(measure, config, todo, args.llc_mib << 20)
    publish(store, key, config, run, " ".join(command), results, out)
    refused = [
        f"{r['op']} at {r['shape']}: {r['refused']}" for r in results if "refused" in r
    ]
    print(f"{len(results) - len(refused)} of {len(results)} points measured")
    if refused:
        raise ValueError(f"{len(refused)} points refused:\n" + "\n".join(refused))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--width", type=int, action="append", default=[], help="a decode width"
    )
    p.add_argument(
        "--prefill",
        type=int,
        nargs=2,
        action="append",
        default=[],
        metavar=("T", "B"),
        help="a prefill of T tokens over B sequences",
    )
    p.add_argument("--context", type=int, help="each decode row's tokens")
    p.add_argument(
        "--plan",
        action="append",
        default=[],
        metavar="RECORD",
        help="a record directory whose steps the planned points cover",
    )
    p.add_argument(
        "--llc-mib",
        type=int,
        required=True,
        help="the device's last-level cache in MiB: 256 on MI300X, MI308X and "
        "MI355X (AMD Infinity Cache, CDNA 3 and CDNA 4 white papers)",
    )
    add_gpu_flag(p)
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument(
        "--out",
        required=True,
        help="where run.json and ops.jsonl go; with --plan, each round's "
        "ops.jsonl goes in round<n> and every point held in plan.jsonl",
    )
    return p


def graph(store, config, shape) -> dict:
    """The `op_graph` `trace ops` published for `config` at `shape`, as JSON."""
    from atom.compass.capture.forward import structure

    key = Key.of(
        Kind.OP_GRAPH, structure=structure(config, shape.phase), shape=str(shape)
    )
    return json.loads(store.read(key).read_member("graph", RANK, "json"))


def _ref(value) -> bool:
    return isinstance(value, dict) and value.keys() in ({"context"}, {"module"})


def signature(op: dict) -> str:
    """A leaf's identity across instances: its operator, arguments and outputs,
    a context key or module path by its kind alone."""
    args = {s: {k: None for k in v} if _ref(v) else v for s, v in op["args"].items()}
    leaf = {"op": op["op"], "args": args, "out": op["out"]}
    return json.dumps(leaf, sort_keys=True, separators=(",", ":"))


def leaves(region: dict, env=None, found=None) -> dict[str, tuple[dict, list]]:
    """Each leaf of a graph's JSON region by `signature`: one of its ops, and
    its instances in run order, each the key or path every reference slot binds."""
    env, found = env or {}, {} if found is None else found
    if "op" in region:
        bound = {}
        for slot, value in region["args"].items():
            if _ref(value):
                ref = ir.ContextRef(next(iter(value.values())))
                bound[slot] = ref.bind(**{n: env[n] for n in ref.index_names}).key
        instances = found.setdefault(signature(region), (region, []))[1]
        instances += [bound] if bound not in instances else []
    elif "repeat" in region:
        for i in range(region["count"]):
            index = {region["index"]: region["start"] + region["step"] * i}
            leaves(region["repeat"], {**env, **index}, found)
    else:
        for item in region.get("seq", region.get("par", ())):
            leaves(item, env, found)
    return found


def measure(config, todo, llc: int) -> list[dict]:
    """Each point's record, shape by shape: its timing, or why it refused."""
    from atom.compass.bench.replay import ReplayRule

    runner = _runner(config)
    try:
        stub, rule, got = Stub(runner, llc), ReplayRule(config), []
        with patch.object(runner.model, "forward", stub), torch.inference_mode():
            for shape, context, found in todo:
                try:
                    got += stub.point(rule, shape, context, found)
                except ValueError as error:  # a batch the KV pool cannot place
                    got += [
                        {"shape": str(shape), "context": context, "op": op["op"]}
                        | {"leaf": sig, "refused": f"ValueError: {error}"}
                        for sig, (op, _) in found.items()
                    ]
        return got
    finally:
        runner.exit()


def _runner(config):
    """ATOM's runner on the one visible GPU, warmed up, its KV pool allocated."""
    from torch.distributed import TCPStore

    from atom.model_engine.model_runner import ModelRunner

    pc = config.parallel_config
    store = TCPStore(
        pc.data_parallel_master_ip,
        pc.data_parallel_base_port,
        is_master=True,
        wait_for_workers=False,
    )
    pc.data_parallel_base_port, pc._managed_distributed_store = store.port, True
    runner = ModelRunner(0, config)
    runner.store = store
    config.num_kvcache_blocks = runner.get_num_blocks()["num_kvcache_blocks"]
    runner.allocate_kv_cache(config.num_kvcache_blocks)
    return runner


class Stub:
    """The model's forward in ATOM's runner, and the leaves timed beside it.

    Each call ATOM makes of the model snapshots the forward context ATOM built
    and the live metadata each leaf names, and returns zero hidden states.
    """

    def __init__(self, runner, llc: int) -> None:
        self.runner, self.llc = runner, llc
        self.weights, self.leaves, self.placed = {}, {}, []
        for t in _weights(runner.model):
            if t.device.type != runner.device.type:
                continue
            self.weights.setdefault(_spec(t), {})[t.data_ptr()] = t
        self.modules = dict(runner.model.named_modules())
        self.contexts = runner.config.compilation_config.static_forward_context

    def __call__(self, input_ids, positions, *args, **kwargs):
        from atom.utils.forward_context import get_forward_context

        fc = get_forward_context()
        live = {"input_ids": input_ids, "positions": positions}
        for name, buffer in self.runner.forward_vars.items():
            live[name] = getattr(buffer, "gpu", buffer)
        for owner in (fc.attn_metadata, fc.context):
            live |= {k: v for k, v in vars(owner).items() if torch.is_tensor(v)}
        device = self.runner.device.type
        live = {
            k: v
            for k, v in live.items()
            if torch.is_tensor(v) and v.device.type == device
        }
        meta = {}
        for sig, (op, _) in self.leaves.items():
            for path, spec in _specs(op["args"]):
                if spec["role"] == "metadata":
                    token = (op["op"], path) in TOKEN_IDS
                    pool = {"input_ids": input_ids} if token else live
                    try:
                        name, view = _meta(spec, path, pool)
                        meta[sig, path] = name, _clone(view)
                    except ValueError as error:  # the leaf refuses, not the step
                        meta[sig, path] = str(error), None
        memo: dict = {}
        snap = copy.copy(fc)
        snap.attn_metadata = _deep(fc.attn_metadata, memo)
        snap.context = _deep(fc.context, memo)
        self.placed.append((snap, meta))
        hidden = self.runner.config.hf_config.get_text_config().hidden_size
        dtype = self.runner.config.torch_dtype
        return input_ids.new_zeros((len(input_ids), hidden), dtype=dtype)

    def point(self, rule, shape, context, found) -> list[dict]:
        """Each leaf of `found` timed at `shape`, rows of `context` tokens."""
        from atom.utils.forward_context import _forward_kv_cache_context

        if shape.phase == "decode":
            rows = [[1, context, True]] * shape.seqs
        else:
            per, extra = divmod(shape.tokens, shape.seqs)
            rows = [[per + (i < extra)] * 2 + [False] for i in range(shape.seqs)]
        view = {"rows": rows, "produces_output": True}
        blocks = sum(-(-row[1] // rule.block_size) for row in rows)
        kv = _forward_kv_cache_context.kv_cache_data
        room = [self.runner.config.num_kvcache_blocks // blocks]
        room += [t.k_cache.shape[0] // len(rows) for t in kv.values() if _stateful(t)]
        plans = {
            s: self._plan(op, i, kv, blocks, len(rows)) for s, (op, i) in found.items()
        }
        need = max(p for _, _, p in plans.values())
        self.leaves, self.placed = found, []
        for p in range(max(min(need, *room), 1)):
            self.runner.forward(rule.batch(view, p * blocks, p * len(rows)))
        got = []
        for sig, (op, inst) in found.items():
            n, size, _ = plans[sig]
            names = {p: m[0] for (s, p), m in self.placed[0][1].items() if s == sig}
            print(f"{shape}: {op['op']} over {n} copies {names or ''}", flush=True)
            record = {"shape": str(shape), "context": context, "op": op["op"]}
            record |= {"leaf": sig, "metadata": names}
            try:
                record |= self._time(op, sig, inst, n) | {"footprint_bytes": n * size}
            except Exception as error:  # noqa: BLE001 - a refusal names what stopped it
                record["refused"] = f"{type(error).__name__}: {error}"
            got.append(
                record | {"instances": len(inst), "placements": len(self.placed)}
            )
        self.placed = []
        return got

    def _plan(self, op, instances, kv, blocks, rows):
        """A leaf's copies, its bytes per call, and the placements its context
        keys take."""
        need, size = 1, sum(
            _span(s["shape"], s["stride"]) * _dtype(s["dtype"]).itemsize
            for _, s in _specs(op["args"])
        )
        for key in instances[0].values():
            if key in self.contexts:
                held = kv.get(f"layer_{self.contexts[key].layer_num}")
                unit = rows if _stateful(held) else blocks
                # ponytail: a tensor's first dim is its blocks or its state slots
                size += sum(t.nbytes // t.shape[0] * unit for t in _fields(held))
                need = None
            elif key in self.modules:
                size += sum(t.nbytes for t in _own(self.modules[key]))
        n = min(CALLS, COLD * self.llc // size + 1) if size else 1
        # cycle over every instance's state and every weight of the model, all
        # held anyway: one weight can read a few percent off another of its spec
        held = [
            len(self.weights.get(_spec_of(s), ()))
            for _, s in _specs(op["args"])
            if s["role"] == "weight"
        ]
        n = min(CALLS, max([n, *held, len(instances) if need is None else 0]))
        return n, size, need or -(-n // len(instances))

    def _time(self, op, sig, instances, n) -> dict:
        """`op` timed cold over `n` copies and warm over one, with its kernels."""
        from atom.utils import forward_context

        fn = _function(op["op"])
        calls = [self._copy(op, sig, instances, c) for c in range(n)]
        try:
            forward_context._forward_context = calls[0][1]
            fn(**calls[0][0])  # compiles and tunes outside the graphs
            kernels = _kernels(fn, calls[0][0])
            cycle = [calls[i % n] for i in range(CALLS)]
            cold = _replay(fn, cycle)
            warm = _replay(fn, calls[:1] * len(cycle)) if n > 1 else None
        finally:
            forward_context._forward_context = forward_context.ForwardContext()
        medians = statistics.median(cold), warm and statistics.median(warm)
        return {
            "kernels": kernels,
            "cold_ns": medians[0],
            "warm_ns": medians[1],
            "warm_over_cold": medians[1] / medians[0] if warm and medians[0] else None,
            "cold_repeats": cold,
            "warm_repeats": warm,
            "copies": n,
            "calls": len(cycle),
            "llc_bytes": self.llc,
            "protocol": PROTOCOL,
        }

    def _copy(self, op, sig, instances, c):
        """Copy `c` of `op`'s call: its keyword arguments and forward context."""
        instance = instances[c % len(instances)]
        p = c // len(instances) % len(self.placed)
        snap, meta = self.placed[p]
        fresh = c >= len(self.placed) * len(instances)

        def build(path, value):
            if isinstance(value, list):
                return [build(f"{path}.{i}", v) for i, v in enumerate(value)]
            if _ref(value) and "module" in value:
                module = self.modules[instance[path]]
                return _module_copy(module) if fresh else module
            if _ref(value):
                return instance[path]
            if isinstance(value, dict):
                return self._tensor(value, meta.get((sig, path)), c, fresh)
            if isinstance(value, str) and value.startswith("torch."):
                return getattr(torch, value[6:], value)
            return value

        return {s: build(s, v) for s, v in op["args"].items()}, snap

    def _tensor(self, spec, meta, c, fresh):
        role = spec["role"]
        if role == "activation":
            return _random(spec, self.runner.device)
        if role == "metadata":
            if meta[1] is None:
                raise ValueError(meta[0])
            return _clone(meta[1]) if fresh else meta[1]
        if role == "weight":
            held = list(self.weights.get(_spec_of(spec), {}).values())
            if not held:
                raise ValueError(f"no tensor of the model is a weight {_text(spec)}")
            return held[c] if c < len(held) else _clone(held[c % len(held)])
        raise ValueError(f"a {role} tensor argument {_text(spec)} is not built")


def _specs(args, path=""):
    """Each tensor argument in `args`, by its slot path, `slot` or `slot.i`."""
    items = args.items() if isinstance(args, dict) else enumerate(args)
    for slot, value in items:
        where = f"{path}.{slot}" if path else str(slot)
        if isinstance(value, list):
            yield from _specs(value, where)
        elif isinstance(value, dict) and "role" in value:
            yield where, value


def _meta(spec, slot, live) -> tuple[str, torch.Tensor]:
    """The live metadata tensor a metadata argument is a view of, and its name:
    one of its dtype with room for its span: one whose name the slot holds or
    that holds the slot's first, then one of its shape and stride, then the
    tightest."""
    need = _span(spec["shape"], spec["stride"])
    fits = [
        (name, t)
        for name, t in live.items()
        if str(t.dtype) == spec["dtype"] and _room(t) >= need
    ]
    if not fits:
        raise ValueError(f"ATOM's live metadata has no room for {slot} {_text(spec)}")
    slot = slot.split(".")[0]
    name, t = min(
        fits,
        key=lambda f: (
            not (f[0] in slot or slot in f[0]),
            _spec(f[1]) != _spec_of(spec),
            _room(f[1]),
        ),
    )
    return name, t.as_strided(spec["shape"], spec["stride"], t.storage_offset())


def _function(name: str):
    """The callable a leaf names: `ns.op.overload`, or a lifted `module:qualname`."""
    if ":" in name:
        module, qualname = name.split(":")
        fn, path = importlib.import_module(module), qualname.split(".")
    else:
        namespace, rest = name.split(".", 1)
        fn, path = getattr(torch.ops, namespace), rest.split(".")
    for attr in path:
        fn = getattr(fn, attr)
    return fn


def _kernels(fn, kwargs) -> list[str]:
    """The device activity one call of `fn` launches, in order."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(**kwargs)
        torch.cuda.synchronize()
    device = [
        e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    return [e.name for e in sorted(device, key=lambda e: e.time_range.start)]


def _replay(fn, calls) -> list[float]:
    """Each timed replay of one graph of `calls`, in ns per call."""
    from atom.utils import forward_context

    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            for kwargs, snap in calls:
                forward_context._forward_context = snap
                fn(**kwargs)
    if any("Graph is empty" in str(w.message) for w in caught):
        return [0.0] * REPEATS
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for i in range(WARMUPS + REPEATS):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        if i >= WARMUPS:
            times.append(start.elapsed_time(end) * 1e6 / len(calls))
    return times


def _span(shape, stride) -> int:
    """Elements a view of `shape` and `stride` reaches from its offset."""
    if 0 in shape:
        return 0
    return 1 + sum((n - 1) * s for n, s in zip(shape, stride))


def _room(t) -> int:
    return t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()


def _dtype(text: str) -> torch.dtype:
    return getattr(torch, text.removeprefix("torch."))


def _spec(t) -> tuple:
    return tuple(t.shape), str(t.dtype), tuple(t.stride())


def _spec_of(spec: dict) -> tuple:
    return tuple(spec["shape"]), spec["dtype"], tuple(spec["stride"])


def _text(spec: dict) -> str:
    return f"{spec['shape']} {spec['dtype']} stride {spec['stride']}"


def _random(spec, device) -> torch.Tensor:
    """A tensor of `spec`'s shape, dtype and stride, random over its span when
    floating, else zero: an integer one may index a dimension of one."""
    dtype = _dtype(spec["dtype"])
    n = _span(spec["shape"], spec["stride"])
    if dtype.is_floating_point:
        flat = torch.randn(n, device=device).mul_(0.1).to(dtype)
    else:
        flat = torch.zeros(n, device=device, dtype=dtype)
    return flat.as_strided(spec["shape"], spec["stride"])


def _clone(t) -> torch.Tensor:
    """`t` on storage of its own, its span copied, its shape and stride kept."""
    n = _span(t.shape, t.stride())
    flat = t.as_strided((n,), (1,), t.storage_offset()).clone()
    return flat.as_strided(t.shape, t.stride())


def _deep(x, memo):
    """`x` with every tensor it reaches through dataclasses, ATOM objects,
    dicts and lists cloned, aliases kept."""
    if id(x) in memo:
        return memo[id(x)]
    if torch.is_tensor(x):
        memo[id(x)] = _clone(x)
    elif isinstance(x, dict):
        memo[id(x)] = {k: _deep(v, memo) for k, v in x.items()}
    elif isinstance(x, list):
        memo[id(x)] = [_deep(v, memo) for v in x]
    elif dataclasses.is_dataclass(x) or type(x).__module__.startswith("atom.utils"):
        if not hasattr(x, "__dict__"):
            return x
        memo[id(x)] = twin = copy.copy(x)
        vars(twin).update({k: _deep(v, memo) for k, v in vars(x).items()})
    else:
        return x
    return memo[id(x)]


def _own(module):
    """The tensors `module` holds itself: attributes, parameters and buffers."""
    held = [*vars(module).values(), *module._parameters.values()]
    return [t for t in [*held, *module._buffers.values()] if torch.is_tensor(t)]


def _module_copy(module):
    """`module` with its own tensors, parameters and buffers cloned."""
    twin = copy.copy(module)
    for name, value in vars(module).items():
        if torch.is_tensor(value):
            vars(twin)[name] = _clone(value)
        elif name in ("_parameters", "_buffers"):
            vars(twin)[name] = {
                k: v if v is None else _clone(v) for k, v in value.items()
            }
    return twin


def _stateful(held) -> bool:
    return bool(getattr(held, "per_request_state", False))


def _fields(held):
    for value in vars(held).values() if held is not None else ():
        if torch.is_tensor(value) and value.dim():
            yield value


def keys(config, run: dict):
    """The `op_measurement` key of a point, `(shape, context, leaf)`, measured
    under `config` on the platform `run` reads."""
    from atom.compass.artifacts.keys import LAYOUT, platform_key
    from atom.compass.fit.samples import run_fields
    from atom.compass.run.memory import key_fields

    fields = run_fields(run)
    platform = platform_key(fields["device"], run["stack"], run["ptl"]).canonical()
    layout = {k: v for k, v in key_fields(config).items() if k in LAYOUT}

    def key(shape, context, leaf) -> Key:
        point = json.dumps([str(shape), context, leaf]).encode()
        return Key.of(
            Kind.OP_MEASUREMENT,
            model=fields["model"],
            **layout,
            machine_spec="sha256:" + hashlib.sha256(platform.encode()).hexdigest(),
            source_root=fields["source_root"],
            point="sha256:" + hashlib.sha256(point).hexdigest(),
        )

    return key


def publish(store, key, config, run: dict, command: str, results, out: Path):
    """Write `results` to ``ops.jsonl`` in `out` and each measured one to
    `store` at its `key`."""
    from atom.compass.fit.publish import conditions
    from atom.compass.fit.samples import run_fields
    from atom.compass.run.step_grid import keyed

    readings = conditions(
        {**run_fields(run), "engine_config": keyed(config)}
        | {k: run[k] for k in ("stack", "host_key", "ptl")}
    )
    roots = roots_for({"atom": "atom", "aiter": "aiter"})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (out / "ops.jsonl").open("w") as lines:
        for record in results:
            at = key(record["shape"], record["context"], record["leaf"])
            lines.write(json.dumps(record | {"token": at.token()}) + "\n")
            if "refused" not in record:
                store.publish(
                    at,
                    provenance=Provenance(command, now, roots),
                    topology=Topology(),
                    conditions=readings,
                    gates=GateState.of(),
                    members={
                        member_name("point", RANK, "json"): json.dumps(record).encode()
                    },
                )


#: A ladder keeps a value only this many times the last it kept, and the largest.
LADDER_STEP = 1.25
#: The spread of a leaf's cold times across contexts, over the largest, past
#: which its kind depends on context; a repeat's spread stays within 4%.
CONTEXT_MOVES = 0.10


def plan(store, key, config, run, args, command: str, out: Path) -> None:
    """Measure the points the steps of `--plan`'s record directories call for,
    round by round, and write every point held to plan.jsonl in `out`."""
    from atom.compass.capture import population_steps
    from atom.compass.capture.forward import decode
    from atom.compass.runner.overrides import schedulable_ladder

    steps = list(population_steps(args.plan))
    contexts = [c for s, line in steps if s.phase == "decode" for c in _rows(line)]
    if not contexts:
        raise ValueError("the --plan records hold no decode step to read contexts of")
    rungs = [decode(r) for r in schedulable_ladder(config)]
    graphs, done, tried = {}, {}, set()

    def measure_round(want, n):
        dependent = _dependent(done.values())
        _load(store, config, args.store, {s for s, _ in want}, graphs)
        todo = []
        for (shape, context), families in want.items():
            found = {}
            for sig, (op, instances) in graphs[shape].items():
                fam = _family(shape, context, op["op"], sig, dependent)[0]
                if (
                    (str(shape), context, sig) in done
                    or families
                    and fam not in families
                ):
                    continue
                if store.directory_for(at := key(shape, context, sig)).exists():
                    held = store.read(at).read_member("point", RANK, "json")
                    done[str(shape), context, sig] = json.loads(held)
                else:
                    found[sig] = op, instances
            todo += [(shape, context, found)] if found else []
        print(f"round {n}: {sum(len(f) for *_, f in todo)} points to measure")
        results = _child(measure, config, todo, args.llc_mib << 20) if todo else []
        (out / f"round{n}").mkdir(exist_ok=True)
        publish(store, key, config, run, command, results, out / f"round{n}")
        done.update(((r["shape"], r["context"], r["leaf"]), r) for r in results)

    measure_round(
        {(s, c): None for s in rungs for c in (min(contexts), max(contexts))}, 0
    )
    dependent, want = _dependent(done.values()), defaultdict(set)
    for shape in rungs:
        for sig, (op, _) in graphs[shape].items():
            if dependent.get(op["op"], True):
                for c in _ladder(contexts):
                    want[shape, c].add(_family(shape, c, op["op"], sig, dependent)[0])
    prefills = {s for s, _ in steps if s.phase == "prefill"}
    _load(store, config, args.store, prefills, graphs)
    seen = defaultdict(dict)
    for shape in sorted(prefills):
        for sig, (op, _) in graphs[shape].items():
            fam, tokens = _family(shape, None, op["op"], sig, dependent)
            seen[fam].setdefault(tokens, shape)
    for fam, at in seen.items():
        for tokens in _ladder(at):
            want[at[tokens], None].add(fam)
    n = 1
    while want:
        measure_round(want, n)
        n += 1
        want = _densify(_families(done.values(), _dependent(done.values())), tried)
    (out / "plan.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in done.values())
    )
    kinds = Counter(r["op"] for r in done.values() if "refused" not in r)
    print(json.dumps({"points per leaf kind": dict(kinds.most_common())}, indent=1))
    print(f"{sum(kinds.values())} of {len(done)} points measured in {n} rounds")


def _coverage_main(argv: list[str], atom_args: list[str]) -> dict:
    from atom.compass.bench.measure import atom_config

    p = argparse.ArgumentParser(prog=f"{PROG} --coverage")
    p.add_argument("--coverage", action="append", required=True, metavar="RECORD")
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="the --plan run's --out")
    args = p.parse_args(argv)
    lines = (Path(args.out) / "plan.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    store = ArtifactStore(args.store)
    return coverage(store, args.store, atom_config(atom_args), args.coverage, records)


def coverage(store, root, config, names, records) -> dict:
    """How the measured points among `records` cover the steps of the record
    directories `names`: the steps and those covered by phase, and per leaf
    kind the steps it left out by why. A step is covered when each leaf of its
    `op_graph` lies on a point or between two of one kernel (`_bracket`): a
    decode leaf of a kind that depends on context over its rows' mean to max
    context (a padded or uneven step's `rowmod(k)` context can lie below the
    mean), any other decode leaf at its width, and a prefill leaf at its
    tokens, unless its kind depends on context and a row was cached: the
    bench's prefill rows are fresh."""
    from atom.compass.capture import population_steps

    steps = list(population_steps(names))
    dependent = _dependent(records)
    axes = {
        fam: ([p[0] for p in points], [p[1] for p in points])
        for fam, points in _families(records, dependent).items()
    }
    graphs, report, out = {}, defaultdict(Counter), defaultdict(Counter)
    _load(store, config, root, {s for s, _ in steps}, graphs)
    families = {
        s: [
            (op["op"], *_family(s, None, op["op"], sig, dependent))
            for sig, (op, _) in g.items()
        ]
        for s, g in graphs.items()
    }
    for shape, line in steps:
        whys, cached = {}, any(row[1] > row[0] for row in line["rows"])
        for name, fam, tokens in families[shape]:
            moves = dependent.get(name, True)
            if fam not in axes:
                whys[name] = "no measured point"
            elif shape.phase == "decode":
                whys[name] = _bracket(*axes[fam], *_rows(line)) if moves else ""
            elif cached and moves:
                whys[name] = "a cached row, measured on fresh rows only"
            else:
                whys[name] = _bracket(*axes[fam], tokens, tokens)
        whys = {name: why for name, why in whys.items() if why}
        report[shape.phase].update(steps=1, covered=not whys)
        for name, why in whys.items():
            out[name][why] += 1
    total = sum(report.values(), Counter())
    return {
        "steps": total["steps"],
        "covered": total["covered"],
        "share": total["covered"] / max(total["steps"], 1),
        "by phase": {phase: dict(c) for phase, c in report.items()},
        "left out, by leaf kind": {k: dict(v) for k, v in sorted(out.items())},
    }


def _rows(line) -> tuple[int, int]:
    """A step's rows' mean and max context."""
    contexts = [row[1] for row in line["rows"]]
    return sum(contexts) // len(contexts), max(contexts)


def _ladder(values) -> list:
    """`values` in order, each kept only `LADDER_STEP` past the last kept,
    and the largest."""
    kept = []
    for v in sorted(set(values)):
        if not kept or v >= kept[-1] * LADDER_STEP:
            kept.append(v)
    return kept + [v] if kept[-1] != v else kept


def _shape(text: str):
    from atom.compass.capture.forward import decode, prefill

    counts = [int(n) for n in re.findall(r"=(\d+)", text)]
    return prefill(*counts) if text.startswith("prefill") else decode(*counts)


def _abstract(value, n: int):
    """`value` with every int equal to `n` replaced by ``"T"``."""
    # ponytail: an unrelated int equal to n (a stride of 1 at T = 1) is
    # abstracted too, so that shape's leaf joins another family and its
    # points bracket nothing; symbolic slots in the contract would end it.
    if isinstance(value, dict):
        return {k: _abstract(v, n) for k, v in value.items()}
    if isinstance(value, list):
        return [_abstract(v, n) for v in value]
    return "T" if type(value) is int and value == n else value


def _family(shape, context, op: str, sig: str, dependent: dict):
    """The family a leaf's point belongs to, and where on its axis: a decode
    leaf at its width over context; a prefill leaf over its tokens, at its rows
    too when its kind depends on context."""
    if shape.phase == "decode":
        return json.dumps([str(shape), sig]), context
    rows = shape.seqs if dependent.get(op, True) else None
    leaf = _abstract(json.loads(sig), shape.tokens)
    return json.dumps(["prefill", rows, leaf], sort_keys=True), shape.tokens


def _dependent(records) -> dict[str, bool]:
    """Whether each leaf kind timed at two contexts of one width depends on
    context: its kernels or its cold times differ there."""
    by = defaultdict(list)
    for r in records:
        if "refused" not in r and r["shape"].startswith("decode"):
            by[r["op"], r["shape"], r["leaf"]].append(r)
    kinds = {}
    for (op, *_), points in by.items():
        if len(points) > 1:
            cold = [p["cold_ns"] for p in points]
            kernels = {tuple(p["kernels"]) for p in points}
            moved = max(cold) - min(cold) > CONTEXT_MOVES * max(cold)
            kinds[op] = kinds.get(op, False) or moved or len(kernels) > 1
    return kinds


def _families(records, dependent) -> dict:
    """Each family's measured points, `(at, kernels, record)` in `at` order."""
    found = defaultdict(list)
    for r in records:
        if "refused" not in r:
            shape = _shape(r["shape"])
            fam, at = _family(shape, r["context"], r["op"], r["leaf"], dependent)
            found[fam].append((at, tuple(r["kernels"]), r))
    return {fam: sorted(points, key=lambda p: p[0]) for fam, points in found.items()}


def _bracket(ats, kernels, lo, hi) -> str:
    """'' if `lo` to `hi` lies on or between two points, `ats` in order, that
    launched the same `kernels` as every point between them; else why not."""
    i, j = bisect.bisect_right(ats, lo) - 1, bisect.bisect_left(ats, hi)
    if i < 0 or j == len(ats):
        return "outside the measured range"
    if len(set(kernels[min(i, j) : max(i, j) + 1])) > 1:
        return "between two kernels"
    return ""


def _densify(families, tried: set) -> dict:
    """The point halfway between each two neighbours of a family that launched
    different kernels and lie more than one apart, unless `tried`."""
    from atom.compass.capture.forward import prefill

    want = defaultdict(set)
    for fam, points in families.items():
        for (a, ka, record), (b, kb, _) in itertools.pairwise(points):
            mid = (a + b) // 2
            if ka == kb or b - a < 2 or (fam, mid) in tried:
                continue
            tried.add((fam, mid))
            shape = _shape(record["shape"])
            at = (
                (shape, mid)
                if shape.phase == "decode"
                else (prefill(mid, shape.seqs), None)
            )
            want[at].add(fam)
    return want


def _load(store, config, root, shapes, graphs: dict) -> None:
    """Each of `shapes`' leaves into `graphs`, its op_graph captured first when
    the store has none."""
    from atom.compass.artifacts import ArtifactRefusal

    missing = []
    for shape in sorted(shapes - graphs.keys()):
        try:
            graphs[shape] = leaves(graph(store, config, shape)["region"])
        except ArtifactRefusal:
            missing.append(shape)
    if missing:
        _capture(config, missing, root)
    for shape in missing:
        graphs[shape] = leaves(graph(store, config, shape)["region"])


def _capture(config, shapes, root) -> None:
    """`trace ops` of `shapes` into the store at `root`, in a process of its
    own that sees no GPU."""
    from atom.compass import capture

    argv = ["--model", config.model, "--tp", str(config.tensor_parallel_size)]
    for s in shapes:
        argv += ["--prefill", str(s.tokens)] if s.phase == "prefill" else ["--width"]
        argv.append(str(s.seqs))
    with patch.dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES=""):
        _child(capture.main, [*argv, "--store", str(root)])


def _child(fn, *args):
    """`fn(*args)` in a spawned process of its own: the device is free again
    at its end, and a capture's process-wide patches end with it."""
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=spawn) as child:
        return child.submit(fn, *args).result()
