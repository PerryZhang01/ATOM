# SPDX-License-Identifier: MIT
"""`compass measure ops`: each leaf of an `op_graph` timed alone on one device, cold.

    compass measure ops GRAPH (--points FILE | --width W... --context C)
        --gpu G --store S --out DIR [--llc-mib M] -- <ATOM args>

GRAPH is an `op_graph` token in S. A point is one leaf, a value for each
symbol its shapes name, and the context in tokens each of the step's decode
rows holds: ``{"leaf": SIG, "bind": {"B": 4}, "context": 1024}``, SIG being
`signature`'s text for the leaf. `--points` reads a JSON list of them;
`--width` makes one per leaf at each width.

The graph holds each leaf's tensors, not the values ATOM passes beside them
(sizes, scalars, a kernel's grid), nor its ambient state (its layer's weights
and KV, the step's attention metadata). So the bench builds ATOM's runner from
the ATOM arguments on the one GPU, its weights a dummy, and runs ATOM's decode
step of width B whose rows each hold the point's context. Where that step
dispatches an operator or launches a Triton kernel whose name, shapes and
dtypes are the leaf's at the point, outside any other operator, the bench takes
that call and times it alone. A leaf the step does not reach so, a FlyDSL
launch among them, refuses by name, as does a call reading host memory or one
a graph cannot capture.

A call is timed by CUDA-graph replay: one graph runs it on N copies of its
tensors, N the least whose total passes the last-level cache (at most
`MAX_COPIES`), so each call reads data the previous ones evicted; a call that
names its layer's ambient state by key rotates the key over every layer the
graph binds it to. A second graph runs it N times on one copy, warm. Each graph
is replayed `WARMUPS` times, then `REPEATS` times timed, and a repeat's value is
the replay's device time over N. A graph that captured no kernel times 0.

DIR gets ``run.json``, the run's identity and the card's PTL state, and
``ops.jsonl``, one line per point: the point, its cold and warm medians and
repeats, warm over cold, the copies and their footprint, keyed by the graph's
token, the device and stack, PTL state and source root.
"""

import argparse
import functools
import json
import multiprocessing
import os
import statistics
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sympy
import torch
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.drive.gpu import add_gpu_flag

PROG = "compass measure ops"
#: The timing protocol's version; it changes whenever a count below does.
PROTOCOL = 1
WARMUPS = 3
REPEATS = 20
#: Calls one graph holds at most; a leaf whose copies would need more falls
#: short of the cache, and its footprint says by how much.
MAX_COPIES = 256
#: The symbol a decode capture writes the step's width as.
WIDTH = "B"


def main(argv: list[str]) -> None:
    cut = argv.index("--") if "--" in argv else len(argv)
    args = _parser().parse_args(argv[:cut])
    if len(args.gpu) != 1:
        raise ValueError(f"--gpu names {len(args.gpu)} GPUs; the op bench uses one")
    store = ArtifactStore(args.store)
    entry = store.read(store.key_from_token(Kind.OP_GRAPH, args.graph))
    rank = next(iter(Topology().ranks()))
    graph = json.loads(entry.read_member("graph", rank, "json"))
    instances = leaves(graph["region"])
    points = (
        json.loads(Path(args.points).read_text())
        if args.points
        else [
            {"leaf": sig, "bind": {WIDTH: w}, "context": args.context}
            for w in args.width
            for sig in instances
        ]
    )
    for point in points:
        concrete(point, instances)  # refuses a point the graph cannot answer
        for guard in graph["guards"]:
            if not sympy.sympify(guard).subs(point["bind"]):
                raise ValueError(f"{point['bind']} breaks the graph's guard {guard}")

    from atom.compass.bench.measure import atom_config
    from atom.compass.drive.gpu import quiet
    from atom.compass.drive.identity import identity
    from atom.compass.run.step_grid import keyed

    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu[0])
    config = atom_config(argv[cut + 1 :])
    if (mine := structure(config)) != graph["structure"]:
        raise ValueError(
            f"the graph traced {graph['structure']!r}, and ATOM's arguments "
            f"resolve {mine!r}"
        )
    if config.compilation_config.level:
        raise ValueError(
            f"--level {config.compilation_config.level} compiles the step, which "
            "fuses the operators the graph's leaves are; measure at --level 0"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "engine.json").write_text(json.dumps(keyed(config)))
    with quiet(args.gpu, out):
        command = [*PROG.split(), *argv]
        run = identity(out / "engine.json", config.model, command, args.gpu)
        run |= {"ptl": ptl(args.gpu[0]), "graph": args.graph}
        (out / "run.json").write_text(json.dumps(run))
        # Its own process, so the device is free again at quiet's last check.
        spawn = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(1, mp_context=spawn) as bench:
            job = bench.submit(measure, config, points, instances, args.llc_mib << 20)
            results = job.result()
    key = {k: run[k] for k in ("graph", "device", "stack", "ptl", "source")}
    with (out / "ops.jsonl").open("w") as f:
        for point, got in zip(points, results):
            f.write(json.dumps({**point, **got, "key": key}) + "\n")
    refused = [
        f"{p['leaf']} at {p['bind']}: {g['refused']}"
        for p, g in zip(points, results)
        if "refused" in g
    ]
    print(f"{len(points) - len(refused)} of {len(points)} points measured")
    if refused:
        raise ValueError(f"{len(refused)} points refused:\n" + "\n".join(refused))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    p.add_argument("graph", help="the op_graph's token in --store")
    how = p.add_mutually_exclusive_group(required=True)
    how.add_argument("--points", help="a JSON list of points")
    how.add_argument("--width", type=int, nargs="+", help="every leaf at each width")
    p.add_argument("--context", type=int, help="each decode row's tokens, --width's")
    p.add_argument(
        "--llc-mib",
        type=int,
        default=256,
        help="the device's last-level cache; MI300 and MI355 series: 256 MiB",
    )
    add_gpu_flag(p)
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="where run.json and ops.jsonl go")
    return p


def signature(op: dict) -> str:
    """A leaf's identity across instances: its operator, shapes and dtypes."""
    leaf = {k: op[k] for k in ("op", "in", "out", "attrs")}
    return json.dumps(leaf, sort_keys=True, separators=(",", ":"))


def leaves(region: dict, env=None, found=None) -> dict[str, list[str]]:
    """Each leaf of a graph's JSON region, by `signature`, to the context keys
    its instances bind, in run order."""
    from atom.compass import ir

    env, found = env or {}, {} if found is None else found
    if "op" in region:
        keys = found.setdefault(signature(region), [])
        if region["context"]:
            ref = ir.ContextRef(region["context"])
            key = ref.bind(**{n: env[n] for n in ref.index_names}).key
            keys += [key] if key not in keys else []
    elif "repeat" in region:
        for i in range(region["count"]):
            index = {region["index"]: region["start"] + region["step"] * i}
            leaves(region["repeat"], {**env, **index}, found)
    else:
        for item in region.get("seq", region.get("par", ())):
            leaves(item, env, found)
    return found


def concrete(point: dict, instances: dict) -> tuple:
    """The call `point`'s leaf makes at its binding: name, shapes and dtypes."""
    if point["leaf"] not in instances:
        raise ValueError(f"the graph has no leaf {point['leaf']}")
    if point.get("context") is None or point["bind"].get(WIDTH, 0) < 1:
        raise ValueError(f"{point} names no context or no width {WIDTH} >= 1")
    leaf = json.loads(point["leaf"])

    def dim(d):
        if isinstance(d, int):
            return d
        expr = sympy.sympify(d)
        if unbound := {str(s) for s in expr.free_symbols} - set(point["bind"]):
            raise ValueError(f"{point} binds no {sorted(unbound)}")
        return int(expr.subs(point["bind"]))

    shapes = [tuple(tuple(map(dim, s)) for s in leaf[k]) for k in ("in", "out")]
    dtypes = [tuple(leaf["attrs"][f"{k}_dtypes"]) for k in ("in", "out")]
    return (leaf["op"], *shapes, *dtypes)


def _call_key(name, inputs, outputs=()) -> tuple:
    ins = [t for t in inputs if isinstance(t, torch.Tensor)]
    outs = [t for t in outputs if isinstance(t, torch.Tensor)]
    shapes = [tuple(tuple(t.shape) for t in ts) for ts in (ins, outs)]
    return (name, *shapes, *[tuple(str(t.dtype) for t in ts) for ts in (ins, outs)])


class Harvest(TorchDispatchMode):
    """Offers each top-level call of a step to `time`, once per wanted key.

    `wanted` maps a call's `concrete` key to the points it answers; `time(fn,
    args, kwargs)` returns the result each of them gets. Calls made inside
    another operator, or by `time` itself, are not offered.
    """

    def __init__(self, wanted: dict, time) -> None:
        super().__init__()
        self.wanted, self.time, self.got, self.depth = wanted, time, {}, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.depth += 1
        try:
            out = func(*args, **kwargs)
        finally:
            self.depth -= 1
        self.offer(str(func), func, args, kwargs, _pytree.tree_leaves(out))
        return out

    def offer(self, name, fn, args, kwargs, outputs=()) -> None:
        if self.depth:
            return
        inputs = _pytree.tree_leaves((args, kwargs))
        points = self.wanted.pop(_call_key(name, inputs, outputs), ())
        if points:
            self.depth += 1
            try:
                got = self.time(fn, args, kwargs, points[0])
            finally:
                self.depth -= 1
            for point in points:
                self.got[id(point)] = got

    def triton(self):
        """`JITFunction.run` patched to offer each launch, as a context."""
        from unittest.mock import patch

        from triton.runtime.jit import JITFunction

        run = JITFunction.run

        def launch(kernel, *args, **kwargs):
            out = run(kernel, *args, **kwargs)
            fn = functools.partial(run, kernel)
            self.offer(f"triton.{kernel.__name__}", fn, args, kwargs)
            return out

        return patch.object(JITFunction, "run", launch)


def measure(config, points, instances, llc: int) -> list[dict]:
    """Each point's result, in order: its timing, or why it refused."""
    from atom.compass.bench.replay import ReplayRule

    runner = _runner(config)
    try:
        rule, got = ReplayRule(config), {}
        steps = defaultdict(list)
        for point in points:
            steps[point["bind"][WIDTH], point["context"]].append(point)
        for (width, context), todo in steps.items():
            view = {"rows": [[1, context, True]] * width, "produces_output": True}
            runner.forward(rule.batch(view))  # builds and tunes at this width
            wanted = defaultdict(list)
            for point in todo:
                wanted[concrete(point, instances)].append(point)
            time = functools.partial(_time, instances=instances, llc=llc)
            harvest = Harvest(wanted, time)
            with harvest.triton(), harvest:
                runner.forward(rule.batch(view))
            got |= harvest.got
        missed = (
            "ATOM's decode step at this width and context makes no top-level "
            "call of this leaf's name, shapes and dtypes"
        )
        return [got.get(id(p), {"refused": missed}) for p in points]
    finally:
        runner.exit()


def _runner(config):
    """ATOM's runner on the one visible GPU, its KV pool allocated."""
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


def _time(fn, args, kwargs, point, *, instances, llc) -> dict:
    """`fn(*args, **kwargs)` timed cold and warm, or why it refused."""
    tensors = [t for t in _pytree.tree_leaves((args, kwargs)) if torch.is_tensor(t)]
    if any(t.device.type != "cuda" for t in tensors):
        return {"refused": "it reads host memory, which a graph replay does not"}
    size = sum(
        {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for t in tensors
        }.values()
    )
    n = min(MAX_COPIES, llc // size + 1) if size else 2
    keys = instances[point["leaf"]]
    try:
        cold = copies(args, kwargs, n, keys)
        cold_ns, warm_ns = _replay(fn, cold), _replay(fn, cold[:1] * n)
    except Exception as error:  # noqa: BLE001 - a refusal names whatever stopped it
        return {"refused": f"{type(error).__name__}: {error}"}
    medians = statistics.median(cold_ns), statistics.median(warm_ns)
    return {
        "cold_ns": medians[0],
        "warm_ns": medians[1],
        "warm_over_cold": medians[1] / medians[0] if medians[0] else None,
        "cold_repeats": cold_ns,
        "warm_repeats": warm_ns,
        "copies": n,
        "footprint_bytes": n * size,
        "llc_bytes": llc,
        "instances": len(keys),
        "protocol": PROTOCOL,
    }


def copies(args, kwargs, n: int, keys: list[str]) -> list[tuple]:
    """`n` copies of `(args, kwargs)`: tensors on fresh storage, aliases kept,
    and a context key in `keys` rotated over `keys`."""
    swap = functools.partial(_swap, keys=keys)
    return [
        _pytree.tree_map(functools.partial(swap, {}, i), (args, kwargs))
        for i in range(n)
    ]


def _swap(clones: dict, i: int, x, *, keys):
    if torch.is_tensor(x):
        storage = x.untyped_storage()
        if storage.data_ptr() not in clones:
            clones[storage.data_ptr()] = storage.clone()
        view = torch.empty(0, dtype=x.dtype, device=x.device)
        fresh = clones[storage.data_ptr()]
        return view.set_(fresh, x.storage_offset(), x.shape, x.stride())
    return keys[i % len(keys)] if isinstance(x, str) and x in keys else x


def _replay(fn, calls) -> list[float]:
    """Each timed replay of one graph of `calls`, in ns per call."""
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            for args, kwargs in calls:
                fn(*args, **kwargs)
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


def structure(config) -> str:
    """The structure `compass trace ops` keys a decode graph of `config` by."""
    from atom.compass.run import _file_digest

    digest = _file_digest(config.model, "config.json")
    mode = "eager" if config.enforce_eager else "graph"
    arch = config.hf_config.architectures[0]
    return f"{arch} {digest} tp={config.tensor_parallel_size} decode {mode}"


def ptl(gpu: int) -> str:
    """The PTL state of rocm-smi's GPU `gpu`, `disabled` where it has none."""
    from atom.compass.drive.gpu import smi

    bus = json.loads(smi("--showbus", "--json"))[f"card{gpu}"]["PCI Bus"]
    path = Path("/sys/bus/pci/devices", bus.lower(), "ptl", "ptl_enable")
    return path.read_text().strip() if path.exists() else "disabled"
