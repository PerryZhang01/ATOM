# SPDX-License-Identifier: MIT
"""Steps of a model, traced with no device into `ir.Graph`s, one per `Shape`.

ATOM's `ModelRunner` runs its own step through the model's own classes under
`TorchDispatchMode` over `FakeTensorMode`, the Python dispatcher on, at one
concrete shape: a decode width, or a prefill of T tokens over B sequences. One
runner steps through every shape of a capture, as it steps through a run.

Each leaf carries its contract: its arguments by schema slot, a tensor as its
shape, dtype, stride and role. A tensor's role is where its storage came from:
the model's parameters, buffers and tensor attributes are weights, what
`allocate_kv_cache` allocated is KV, what the model's forward and the sampler
after it allocated is an activation, and anything else (the runner's buffers,
what its input preparation allocated) is metadata.

A raw Triton or FlyDSL launch lifts to its innermost ATOM function: one leaf,
that function by its module and name, its parameters as the arguments, with
what else the function dispatched folded in. A function that reassigned a
parameter before the launch, or that dispatched a leaf of its own other than a
host query (`HOST_QUERIES`), is refused by name.

The engine is ATOM's own `Config` for ATOM's engine arguments, on a declared
device of one arch. Under DP the capture runs one rank and every peer runs its
shape (`peers`).
"""

import ast
import contextlib
import inspect
import pathlib
import re
import sys
import textwrap
from typing import NamedTuple
from unittest.mock import patch

import numpy
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

import atom
from atom.compass import ir

from . import driverless, peers
from .graph import LAYER, Fork, Join, Traced, _json, build

#: The fewest KV blocks a capture allocates; a shape that needs more gets more.
KV_BLOCKS = 64
#: Torch's own namespaces; an operator in any other hides its kernels.
TORCH_NAMESPACES = ("aten", "prim", "prims", "_c10d_functional", "c10d")
#: ATOM's H2D publication copies the runner's host-staged inputs to the device:
#: input preparation, not the step's device work, so its copies are left out.
H2D = "atom/utils/h2d.py"
#: Host queries a lifted function may dispatch: they launch nothing.
HOST_QUERIES = ("aiter.get_gfx_custom_op.default",)
#: Operators that ask a fake tensor where it is and run nothing on a device.
UNDISPATCHED = ("profiler.", "prim.device")

_ROOT = pathlib.Path(atom.__file__).resolve().parent.parent
_IS_ATOM: dict = {}


class Shape(NamedTuple):
    """A step's shape: `tokens` query tokens over `seqs` sequences."""

    phase: str  # "decode" or "prefill"
    tokens: int
    seqs: int

    def __str__(self) -> str:
        if self.phase == "prefill":
            return f"prefill T={self.tokens} B={self.seqs}"
        return f"decode B={self.seqs}"


def decode(width: int) -> Shape:
    if width < 1:
        raise ValueError(f"a decode width is at least 1, got {width}")
    return Shape("decode", width, width)


def prefill(tokens: int, seqs: int) -> Shape:
    if not 1 <= seqs <= tokens:
        raise ValueError(
            f"a prefill of {tokens} tokens over {seqs} sequences: each sequence "
            "takes at least one token"
        )
    return Shape("prefill", tokens, seqs)


def _atom_frames():
    """The ATOM frames on the stack, innermost first, Compass's own excluded."""
    frames, frame = [], sys._getframe(1)
    while frame:
        name = frame.f_code.co_filename
        if name not in _IS_ATOM:
            path = pathlib.Path(name)
            _IS_ATOM[name] = path.is_relative_to(
                _ROOT / "atom"
            ) and not path.is_relative_to(_ROOT / "atom" / "compass")
        if _IS_ATOM[name]:
            frames.append(frame)
        frame = frame.f_back
    return frames


def _where(frame) -> str:
    path = pathlib.Path(frame.f_code.co_filename).relative_to(_ROOT)
    return f"{path}:{frame.f_lineno} in {frame.f_code.co_qualname}"


def _storage(tensor) -> StorageWeakRef:
    return StorageWeakRef(tensor.untyped_storage())


def _reassigned(frame, names) -> list[str]:
    """The parameters `frame`'s function stores to before its current line."""
    lines, first = inspect.getsourcelines(frame.f_code)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    return sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in names
            and node.lineno + first - 1 < frame.f_lineno
        }
    )


class _Recorder(TorchDispatchMode):
    """Every dispatched operator and launched kernel, as `ir.Op`, by layer.

    What is not ATOM's work at the traced shape goes to `left_out`, with why.
    `creating` is the role a storage allocated now takes; `roles` holds every
    role given so far.
    """

    def __init__(self, contexts, fake_mode=None):
        super().__init__()
        self.contexts, self.modules, self.roles = contexts, {}, {}
        self.creating, self.on, self.fake_mode = None, False, fake_mode
        self.start()

    def start(self):
        self.layer, self.items, self.stacks, self.left_out = None, [], [], []
        self.lifting = None  # the frame the last item lifted to

    @contextlib.contextmanager
    def quiet(self):
        """Record nothing inside: a stand-in's own work is not ATOM's."""
        on, self.on = self.on, False
        try:
            yield
        finally:
            self.on = on

    def _fake(self, tensor):
        if not isinstance(tensor, FakeTensor):
            return self.fake_mode.from_tensor(tensor)
        tensor.constant = None
        return tensor

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device = kwargs.get("device")
        if device is not None and torch.device(device).type != "cpu":
            # A host array headed for the device, real or a fake holding its
            # values, is faked without them first, or the fake mode would copy
            # the values onto a device that is not there.
            args = _pytree.tree_map_only(torch.Tensor, self._fake, args)
        out = func(*args, **kwargs)
        leaves = _pytree.tree_leaves((args, kwargs))
        ins = [t for t in leaves if isinstance(t, torch.Tensor)]
        outs = [t for t in _pytree.tree_leaves(out) if isinstance(t, torch.Tensor)]
        if self.creating is not None:
            read = set(map(_storage, ins))
            for storage in map(_storage, outs):
                if storage not in read:
                    self.roles[storage] = self.creating
        name = str(func)
        if self.on and not name.startswith(UNDISPATCHED):
            torch_own = name.split(".")[0] in TORCH_NAMESPACES
            kind = ir.NodeKind.CAPTURED if torch_own else ir.NodeKind.OPAQUE_LEAF
            stack = _atom_frames()
            why = "h2d" if stack and _where(stack[0]).startswith(H2D) else None
            if ins + outs and all(t.device.type == "cpu" for t in ins + outs):
                why = "host"  # host arithmetic on host tensors runs no kernel
            if driverless.standing_in[0] and not name.startswith("_c10d_functional."):
                why = "stand-in"
            slots = {a.name: v for a, v in zip(func._schema.arguments, args)}
            self.add(name, kind, {**slots, **kwargs}, outs, why, stack)
        return out

    def add(self, name, kind, slots, outs, why, stack):
        op = self._op(name, kind, slots, [tuple(t.shape) for t in outs])
        if why:
            self.left_out.append((why, op))
        elif self.lifting is not None and any(f is self.lifting for f in stack):
            self._fold(op, self.lifting)
        else:
            self.lifting = None
            self.items.append((self.layer, op))
            self.stacks.append(stack)

    def launch(self, name):
        """A raw Triton or FlyDSL launch, lifted to its innermost ATOM function."""
        if not self.on:
            return
        stack = _atom_frames()
        if not stack:
            raise ValueError(f"{name} launches from no ATOM function to lift it to")
        frame = stack[0]
        if frame is self.lifting:  # another kernel of the function being lifted
            return
        code = frame.f_code
        count = code.co_argcount + code.co_kwonlyargcount
        count += bool(code.co_flags & inspect.CO_VARARGS)
        count += bool(code.co_flags & inspect.CO_VARKEYWORDS)
        params = code.co_varnames[:count]
        if reassigned := _reassigned(frame, params):
            raise ValueError(
                f"{name} lifts to {_where(frame)}, which reassigns its parameters "
                f"{reassigned} before the launch: they are not what it was called with"
            )
        while (
            self.items
            and self.items[-1][0] == self.layer
            and any(f is frame for f in self.stacks[-1])
        ):
            self.stacks.pop()
            self._fold(self.items.pop()[1], frame)
        lifted = f"{frame.f_globals['__name__']}:{code.co_qualname}"
        slots = {p: frame.f_locals[p] for p in params}
        op = self._op(lifted, ir.NodeKind.OPAQUE_LEAF, slots, ())
        self.items.append((self.layer, op))
        self.stacks.append([])
        self.lifting = frame

    def _fold(self, op, frame):
        if op.kind is ir.NodeKind.OPAQUE_LEAF and op.name not in HOST_QUERIES:
            raise ValueError(
                f"a launch lifts to {_where(frame)}, which also dispatches the leaf "
                f"{op.name}: one leaf cannot hold another"
            )

    def wait(self, waiter, waited):
        """A side stream waiting on the main one forks; the reverse joins."""
        if not self.on:
            return
        if waited is driverless.main_stream and waiter is not waited:
            item = Fork(waiter.stream_id)
        elif waiter is driverless.main_stream and waited not in (None, waiter):
            item = Join(waited.stream_id)
        else:
            return
        self.items.append((self.layer, item))
        self.stacks.append([])
        self.lifting = None

    def _op(self, name, kind, slots, out_shapes):
        """`name` as an `Op`, or a refusal naming it and what it cannot hold."""
        try:
            args = tuple((slot, self._arg(value)) for slot, value in slots.items())
            ref = next((v for _, v in args if type(v) is ir.ContextRef), None)
            stream = driverless.current[0].stream_id
            return ir.Op(name, kind, args, out_shapes, stream, ref)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} has no contract a graph can hold: {exc}"
            ) from None

    def _arg(self, value):
        if isinstance(value, torch.Tensor):
            role = self.roles.get(_storage(value), ir.Role.METADATA)
            shape, stride = tuple(value.shape), value.stride()
            return ir.Tensor(shape, str(value.dtype), stride, role)
        if isinstance(value, torch.nn.Module):
            if id(value) not in self.modules:
                raise TypeError(f"a {type(value).__name__} outside the model")
            return ir.ModuleRef(self._layered(self.modules[id(value)]))
        if isinstance(value, str) and value in self.contexts:
            return ir.ContextRef(self._layered(value))
        if isinstance(value, (list, tuple)):
            return tuple(self._arg(v) for v in value)
        if isinstance(value, (torch.dtype, torch.device, torch.layout)):
            return str(value)
        if isinstance(value, numpy.generic):
            return value.item()
        if isinstance(value, dict):
            return tuple((k, self._arg(v)) for k, v in value.items())
        if hasattr(value, "__dict__") and not callable(value):
            # A host object the engine built from its configuration, which the
            # graph's key names: its type stands for it.
            return f"<{type(value).__qualname__}>"
        return value

    def _layered(self, path):
        if self.layer is None:
            return path
        return re.sub(rf"\blayers\.{self.layer}\b", f"layers.{{{LAYER}}}", path)


def _patches(recorder):
    """What the step runs under besides the modes, as one stack of patches.

    * A raw Triton or FlyDSL launch goes to the driver below the dispatcher: it
      is lifted to its ATOM function and not launched.
    * `CpuGpuBuffer.__init__` (ATOM's own) allocates its host side and that
      side's numpy view outside the fake mode, unpinned (pinning is a driver
      allocation); its device side is ATOM's `zeros_like`, faked.
    * GDN's causal-conv1d metadata is host arithmetic over the staged
      `cu_seqlens`, and it runs outside the fake mode too.
    * A real host tensor's values are read on the host, `tolist` as `numpy`.
    * ATOM's MoE wrappers run their bodies, not their fakes: inside are the
      collectives, the streams and the expert kernels.
    * The checkpoint's weights iterator yields nothing: ATOM's dummy load still
      runs it, which would open every shard, or download them for a hub name.
    """
    import flydsl.compiler
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from triton.runtime.jit import JITFunction

    from atom.model_loader import loader
    from atom.model_ops import module_dispatch_ops, moe
    from atom.model_ops.attentions import gdn_attn
    from atom.utils import CpuGpuBuffer

    def run(kernel, *args, **kwargs):
        recorder.launch(f"triton.{kernel.__name__}")

    def launch(kernel, *args):
        recorder.launch(f"flydsl.{getattr(kernel, 'func', kernel).__name__}")

    init, zeros, numpy = CpuGpuBuffer.__init__, torch.zeros, torch.Tensor.numpy
    conv1d, tolist = gdn_attn.compute_causal_conv1d_metadata, torch.Tensor.tolist

    def host_tolist(tensor):
        if isinstance(tensor, FakeTensor):
            return tolist(tensor)
        with unset_fake_temporarily():
            return tolist(tensor)

    def host_zeros(*size, pin_memory=False, **kwargs):
        if torch.device(kwargs.get("device") or "cpu").type != "cpu":
            return zeros(*size, **kwargs)
        with unset_fake_temporarily():
            return zeros(*size, **kwargs)

    def host_numpy(tensor, *args, **kwargs):
        with unset_fake_temporarily():
            return numpy(tensor, *args, **kwargs)

    def staged_init(self, *args, **kwargs):
        with (
            patch.object(torch, "zeros", host_zeros),
            patch.object(torch.Tensor, "numpy", host_numpy),
        ):
            init(self, *args, **kwargs)

    def host_conv1d(*args, **kwargs):
        with unset_fake_temporarily():
            return conv1d(*args, **kwargs)

    stack = contextlib.ExitStack()
    for owner, name, value in (
        (JITFunction, "run", run),
        (flydsl.compiler, "compile", launch),
        (CpuGpuBuffer, "__init__", staged_init),
        (gdn_attn, "compute_causal_conv1d_metadata", host_conv1d),
        (torch.Tensor, "tolist", host_tolist),
        (loader, "safetensors_weights_iterator", lambda *a, **k: iter(())),
        (torch.ops.aiter, "moe_forward", moe.moe_forward),
        (
            torch.ops.aiter,
            "maybe_dual_stream_forward",
            module_dispatch_ops.maybe_dual_stream_forward,
        ),
    ):
        stack.enter_context(patch.object(owner, name, value))
    return stack


def _build_runner(config, fake_mode):
    """ATOM's `ModelRunner`, built in the mode in the model's dtype, which it
    leaves torch's default for the steps, as ATOM's runner does.

    The group already exists at its width, warmup would run a forward of its
    own, the checkpoint is ATOM's empty dummy (built and post-processed, no
    shard opened, under `_patches`), and the declared device's memory is all
    free.
    """
    from atom.model_engine.model_runner import DeviceMemoryReadings, ModelRunner
    from atom.utils import h2d

    class _CapturedRunner(ModelRunner):
        def _setup_device_and_distributed(self, rank, config):
            self.device = torch.device("cuda:0")

        def _maybe_warmup(self):
            return

        def _read_device_memory(self):
            total = torch.cuda.get_device_properties(0).total_memory
            return DeviceMemoryReadings(total, total, 0, 0)

    # A fake tensor has no address: the alias check sees each buffer empty.
    with fake_mode, patch.object(h2d, "_range", lambda t: (0, 0)):
        return _CapturedRunner(0, config)


def _weights(model):
    """Every tensor the model holds: parameters, buffers and tensor attributes."""
    for module in model.modules():
        held = [*module._parameters.values(), *module._buffers.values()]
        for value in [*held, *vars(module).values()]:
            if isinstance(value, torch.Tensor):
                yield value


def _batch(shape, block_size):
    """`shape.seqs` sequences of `shape.tokens` new tokens between them, the
    first `tokens % seqs` one longer, each in blocks of its own; and the
    blocks they take."""
    from atom.model_engine import sequence
    from atom.model_engine.scheduler import ScheduledBatch

    per, extra = divmod(shape.tokens, shape.seqs)
    counts = [per + (i < extra) for i in range(shape.seqs)]
    seqs, blocks = {}, 0
    for index, count in enumerate(counts):
        # An id of 0 would draw one from the class counter instead.
        seq = sequence.Sequence([0] * count, block_size=block_size, id=index + 1)
        seq.status = sequence.SequenceStatus.RUNNING
        seq.type = sequence.SequenceType[shape.phase.upper()]
        taken = -(-count // block_size)
        seq.block_table = sequence.new_block_table(list(range(blocks, blocks + taken)))
        seqs[seq.id], blocks = seq, blocks + taken
    batch = ScheduledBatch(
        seqs=seqs,
        num_scheduled_tokens=numpy.array(counts, dtype=numpy.int32),
        total_tokens_num=shape.tokens,
        total_seqs_num=shape.seqs,
        **{f"total_tokens_num_{shape.phase}": shape.tokens},
        **{f"total_seqs_num_{shape.phase}": shape.seqs},
        is_dummy_run=True,
    )
    return batch, blocks


def _config(model, tp, atom_args):
    """ATOM's `Config` for its engine arguments `atom_args`, eager, at level 0,
    the checkpoint an empty dummy.

    Under `--enable-dp-attention` each TP rank is a DP rank of TP 1, as
    `CoreManager` rewrites it.
    """
    import argparse
    import dataclasses

    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser(prog="trace ops --"))
    forced = ["--enforce-eager", "--level", "0", "--load_dummy", "empty"]
    args = parser.parse_args(["--model", model, "-tp", str(tp), *atom_args, *forced])
    engine = EngineArgs.from_cli_args(args)
    names = {field.name for field in dataclasses.fields(Config)}
    kwargs = engine._get_engine_kwargs().items()
    config = Config(engine.model, **{k: v for k, v in kwargs if k in names})
    if config.enable_dp_attention:
        parallel = config.parallel_config
        parallel.data_parallel_size *= config.tensor_parallel_size
        parallel.data_parallel_size_local = parallel.data_parallel_size
        config.tensor_parallel_size = 1
    return config


def capture(model: str, tp: int, shapes, tmpdir: str, atom_args=(), arch="gfx942"):
    """Trace one step of `model` at TP `tp` at each of `shapes`, in this process,
    under ATOM's engine arguments `atom_args` on a declared `arch`.

    Yields, per shape, the graph, the model's config digest, what the capture
    declared, the operators it left out of the graph among it.
    """
    declared = driverless.install(tmpdir, arch)

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from atom.compass.run import _file_digest
    from atom.config import set_current_atom_config

    config = _config(model, tp, atom_args)
    tbo = {"prefill": config.enable_tbo, "decode": config.enable_tbo_decode}
    if refused := [str(shape) for shape in shapes if tbo[shape.phase]]:
        raise ValueError(
            f"{', '.join(refused)} under --enable-tbo: a TBO step runs two "
            "micro-batches interleaved, which one op_graph cannot hold"
        )
    tp, dp = config.tensor_parallel_size, config.parallel_config.data_parallel_size
    ep = int(config.enable_expert_parallel)
    declared["group"] = driverless.group(tp, dp)
    set_current_atom_config(config)
    digest = _file_digest(config.model, "config.json")
    structure = (
        f"{config.hf_config.architectures[0]} {digest} tp={tp} dp={dp} ep={ep} "
        f"{{}} eager {arch}"
    )
    fake_mode = driverless.driverless_mode(ShapeEnv())
    contexts = config.compilation_config.static_forward_context
    recorder, block_size = _Recorder(contexts, fake_mode), config.kv_cache_block_size
    driverless.on_wait = recorder.wait
    stack = _patches(recorder)
    if dp > 1:
        for owner, name, value in peers.patches(recorder):
            stack.enter_context(patch.object(owner, name, value))
        if ep:
            peers.install_mori(recorder)
    with stack:
        runner = _build_runner(config, fake_mode)
        stacks = [
            module
            for name, module in runner.model.named_modules()
            if isinstance(module, torch.nn.ModuleList) and name.endswith("layers")
        ]
        if not stacks:
            raise ValueError(f"{type(runner.model).__name__} has no `layers` to fold")
        for index, layer in enumerate(max(stacks, key=len)):
            layer.register_forward_pre_hook(
                lambda m, a, i=index: setattr(recorder, "layer", i)
            )
            layer.register_forward_hook(
                lambda m, a, o: setattr(recorder, "layer", None)
            )
        runner.model.register_forward_pre_hook(
            lambda m, a: setattr(recorder, "creating", ir.Role.ACTIVATION)
        )
        recorder.modules = {id(m): n for n, m in runner.model.named_modules()}
        weights = map(_storage, _weights(runner.model))
        recorder.roles = dict.fromkeys(weights, ir.Role.WEIGHT)
        recorder.creating = ir.Role.KV
        blocks = max(_batch(shape, block_size)[1] for shape in shapes)
        with fake_mode:
            runner.get_num_blocks()  # sizes the pools, as `EngineCore` does
        # A warmup copy whose bytes nobody reads, like `_maybe_warmup`.
        runner.attn_metadata_builder.warmup_per_req_cache = lambda: None
        with fake_mode, recorder:
            runner.allocate_kv_cache(max(KV_BLOCKS, blocks))
        for shape in shapes:
            recorder.creating = ir.Role.METADATA
            with fake_mode, torch._C._EnablePythonDispatcher(), recorder:
                # A runner's first step takes its own path, so the traced step
                # follows one of its own shape: no graph depends on another's.
                runner.forward(_batch(shape, block_size)[0])
                recorder.start()
                recorder.creating, recorder.on = ir.Role.METADATA, True
                runner.forward(_batch(shape, block_size)[0])
            recorder.creating, recorder.on = None, False
            crossing, traced = [], Traced(structure.format(shape.phase), str(shape))
            graph = build(recorder.items, traced, crossing)
            left_out = [*recorder.left_out, *(("step-crossing", op) for op in crossing)]
            declared["left_out"] = [{"why": why, **_json(op)} for why, op in left_out]
            yield graph, digest, dict(declared)
