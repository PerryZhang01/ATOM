# SPDX-License-Identifier: MIT
"""One step of a model, traced with no device into an `ir.Graph`.

ATOM's `ModelRunner` runs its own step through the model's own classes under
`TorchDispatchMode` over `FakeTensorMode(ShapeEnv())`, the Python dispatcher
on. A prefill step, and any step under DP, is traced at its concrete width,
which its key names. A decode step without DP keeps its width one free symbol,
written `B`, set on the `ScheduledBatch`; ATOM derives every other bound from
it. `SymInt.__index__` and `__int__` are `guard_int`, so a host fill that asks
the width for a number would pin `B` to the hint: the capture answers such an
ask with the hint, logs its ATOM line, and refuses a graph in which anything
specialised.
"""

import contextlib
import pathlib
import re
import sys
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

import atom
from atom.compass import ir

from . import driverless, peers
from .graph import LAYER, Fork, Join, Traced, _json, build

#: The name the step's width is written under in every shape.
SYMBOL = "B"
#: Torch specialises a size hint of 0 or 1 to a constant, silently.
MIN_WIDTH = 2
#: The KV pool the step indexes; a block count, not a reading.
KV_BLOCKS = 64
#: Torch's own namespaces; an operator in any other hides its kernels.
TORCH_NAMESPACES = ("aten", "prim", "prims", "_c10d_functional", "c10d")
#: ATOM's H2D publication copies with a count it took `operator.index` of, so
#: its copies are at the hint, describe one width, and are left out.
H2D = "atom/utils/h2d.py"

_ROOT = pathlib.Path(atom.__file__).resolve().parent.parent


def _site() -> str:
    """The innermost ATOM line on the stack, Compass's own excluded."""
    frame = sys._getframe(1)
    while frame:
        path = pathlib.Path(frame.f_code.co_filename)
        if path.is_relative_to(_ROOT / "atom") and not path.is_relative_to(
            _ROOT / "atom" / "compass"
        ):
            return (
                f"{path.relative_to(_ROOT)}:{frame.f_lineno} in {frame.f_code.co_name}"
            )
        frame = frame.f_back
    return "outside atom"


class _Recorder(TorchDispatchMode):
    """Every dispatched operator and launched kernel, as `ir.Op`, by layer.

    What is not ATOM's work at the traced width goes to `left_out`, with why.
    """

    def __init__(self, fake_mode, symbols, scope, contexts):
        super().__init__()
        self.fake_mode, self.symbols = fake_mode, symbols
        self.scope, self.contexts = scope, contexts
        self.layer, self.items, self.symbolic, self.on = None, [], False, False
        self.left_out = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        device = (kwargs or {}).get("device")
        if device is not None and torch.device(device).type != "cpu":
            # A host array headed for the device, real or a fake holding its
            # values, is faked without them first, or the fake mode would copy
            # the values onto a device that is not there.
            args = _pytree.tree_map_only(torch.Tensor, self._fake, args)
        out = func(*args, **(kwargs or {}))
        name = str(func)
        if not name.startswith("profiler."):
            torch_own = name.split(".")[0] in TORCH_NAMESPACES
            kind = ir.NodeKind.CAPTURED if torch_own else ir.NodeKind.OPAQUE_LEAF
            leaves = _pytree.tree_leaves((args, kwargs))
            why = "h2d" if _site().startswith(H2D) else None
            if driverless.standing_in[0] and not name.startswith("_c10d_functional."):
                why = "stand-in"
            self.add(name, kind, leaves, _pytree.tree_leaves(out), why)
        return out

    def _fake(self, tensor):
        if not isinstance(tensor, FakeTensor):
            return self.fake_mode.from_tensor(tensor)
        tensor.constant = None
        return tensor

    def add(self, name, kind, inputs, outputs=(), why=None):
        if not self.on:  # the forward only, not the build before it
            return
        ins = [t for t in inputs if isinstance(t, torch.Tensor)]
        outs = [t for t in outputs if isinstance(t, torch.Tensor)]
        key = next(
            (s for s in inputs if isinstance(s, str) and s in self.contexts), None
        )
        if key is not None and self.layer is not None:
            key = re.sub(rf"\blayers\.{self.layer}\b", f"layers.{{{LAYER}}}", key)
        op = ir.Op(
            name=name,
            kind=kind,
            in_shapes=[self._shape(t) for t in ins],
            out_shapes=[self._shape(t) for t in outs],
            attrs={
                "in_dtypes": tuple(str(t.dtype) for t in ins),
                "out_dtypes": tuple(str(t.dtype) for t in outs),
            },
            stream_id=driverless.current[0].stream_id,
            context_ref=key and ir.ContextRef(key),
        )
        if why:
            self.left_out.append((why, op))
        else:
            self.items.append((self.layer, op))

    @contextlib.contextmanager
    def quiet(self):
        """Record nothing inside: a stand-in's own work is not ATOM's."""
        on, self.on = self.on, False
        try:
            yield
        finally:
            self.on = on

    def wait(self, waiter, waited):
        """A side stream waiting on the main one forks; the reverse joins."""
        if not self.on:
            return
        if waited is driverless.main_stream and waiter is not waited:
            self.items.append((self.layer, Fork(waiter.stream_id)))
        elif waiter is driverless.main_stream and waited not in (None, waiter):
            self.items.append((self.layer, Join(waited.stream_id)))

    def _shape(self, tensor):
        return tuple(self._dim(dim) for dim in tensor.shape)

    def _dim(self, dim):
        text = str(dim)
        if re.fullmatch(r"\d+", text):
            return int(text)
        self.symbolic = True
        return ir.SymDim(_named(text, self.symbols), self.scope)


def _patches(recorder, log):
    """What the step runs under besides the modes, as one stack of patches.

    * A raw Triton or FlyDSL launch goes to the driver below the dispatcher: it
      is recorded as an op and not launched.
    * `CpuGpuBuffer.__init__` (ATOM's own) allocates its host side and that
      side's numpy view outside the fake mode, unpinned (pinning is a driver
      allocation); its device side is ATOM's `zeros_like`, faked.
    * A real host tensor's values are read on the host, `tolist` as `numpy`.
    * ATOM's MoE wrappers run their bodies, not their fakes: inside are the
      collectives, the streams and the expert kernels.
    * A host ask of a symbolic count is answered with its hint and its ATOM
      line logged; `__bool__` is untouched, so a branch on the width guards.
      The block table's padded width is compared with concrete host rows.
    """
    import flydsl.compiler
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from triton.runtime.jit import JITFunction

    from atom.model_ops import module_dispatch_ops, moe
    from atom.utils import CpuGpuBuffer
    from atom.utils.block_tables import BlockTableState

    def run(kernel, *args, **kwargs):
        leaves = [*args, *kwargs.values()]
        recorder.add(f"triton.{kernel.__name__}", ir.NodeKind.CAPTURED, leaves)

    def launch(kernel, *args):
        name = getattr(kernel, "func", kernel).__name__
        recorder.add(f"flydsl.{name}", ir.NodeKind.CAPTURED, args)

    init, zeros, numpy = CpuGpuBuffer.__init__, torch.zeros, torch.Tensor.numpy
    tolist = torch.Tensor.tolist

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

    dual_stream = module_dispatch_ops.maybe_dual_stream_forward
    index, prepare = torch.SymInt.__index__, BlockTableState.prepare

    def resolved(self):
        if self.node.hint is None:  # unbacked: torch raises rather than guess
            return index(self)
        log.append(_site())
        return int(self.node.hint)

    def concrete_pad(self, rows, *, pad_to=None, **kwargs):
        pad_to = int(pad_to) if isinstance(pad_to, torch.SymInt) else pad_to
        return prepare(self, rows, pad_to=pad_to, **kwargs)

    stack = contextlib.ExitStack()
    for owner, name, value in (
        (JITFunction, "run", run),
        (flydsl.compiler, "compile", launch),
        (CpuGpuBuffer, "__init__", staged_init),
        (torch.SymInt, "__index__", resolved),
        (torch.SymInt, "__int__", resolved),
        (torch.SymInt, "__hash__", lambda self: hash(resolved(self))),
        (BlockTableState, "prepare", concrete_pad),
        (torch.Tensor, "tolist", host_tolist),
        (torch.ops.aiter, "moe_forward", moe.moe_forward),
        (torch.ops.aiter, "maybe_dual_stream_forward", dual_stream),
    ):
        stack.enter_context(patch.object(owner, name, value))
    return stack


def _build_runner(config, fake_mode):
    """ATOM's `ModelRunner`, built in the mode in the model's dtype.

    The group already exists at its width, warmup would run a forward of its
    own, no checkpoint is read, and the declared device is empty.
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

    previous = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    try:
        # A fake tensor has no address: the alias check sees each buffer empty.
        with fake_mode, patch.object(h2d, "_range", lambda t: (0, 0)):
            return _CapturedRunner(0, config)
    finally:
        torch.set_default_dtype(previous)


def _engine_args(model, tp, atom_args):
    """ATOM's `EngineArgs` for its own arguments, eager, at level 0, no checkpoint."""
    import argparse

    from atom.model_engine.arg_utils import EngineArgs

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser(prog="trace ops --"))
    forced = ["--enforce-eager", "--level", "0", "--load_dummy", "empty"]
    args = parser.parse_args(["--model", model, "-tp", str(tp), *atom_args, *forced])
    return EngineArgs.from_cli_args(args)


def _config(engine):
    """ATOM's `Config` for `engine`.

    Under `--enable-dp-attention` each TP rank is a DP rank of TP 1, as
    `CoreManager` rewrites it.
    """
    import dataclasses

    from atom.config import Config

    names = {field.name for field in dataclasses.fields(Config)}
    kwargs = engine._get_engine_kwargs().items()
    config = Config(engine.model, **{k: v for k, v in kwargs if k in names})
    if config.enable_dp_attention:
        parallel = config.parallel_config
        parallel.data_parallel_size *= config.tensor_parallel_size
        parallel.data_parallel_size_local = parallel.data_parallel_size
        config.tensor_parallel_size = 1
    return config


def _batch(width, axis, block_size, prefill):
    """The step, its width fields rebound to `axis`.

    Decode: `width` one-token sequences. Prefill: one sequence of `width`
    tokens. `ScheduledBatch.__init__` compares its staged token array's length
    with the count, which would solve a symbol there, so it is built concrete.
    """
    import numpy as np

    from atom.model_engine import sequence
    from atom.model_engine.scheduler import ScheduledBatch

    kind = "prefill" if prefill else "decode"
    rows = [width] if prefill else [1] * width
    seqs = {}
    for index, tokens in enumerate(rows):
        seq = sequence.Sequence([0] * tokens, block_size=block_size, id=index)
        seq.status = sequence.SequenceStatus.RUNNING
        seq.type = sequence.SequenceType[kind.upper()]
        blocks = range(index, index + -(-tokens // block_size))
        seq.block_table = sequence.new_block_table(blocks)
        seqs[seq.id] = seq
    fields = {
        "total_tokens_num": width,
        f"total_tokens_num_{kind}": width,
        "total_seqs_num": len(rows),
        f"total_seqs_num_{kind}": len(rows),
    }
    batch = ScheduledBatch(
        seqs=seqs,
        num_scheduled_tokens=np.asarray(rows, dtype=np.int32),
        **fields,
        is_dummy_run=True,
    )
    for name, value in fields.items():
        if value == width:
            setattr(batch, name, axis)
    return batch


def capture(model, tp, width, tmpdir, atom_args=(), prefill=False, arch="gfx942"):
    """Trace one step of `model` under ATOM's `atom_args` on a declared `arch`.

    The step decodes `width` sequences, or prefills `width` tokens. A prefill,
    or a step under DP, is traced at `width`, which the key names; a decode
    without DP holds its width as a symbol hinted `width`. Returns the graph,
    the model's config digest, and what the capture declared, the operators it
    left out of the graph among it.
    """
    engine = _engine_args(model, tp, atom_args)
    concrete = prefill or engine.enable_dp_attention or engine.data_parallel_size > 1
    if not concrete and width < MIN_WIDTH:
        raise ValueError(
            f"--width {width} traces no symbol: torch specialises a size hint of "
            f"0 or 1; trace at >= {MIN_WIDTH} and evaluate at 1 afterwards"
        )
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from atom.compass.run import _file_digest
    from atom.config import set_current_atom_config

    declared = driverless.install(tmpdir, arch)
    config = _config(engine)
    tp, dp = config.tensor_parallel_size, config.parallel_config.data_parallel_size
    declared["group"] = driverless.group(tp, dp)
    set_current_atom_config(config)
    digest = _file_digest(config.model, "config.json")
    if prefill:
        phase = f"prefill T={width} B=1"
    else:
        phase = f"decode B={width}" if concrete else "decode"
    ep = int(config.enable_expert_parallel)
    structure = (
        f"{config.hf_config.architectures[0]} {digest} tp={tp} dp={dp} ep={ep} "
        f"{phase} eager {arch}"
    )
    shape_env = ShapeEnv()
    fake_mode = driverless.driverless_mode(shape_env)
    axis = width
    if not concrete:
        # Built outside the mode: a tensor made inside it is already static.
        with unset_fake_temporarily():
            template = torch.zeros(width, dtype=torch.int32)
        axis = fake_mode.from_tensor(template, static_shapes=False).shape[0]
    symbols = {} if concrete else {str(axis): SYMBOL}
    contexts = config.compilation_config.static_forward_context
    recorder = _Recorder(fake_mode, symbols, structure, contexts)
    resolutions = []
    driverless.on_wait = recorder.wait
    stack = _patches(recorder, resolutions)
    if dp > 1:
        # Every peer scheduled this rank's rows: the MoE sees all of them.
        moe_rows = dp * width
        for owner, name, value in peers.patches(recorder, width, moe_rows):
            stack.enter_context(patch.object(owner, name, value))
        if ep:
            peers.install_mori(recorder, moe_rows)
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
        batch = _batch(width, axis, config.kv_cache_block_size, prefill)
        with fake_mode:
            # What EngineCore does, at a pool of `KV_BLOCKS`.
            sizing = runner.get_num_blocks()
            config.pool_entries = sizing.get("pool_entries", {})
            config.pool_entries_per_req = sizing.get("pool_entries_per_req", {})
            # A warmup copy whose bytes nobody reads, like `_maybe_warmup`.
            runner.attn_metadata_builder.warmup_per_req_cache = lambda: None
            runner.allocate_kv_cache(KV_BLOCKS)
            recorder.on = True
            with torch._C._EnablePythonDispatcher(), recorder:
                runner.forward(batch)
    if not concrete and (shape_env.replacements or not recorder.symbolic):
        specialised = [f"{k} = {v}" for k, v in shape_env.replacements.items()]
        raise RuntimeError(
            "the step's width specialised: "
            + (", ".join(specialised) or f"no recorded shape carries {SYMBOL}")
        )
    guards = {_named(str(g.expr), symbols) for g in shape_env.guards}
    declared["host_resolutions"] = resolutions
    crossing = []
    graph = build(recorder.items, Traced(structure, tuple(sorted(guards))), crossing)
    left_out = [*recorder.left_out, *(("step-crossing", op) for op in crossing)]
    declared["left_out"] = [{"why": why, **_json(op)} for why, op in left_out]
    return graph, digest, declared


def _named(text, symbols):
    """`text` with each traced symbol written under its name."""
    return re.sub(r"\b[su]\d+\b", lambda m: symbols.get(m.group(0), m.group(0)), text)
