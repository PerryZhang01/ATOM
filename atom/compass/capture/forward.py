# SPDX-License-Identifier: MIT
"""One decode step of a model, traced with no device into an `ir.Graph`.

ATOM's own `ModelRunner` runs its own decode step through the model's own
classes under `TorchDispatchMode` over `FakeTensorMode(ShapeEnv())`, with the
Python dispatcher on. The step's width is one free symbol, rendered `B`, set
on the `ScheduledBatch`; ATOM derives every other bound from it.

`SymInt.__index__` and `__int__` are `guard_int`: a host fill or slice that
asks the width for a number would record `Eq(B, hint)` and make the graph
constant. The capture answers those asks with the hint and logs each with its
ATOM line instead, and refuses a graph in which anything specialised.
"""

import contextlib
import pathlib
import re
import sys
import unittest.mock

import torch
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

import atom
from atom.compass import ir

from . import driverless
from .graph import LAYER, Fork, Join, Traced, build

#: The name the step's width is written under in every shape.
SYMBOL = "B"
#: Torch specialises a size hint of 0 or 1 to a constant, silently.
MIN_WIDTH = 2
#: The KV pool the step indexes; a block count, not a reading.
KV_BLOCKS = 64
#: The batch fields that carry a decode step's width: one query row per
#: sequence, so one symbol.
WIDTH_FIELDS = (
    "total_tokens_num",
    "total_tokens_num_decode",
    "total_seqs_num",
    "total_seqs_num_decode",
)
#: Namespaces whose operators are torch's own; any other hides its kernels.
TORCH_NAMESPACES = ("aten", "prim", "prims", "_c10d_functional", "c10d")
#: ATOM's H2D publication checks each count with `operator.index` and copies
#: with the number, so its copies are at the hint and describe one width.
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
            name = frame.f_code.co_name
            return f"{path.relative_to(_ROOT)}:{frame.f_lineno} in {name}"
        frame = frame.f_back
    return "outside atom"


class _Recorder(TorchDispatchMode):
    """Every dispatched operator and launched kernel, as `ir.Op`, by layer."""

    def __init__(self, axis, scope, contexts):
        super().__init__()
        self.axis, self.scope, self.contexts = str(axis), scope, contexts
        self.layer = None
        self.items = []  # (layer index or None, Op | Fork | Join)
        self.symbolic = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        if not name.startswith("profiler.") and not _site().startswith(H2D):
            leaves = _pytree.tree_leaves((args, kwargs))
            kind = (
                ir.NodeKind.CAPTURED
                if name.split(".")[0] in TORCH_NAMESPACES
                else ir.NodeKind.OPAQUE_LEAF
            )
            self.add(name, kind, leaves, _pytree.tree_leaves(out))
        return out

    def add(self, name, kind, inputs, outputs=()):
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
            context_ref=None if key is None else ir.ContextRef(key),
        )
        self.items.append((self.layer, op))

    def wait(self, waiter, waited):
        """A side stream waiting on the main one forks; the reverse joins."""
        if waited is None or waiter is waited:
            return
        main = driverless.main_stream
        if waited is main:
            self.items.append((self.layer, Fork(waiter.stream_id)))
        elif waiter is main:
            self.items.append((self.layer, Join(waited.stream_id)))

    def _shape(self, tensor):
        return tuple(self._dim(dim) for dim in tensor.shape)

    def _dim(self, dim):
        text = str(dim)
        if re.fullmatch(r"\d+", text):
            return int(text)
        self.symbolic = True
        return ir.SymDim(re.sub(rf"\b{self.axis}\b", SYMBOL, text), self.scope)


@contextlib.contextmanager
def _launches(recorder):
    """Record a raw Triton or FlyDSL launch as an op, and do not launch it.

    It goes to the driver below the dispatcher, where a fake tensor has no
    address to hand it.
    """
    from triton.runtime.jit import JITFunction

    def run(kernel, *args, **kwargs):
        leaves = [*args, *kwargs.values()]
        recorder.add(f"triton.{kernel.__name__}", ir.NodeKind.CAPTURED, leaves)

    original = JITFunction.run
    JITFunction.run = run
    try:
        import flydsl.compiler as flydsl
    except ImportError:
        flydsl = None
    if flydsl is not None:
        compile_and_launch = flydsl.compile

        def launch(kernel, *args):
            name = getattr(kernel, "func", kernel).__name__
            recorder.add(f"flydsl.{name}", ir.NodeKind.CAPTURED, args)

        flydsl.compile = launch
    try:
        yield
    finally:
        JITFunction.run = original
        if flydsl is not None:
            flydsl.compile = compile_and_launch


@contextlib.contextmanager
def _staged_buffers():
    """Run ATOM's own `CpuGpuBuffer.__init__` with its host side kept real.

    Its host tensor and that tensor's numpy view are allocated outside the
    fake mode (a fake has no numpy view); pinning is a driver allocation and is
    dropped. The device side is ATOM's own `zeros_like`, faked.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    from atom.utils import CpuGpuBuffer

    original_init = CpuGpuBuffer.__init__
    real_zeros, real_numpy = torch.zeros, torch.Tensor.numpy

    def zeros(*size, **kwargs):
        kwargs.pop("pin_memory", None)
        device = kwargs.get("device")
        if device is not None and torch.device(device).type != "cpu":
            return real_zeros(*size, **kwargs)
        with unset_fake_temporarily():
            return real_zeros(*size, **kwargs)

    def numpy(tensor, *args, **kwargs):
        with unset_fake_temporarily():
            return real_numpy(tensor, *args, **kwargs)

    def staged_init(self, *args, **kwargs):
        torch.zeros, torch.Tensor.numpy = zeros, numpy
        try:
            original_init(self, *args, **kwargs)
        finally:
            torch.zeros, torch.Tensor.numpy = real_zeros, real_numpy

    CpuGpuBuffer.__init__ = staged_init
    try:
        yield
    finally:
        CpuGpuBuffer.__init__ = original_init


@contextlib.contextmanager
def _resolved_on_the_host(log):
    """Answer a host ask of a symbolic count with its hint, logging the ATOM line.

    `__bool__` is left alone, so a branch on the width still guards.
    """
    from atom.utils.block_tables import BlockTableState

    original = torch.SymInt.__index__, torch.SymInt.__int__, torch.SymInt.__hash__
    original_prepare = BlockTableState.prepare

    def resolved(self):
        hint = self.node.hint
        if hint is None:  # unbacked: torch raises rather than guess
            return original[0](self)
        log.append(_site())
        return int(hint)

    def prepare(self, rows, *, pad_to=None, **kwargs):
        # The padded width is compared with the host rows, which are concrete.
        if isinstance(pad_to, torch.SymInt):
            pad_to = int(pad_to)
        return original_prepare(self, rows, pad_to=pad_to, **kwargs)

    torch.SymInt.__index__ = torch.SymInt.__int__ = resolved
    torch.SymInt.__hash__ = lambda self: hash(resolved(self))
    BlockTableState.prepare = prepare
    try:
        yield
    finally:
        torch.SymInt.__index__, torch.SymInt.__int__, torch.SymInt.__hash__ = original
        BlockTableState.prepare = original_prepare


def _build_runner(config, fake_mode):
    """ATOM's `ModelRunner`, built in the mode in the model's dtype, reading nothing.

    The group already exists at its width, warmup runs a forward of its own,
    and no checkpoint is loaded.
    """
    from atom.model_engine.model_runner import ModelRunner
    from atom.utils import h2d

    class _CapturedRunner(ModelRunner):
        def _setup_device_and_distributed(self, rank, config):
            self.device = torch.device("cuda:0")

        def _build_and_load_model(self, built_class):
            self.model = built_class(config)
            torch.set_default_device(None)

        def _maybe_warmup(self):
            return

    previous = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    try:
        # A fake tensor has no address: the alias check sees each buffer empty.
        with fake_mode, unittest.mock.patch.object(h2d, "_range", lambda t: (0, 0)):
            return _CapturedRunner(0, config)
    finally:
        torch.set_default_dtype(previous)


def _decode_batch(width, axis, block_size):
    """`width` one-token sequences, the count fields rebound to `axis`.

    `ScheduledBatch.__init__` compares its staged token array's length with the
    count, which would solve a symbol there, so it is built concrete.
    """
    import numpy as np

    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import (
        Sequence,
        SequenceStatus,
        SequenceType,
        new_block_table,
    )

    seqs = {}
    for index in range(width):
        seq = Sequence([0], block_size=block_size, id=index)
        seq.status, seq.type = SequenceStatus.RUNNING, SequenceType.DECODE
        seq.block_table = new_block_table([index])
        seqs[seq.id] = seq
    batch = ScheduledBatch(
        seqs=seqs,
        num_scheduled_tokens=np.ones(width, dtype=np.int32),
        **dict.fromkeys(WIDTH_FIELDS, width),
        is_dummy_run=True,
    )
    for name in WIDTH_FIELDS:
        setattr(batch, name, axis)
    return batch


def _step_axis(fake_mode, width):
    """A free symbol with hint `width`, made from a tensor built outside the mode."""
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        template = torch.zeros(width, dtype=torch.int32)
    return fake_mode.from_tensor(template, static_shapes=False).shape[0]


def _decoder_layers(model):
    """The longest `layers` list in the model: its decoder stack."""
    stacks = [
        module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList) and name.split(".")[-1] == "layers"
    ]
    if not stacks:
        raise ValueError(f"{type(model).__name__} has no `layers` list to fold")
    return max(stacks, key=len)


def capture(model: str, tp: int, width: int, tmpdir: str):
    """Trace one decode step of `model` at TP `tp`, its width a symbol hinted `width`.

    Returns the graph, the model's config digest, and what the capture declared.
    """
    if width < MIN_WIDTH:
        raise ValueError(
            f"--width {width} traces no symbol: torch specialises a size hint of "
            f"0 or 1; trace at >= {MIN_WIDTH} and evaluate at 1 afterwards"
        )
    declared = driverless.install(tp, tmpdir)

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from atom.compass.run import _file_digest
    from atom.config import Config, set_current_atom_config

    config = Config(
        model=model, tensor_parallel_size=tp, load_dummy="empty", enforce_eager=True
    )
    set_current_atom_config(config)
    digest = _file_digest(config.model, "config.json")
    architecture = config.hf_config.architectures[0]
    structure = f"{architecture} {digest} tp={tp} decode eager"
    shape_env = ShapeEnv()
    fake_mode = driverless.driverless_mode(shape_env)
    resolutions = []
    with _staged_buffers():
        runner = _build_runner(config, fake_mode)
        axis = _step_axis(fake_mode, width)
        recorder = _Recorder(
            axis, structure, config.compilation_config.static_forward_context
        )
        driverless.on_wait = recorder.wait
        for index, layer in enumerate(_decoder_layers(runner.model)):
            layer.register_forward_pre_hook(
                lambda m, a, i=index: setattr(recorder, "layer", i)
            )
            layer.register_forward_hook(
                lambda m, a, o: setattr(recorder, "layer", None)
            )
        # Before the batch: its constructor already asks its counts for a number.
        with _resolved_on_the_host(resolutions):
            batch = _decode_batch(width, axis, config.kv_cache_block_size)
            with fake_mode:
                runner.allocate_kv_cache(KV_BLOCKS)
                with torch._C._EnablePythonDispatcher(), _launches(recorder), recorder:
                    runner.forward(batch)
    if shape_env.replacements or not recorder.symbolic:
        raise RuntimeError(
            "the step's width specialised: "
            + (
                ", ".join(f"{k} = {v}" for k, v in shape_env.replacements.items())
                or f"no recorded shape carries {SYMBOL}"
            )
        )
    guards = sorted(
        {re.sub(rf"\b{axis}\b", SYMBOL, str(g.expr)) for g in shape_env.guards}
    )
    graph = build(recorder.items, Traced(structure, tuple(guards)))
    declared["host_resolutions"] = resolutions
    return graph, digest, declared
