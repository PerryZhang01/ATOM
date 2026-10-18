# SPDX-License-Identifier: MIT
"""One decode step of a model, traced with no device into an `ir.Graph`.

ATOM's `ModelRunner` runs its own decode step through the model's own classes
under `TorchDispatchMode` over `FakeTensorMode(ShapeEnv())`, the Python
dispatcher on. The step's width is one free symbol, written `B`, set on the
`ScheduledBatch`; ATOM derives every other bound from it. `SymInt.__index__`
and `__int__` are `guard_int`, so a host fill that asks the width for a number
would pin `B` to the hint: the capture answers such an ask with the hint, logs
its ATOM line, and refuses a graph in which anything specialised.
"""

import contextlib
import pathlib
import re
import sys
from unittest.mock import patch

import torch
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

import atom
from atom.compass import ir

from . import driverless
from .graph import LAYER, Fork, Join, Traced, _json, build

#: The name the step's width is written under in every shape.
SYMBOL = "B"
#: Torch specialises a size hint of 0 or 1 to a constant, silently.
MIN_WIDTH = 2
#: The KV pool the step indexes; a block count, not a reading.
KV_BLOCKS = 64
#: A decode step's width: one query row per sequence, so one symbol for all four.
WIDTH_FIELDS = (
    "total_tokens_num",
    "total_tokens_num_decode",
    "total_seqs_num",
    "total_seqs_num_decode",
)
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

    def __init__(self, axis, scope, contexts):
        super().__init__()
        self.axis, self.scope, self.contexts = str(axis), scope, contexts
        self.layer, self.items, self.symbolic, self.on = None, [], False, False
        self.left_out = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
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
        return ir.SymDim(re.sub(rf"\b{self.axis}\b", SYMBOL, text), self.scope)


def _patches(recorder, log):
    """What the step runs under besides the modes, as one stack of patches.

    * A raw Triton or FlyDSL launch goes to the driver below the dispatcher: it
      is recorded as an op and not launched.
    * `CpuGpuBuffer.__init__` (ATOM's own) allocates its host side and that
      side's numpy view outside the fake mode, unpinned (pinning is a driver
      allocation); its device side is ATOM's `zeros_like`, faked.
    * A host ask of a symbolic count is answered with its hint and its ATOM
      line logged; `__bool__` is untouched, so a branch on the width guards.
      The block table's padded width is compared with concrete host rows.
    """
    import flydsl.compiler
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from triton.runtime.jit import JITFunction

    from atom.utils import CpuGpuBuffer
    from atom.utils.block_tables import BlockTableState

    def run(kernel, *args, **kwargs):
        leaves = [*args, *kwargs.values()]
        recorder.add(f"triton.{kernel.__name__}", ir.NodeKind.CAPTURED, leaves)

    def launch(kernel, *args):
        name = getattr(kernel, "func", kernel).__name__
        recorder.add(f"flydsl.{name}", ir.NodeKind.CAPTURED, args)

    init, zeros, numpy = CpuGpuBuffer.__init__, torch.zeros, torch.Tensor.numpy

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
    ):
        stack.enter_context(patch.object(owner, name, value))
    return stack


def _build_runner(config, fake_mode):
    """ATOM's `ModelRunner`, built in the mode in the model's dtype.

    The group already exists at its width, warmup would run a forward of its
    own, and no checkpoint is read.
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
        with fake_mode, patch.object(h2d, "_range", lambda t: (0, 0)):
            return _CapturedRunner(0, config)
    finally:
        torch.set_default_dtype(previous)


def _decode_batch(width, axis, block_size):
    """`width` one-token sequences, the width fields rebound to `axis`.

    `ScheduledBatch.__init__` compares its staged token array's length with the
    count, which would solve a symbol there, so it is built concrete.
    """
    import numpy as np

    from atom.model_engine import sequence
    from atom.model_engine.scheduler import ScheduledBatch

    seqs = {}
    for index in range(width):
        seq = sequence.Sequence([0], block_size=block_size, id=index)
        seq.status = sequence.SequenceStatus.RUNNING
        seq.type = sequence.SequenceType.DECODE
        seq.block_table = sequence.new_block_table([index])
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


def capture(model: str, tp: int, width: int, tmpdir: str):
    """Trace one decode step of `model` at TP `tp`, its width a symbol hinted `width`.

    Returns the graph, the model's config digest, and what the capture declared,
    the operators it left out of the graph among it.
    """
    if width < MIN_WIDTH:
        raise ValueError(
            f"--width {width} traces no symbol: torch specialises a size hint of "
            f"0 or 1; trace at >= {MIN_WIDTH} and evaluate at 1 afterwards"
        )
    declared = driverless.install(tp, tmpdir)

    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from atom.compass.run import _file_digest
    from atom.config import Config, set_current_atom_config

    config = Config(
        model=model, tensor_parallel_size=tp, load_dummy="empty", enforce_eager=True
    )
    set_current_atom_config(config)
    digest = _file_digest(config.model, "config.json")
    structure = f"{config.hf_config.architectures[0]} {digest} tp={tp} decode eager"
    shape_env = ShapeEnv()
    fake_mode = driverless.driverless_mode(shape_env)
    # Built outside the mode: a tensor made inside it is already static.
    with unset_fake_temporarily():
        template = torch.zeros(width, dtype=torch.int32)
    axis = fake_mode.from_tensor(template, static_shapes=False).shape[0]
    contexts = config.compilation_config.static_forward_context
    recorder, resolutions = _Recorder(axis, structure, contexts), []
    driverless.on_wait = recorder.wait
    with _patches(recorder, resolutions):
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
        batch = _decode_batch(width, axis, config.kv_cache_block_size)
        with fake_mode:
            runner.allocate_kv_cache(KV_BLOCKS)
            recorder.on = True
            with torch._C._EnablePythonDispatcher(), recorder:
                runner.forward(batch)
    if shape_env.replacements or not recorder.symbolic:
        specialised = [f"{k} = {v}" for k, v in shape_env.replacements.items()]
        raise RuntimeError(
            "the step's width specialised: "
            + (", ".join(specialised) or f"no recorded shape carries {SYMBOL}")
        )
    guards = {re.sub(rf"\b{axis}\b", SYMBOL, str(g.expr)) for g in shape_env.guards}
    declared["host_resolutions"] = resolutions
    crossing = []
    graph = build(recorder.items, Traced(structure, tuple(sorted(guards))), crossing)
    left_out = [*recorder.left_out, *(("step-crossing", op) for op in crossing)]
    declared["left_out"] = [{"why": why, **_json(op)} for why, op in left_out]
    return graph, digest, declared
