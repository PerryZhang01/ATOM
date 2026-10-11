# SPDX-License-Identifier: MIT
"""One forward of the published Qwen3.8-27B, traced with no device, at TP1 and TP2.

This file is both the test and the capture driver. `pytest` runs the tests at
the bottom; each of them runs this same file as a script in a fresh interpreter
and reads the JSON record it prints. The subprocess is not isolation for its own
sake -- three pieces of the capture are process-global and one-shot, so two
widths cannot share an interpreter:

* `torch.distributed` is initialised once, at one world size;
* aiter's model-parallel state is module-global and asserts when re-entered;
* the `torch.cuda` and Triton substitutions below must be installed *before*
  `import atom`, and an import happens once per process.

Running it as a script also keeps the substitutions off every other test in the
tier: nothing here mutates `torch.cuda` in the pytest process.

What the capture is
-------------------
`TorchDispatchMode` over `FakeTensorMode(ShapeEnv())` with the Python dispatcher
enabled, around ATOM's own `ModelRunner` driving its own decode step through
ATOM's own model classes. No stub model and no synthetic config: the config is
the published `Qwen/Qwen3.8-27B` `config.json`, vendored beside this file and
checked byte-for-byte by sha256 before it is used.

**The inventories are DIAGNOSTIC, not captures.** Raw `@triton.jit` launches go
straight to the AMD driver and never enter the torch dispatcher, so
`FakeTensorMode` cannot fake them: the launcher asks a `FakeTensor` for its
`data_ptr` and the first kernel reached ends the trace. They are recorded and
not executed, which keeps the trace alive long enough to enumerate what a step
reaches -- but anything downstream of a skipped kernel read uninitialised fake
memory. The record says so in `diagnostic_inventory`, and these operator counts
are not a cost-model input at any width.

What is substituted, and what is not
------------------------------------
Substituted, all of it device facts declared by the caller rather than read from
a runtime: the `torch.cuda` namespace (17 names, each one read by a capture --
the record counts every read, see `_declare_cuda`); `rocminfo`, which aiter
shells out to at import and which needs `/dev/kfd`; Triton's active device
target; the collective *transport*, so
a group of width N needs no peer; and the three primitives `CpuGpuBuffer`
reaches for, because its two allocations and its numpy view have to straddle
the mode -- see `_staged_allocators`. What is *not* substituted there is the
constructor: ATOM's own `CpuGpuBuffer.__init__` executes, and the record
carries which one ran and what it allocated, so a change inside it cannot be
silent here.

Not substituted: the process group's width. The group here reports width 2
because it has width 2, and every collective ATOM issues is dispatched and
recorded. `apply_simulated_tp` is not what produced it: at one physical rank it
both erases and fabricates -- row-parallel `all_reduce` becomes the identity and
appears nowhere, while one `all_gather` becomes six dispatched ops over a
half-zeros tensor -- so a TP2 inventory taken through it is not a TP2
inventory. A sentinel sits over both bindings of it, but its one call site in
ATOM (held by a scan of ATOM's source, not assumed) is inside the method this
capture's runner overrides, so today the
sentinel is a guard over a path the capture does not execute rather than an
observation that ATOM at TP2 avoids it. What does observe the width is the
group's own `world_size` and the collectives it issued.

Where the shapes specialise, and what stopped it
------------------------------------------------
Three sites, in an order rather than a set. A free symbol is solved by whichever
line reaches it first, so repairing one does not always close what it solved:
closing the first moves the caller's bound into an ATOM assertion helper in
another file and a later phase of the step -- `forward_context.py`, reached
from the `run_model` call in `ModelRunner.forward`, with `prepare_inputs`
already returned. The plain symbolic pass records the first two; a third pass
simulates the first closed, from outside ATOM, and records what is behind it.
All three are pinned, and there is no claim that three is all there are -- only
that these three are what this instrument can reach today.

They are all symptoms of one line of torch: **`SymInt.__index__` and
`SymInt.__int__` are `guard_int`**. Every host-side use of the step's width
needs a number, gets the trace-time hint, and *records the ask as a guard that
makes the symbol a constant*. Sixteen ATOM lines do it in one forward, so
closing them one at a time does not converge.

The fourth pass, `--step-symbol`, is the one that specialises nowhere. It keeps
the conversion -- a host fill genuinely needs a number, and the number it needs
is the hint, which is the count ATOM computed -- and replaces the *recording*
with a log of every conversion and the ATOM line it happened on
(`_resolve_on_the_host`). That log is checked against a declared multiset,
`EXPECTED_HOST_RESOLUTIONS`, so a conversion at a line this capture did not
expect fails rather than joining the log unremarked. `__bool__` is untouched,
so a branch on a width still guards. It symbolises no staged buffer: their
dimensions are engine capacities, which is what the second site was really
about. The step's width is a symbol on the `ScheduledBatch` and ATOM derives
every bound from it -- with one exception, `ScheduledBatch.__init__` itself,
which solves the width by comparison rather than conversion, so the batch is
built at the concrete width and its four count fields are rebound afterwards
(see `_decode_batch`).

One line of ATOM changed for it, and only one: `assert_shape_contract`'s `_rows`
returns `t.shape[0]` rather than `int(t.shape[0])`. That is the third site, and
the only one of the three not reachable from a capture-time substitution.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import hashlib
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import traceback

import numpy
import torch
from torch.utils._python_dispatch import TorchDispatchMode

# The published checkpoint this config came from: `Qwen/Qwen3.8-27B` at
# revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. The file beside this one is
# that revision's `config.json`, byte for byte -- 4,312 bytes, sha256 below,
# which is also its git blob id 706cebd746c4b6f2b1d1f892630867acfdfd3df8. Only
# the config is vendored; no weight byte is read by anything here.
CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
CONFIG_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

# The decode step that is traced. Two sequences of one token each: two, not one,
# because a dimension whose trace-time hint is 1 is silently specialised to a
# constant, and a decode step has one token per sequence.
DECODE_SEQS = 2

# The narrowest width this file will trace, and the reason it is not 1. Torch
# specialises a size hint of 0 or 1 to a constant without saying so, so a
# `--step-symbol` capture at width 1 does not produce a symbol at all: it
# produces a fully concrete record with `step_symbol: true` on it, whose family
# census is byte-for-byte the concrete control's. That record reads as a
# successful symbolic capture and is exactly the failure mode this file exists
# to detect, so the capture refuses the width instead of emitting it, and
# `_step_axis` refuses again if a symbol does not come back.
MIN_STEP_WIDTH = 2

# ATOM's own default, restated here because the second specialisation depends
# on it: the block-table buffer is `max_num_seqs` by `max_model_len //
# block_size`, and the published config's 262,144 positions over 16-token
# blocks is the 16,384 that dimension takes.
BLOCK_SIZE = 16

# The KV pool the step indexes into. A block count, not a measurement: no
# device is read for it, and a fixed number is what makes this reproducible.
KV_BLOCKS = 64

# What counts as a collective in the inventory. ATOM issues them two ways: as
# aiter custom operators, which carry a registered fake implementation, and
# through `c10d`'s own entry points.
COLLECTIVE_OPS = r"c10d|aiter\.(all_reduce|all_gather|reduce_scatter|gather)"

# Declared device readings -- node 18's MI308X. Nothing here reads a device.
ARCH = "gfx942"
CU_COUNT = 80
CAPABILITY = (9, 4)
TOTAL_MEMORY_BYTES = 192 * (1 << 30)


# ── what the capture is expected to find ──────────────────────────────────
# Every figure below was measured by running this file. The distinct-operator
# counts are asserted and the totals are not: a total moves with any change to
# ATOM's forward -- a fused kernel, one more view -- while the distinct set
# moves when the kinds of work change, which is the thing worth holding.

TP1_DISTINCT_OPS = 33
TP2_DISTINCT_OPS = 38

# What the width changes, by name. It swaps the embedding for its
# vocab-parallel form, adds the collectives, and adds the `movedim` the
# non-custom gather uses to rearrange what it gathered. Nothing else.
TP1_ONLY_OPS = ("aten.embedding.default",)
TP2_ONLY_OPS = (
    "_c10d_functional.all_gather_into_tensor.default",
    "_c10d_functional.broadcast.default",
    "_c10d_functional.wait_tensor.default",
    "aiter.all_reduce_.default",
    "aiter.masked_embedding.default",
    "aten.movedim.int",
)

# The lines a TP2 decode step issues a collective from. Named, because a
# collective's operator says what ran and only its call site says which of
# ATOM's communication paths ran it.
ROW_PARALLEL = (
    "atom/model_ops/communication_op.py:58 in tensor_model_parallel_all_reduce"
)
VOCAB_EMBEDDING = "atom/model_ops/embed_head.py:175 in forward"
VOCAB_LM_HEAD = "atom/model_ops/embed_head.py:257 in forward"
SAMPLER = "atom/model_engine/model_runner.py:3165 in postprocess"

# Where a free symbol stops being free, each as the value it takes and the
# innermost ATOM frames it takes it through. Three, not two, and in this order:
# a symbol is solved by whichever line reaches it first, so which sites are
# visible is a property of the order and not only of the code. Sites one and
# two are what a plain symbolic pass records; site three is what site one hides,
# and it shows up in the pass that simulates site one closed.
SITE_ONE = (
    "2",
    ("atom/model_ops/attentions/aiter_attention.py:1115 in prepare_decode",),
)
SITE_TWO = (
    "16384",
    (
        "atom/model_ops/attentions/aiter_attention.py:1142 in prepare_decode",
        "atom/utils/__init__.py:725 in copy_to_gpu",
    ),
)
# `assert_shape_contract`, reached with the bound still free once site one no
# longer takes `__index__` of it. It was recorded one frame deeper, in the
# `_rows` helper, where `int(t.shape[0])` converted a dimension before the
# assertion compared it; `_rows` no longer converts, so what is left is the
# assertion itself, equating the width the caller was handed with the width the
# staged buffer carries.
#
# That equality is not a defect and is not closed. In the probe that reaches
# here, the injected bound and the staged buffer's rows are **two symbols**,
# and ATOM's contract is that they are one number -- so solving one against the
# other is the contract doing its job. It is also why the step-symbol pass
# below uses one symbol for the whole step: with one, the same assertion is
# `s == s` and installs nothing.
SITE_THREE = (
    "2",
    ("atom/utils/forward_context.py:444 in assert_shape_contract",),
)

# `CpuGpuBuffer.__init__` as it is on this tree, and how many buffers one
# runner builds through it. ATOM's own `__init__` executes under this capture,
# so these say which constructor answered and how often -- the half of the
# second site's pin that no specialisation site covers, because a repair inside
# `__init__` changes what it allocates rather than where a symbol is solved.
# Measured: 21 buffers, each one host allocation and one numpy view.
BUFFER_INIT = "atom/utils/__init__.py:700"
BUFFER_COUNT = 21

# The `torch.cuda` names `_declare_cuda` stubs, which are exactly the names a
# capture reads. Measured by counting every read of a stubbed name over a whole
# capture: the same 17 at both widths and in every pass this file runs.
CUDA_NAMES_READ = frozenset(
    {
        "Event",
        "Stream",
        "_lazy_init",
        "current_device",
        "current_stream",
        "default_stream",
        "device_count",
        "get_device_capability",
        "get_device_properties",
        "get_rng_state",
        "is_available",
        "memory_allocated",
        "memory_stats",
        "set_device",
        "set_rng_state",
        "stream",
        "synchronize",
    }
)

# Every `torch.cuda` name read by code outside torch, stubbed or not, with who
# reads it. Measured over a whole capture: the same 11 at both widths and in
# every pass this file runs. The 8 stubs missing here are read by torch alone.
CUDA_NAMES_READ_OUTSIDE_TORCH = frozenset(
    {
        "CUDAGraph",  # annotations: atom.utils.cuda_graph, tbo, transformers
        "Event",  # atom.model_engine.model_runner
        "ExternalStream",  # annotation: RapidServeModelRunner
        "Stream",  # atom, aiter, flydsl, transformers
        "current_device",  # aiter, atom.model_ops.attention_mha
        "current_stream",  # atom.utils.forward_context
        "device_count",  # atom.model_ops.fla_ops.utils
        "get_device_properties",  # aiter
        "is_available",  # aiter.ops.gemm_op_a6w6, atom.utils.forward_context
        "memory_stats",  # atom.model_engine.model_runner
        "stream",  # atom.model_engine.model_runner
    }
)

# The families the concrete census is split into at each width. The concrete
# result is that none of their entries is anything but an integer, and that
# sentence is only as good as the count under it -- a family that stopped being
# recorded would read as a family with no symbol in it. So the test holds which
# families there are and that each carries entries, not how many: the counts
# move with any change to ATOM's forward that leaves the inventory exactly as
# concrete, one more view or one more elementwise operator per layer, and the
# record reports them anyway.
#
# The price, chosen rather than missed: a recorder that loses *part* of a
# family -- the GEMMs' input shapes, say, two thirds of that family's entries --
# passes, because the family is still present and still counted. A family that
# goes to zero, or a total that falls under the floor, does not.
CONCRETE_FAMILIES = {
    1: frozenset(
        {
            "activation",
            "allocation",
            "attention",
            "bookkeeping",
            "elementwise",
            "embedding",
            "gemm",
            "normalisation",
            "sampling",
            "transfer",
            "view",
        }
    ),
}
CONCRETE_FAMILIES[2] = CONCRETE_FAMILIES[1] | {"collective"}

# What the census reads on the symbolic probe, keyed by whether site one is
# simulated closed: the families carrying a non-integer shape entry, and the
# operators carrying one. Kinds, not counts.
#
# This is the positive half of the concrete result. The concrete census reads
# zero, and a census that could not tell a symbol from a number would read zero
# too; these are the same census reading what it is there to find. It reads 26
# entries today, and 32 with the repair, but those are not held: a staging
# refactor that behaves identically -- `reshape` for `view` on a contiguous
# buffer in `_copy_mrope_to_gpu` -- moves 26 to 27 while every site and line
# pin in this file holds. What holds the detector instead is the loops that
# count it, read against each other. That refactor still fails the probe test,
# at the operator list: the `reshape` stops being dispatched, so the kinds
# carrying a symbol change, and a change of kind is what this list is for.
PROBE_FAMILIES = {
    False: frozenset({"bookkeeping", "view"}),
    True: frozenset({"bookkeeping", "transfer", "view"}),
}
PROBE_NON_NUMERIC_OPS = {
    False: [
        "aten.as_strided.default",
        "aten.reshape.default",
        "aten.slice.Tensor",
        "prim.device.default",
    ],
    True: [
        "aten.as_strided.default",
        "aten.copy_.default",
        "aten.reshape.default",
        "aten.slice.Tensor",
        "prim.device.default",
    ],
}


# ── the step-symbol pass: what a capture that specialises nowhere records ──

# A second width, used only to show the step's symbol is free rather than
# assumed to be. Eight, not three: it crosses no ATOM threshold this step
# reads, and it is far enough from two that a dimension quietly carrying the
# trace-time hint instead of the symbol shows up as an eight where a two should
# be.
SECOND_WIDTH = 8

# The batch fields that carry the step's width. A decode step's token count and
# its sequence count are one number -- one query row per sequence -- so one
# symbol stands in all four, and no arithmetic here relates them.
STEP_WIDTH_FIELDS = (
    "total_tokens_num",
    "total_tokens_num_decode",
    "total_seqs_num",
    "total_seqs_num_decode",
)

# What each operator is, for a census that is a breakdown rather than a total.
# A total of shape entries carrying a symbol is the figure that has looked like
# progress twice on this property while every symbol sat on a view: the three
# families that decide a step's cost are the claim, and they are named here.
#
# Every name here is a **whole operator name**, matched exactly against the
# operator with its overload dropped -- `aten.add.Tensor` is classified as
# `aten.add`, `aten.empty.memory_format` as `aten.empty`. An operator matching
# none of them lands in `unclassified`, which the tests require to be empty, so
# a new operator on ATOM's forward is a classification decision somebody makes
# rather than a bucket it falls into quietly.
#
# This used to match with `str.startswith`, and that defeated the very guard it
# was written to support: `aten.add` captured `aten.addmm` and `aten.addbmm`,
# which are GEMMs and would have been counted as elementwise work; `aten.slice`
# captured `aten.slice_scatter`, which writes rather than views, and
# `aten.select` captured `aten.select_scatter`. Nothing ATOM dispatches on this
# step is one of those four today -- the census below is unchanged by the fix --
# but the bucket that is supposed to fail could not have seen them, so the three
# cost-bearing families would have understated with nothing going red.
#
# A name declared here that this step never dispatches is a classification made
# ahead of the observation, not an error: `aiter.v4_attention_with_output` is
# the other attention entry point ATOM can take.
OP_FAMILIES = (
    ("gemm", ("aiter.gemm_a16w16",)),
    (
        "attention",
        (
            "aiter.unified_attention_with_output_base",
            "aiter.linear_attention_with_output_base",
            "aiter.v4_attention_with_output",
        ),
    ),
    (
        "normalisation",
        (
            "aiter._fused_qk_rmsnorm_group_quant_kernel",
            "aten.mean",
            "aten.rsqrt",
            "aten.pow",
        ),
    ),
    ("activation", ("aiter.silu_and_mul", "aten.silu", "aten.sigmoid")),
    ("embedding", ("aten.embedding", "aiter.masked_embedding")),
    (
        "collective",
        (
            "aiter.all_reduce_",
            "_c10d_functional.all_reduce",
            "_c10d_functional.all_gather_into_tensor",
            "_c10d_functional.broadcast",
            "_c10d_functional.wait_tensor",
        ),
    ),
    ("sampling", ("aiter.mixed_sample_outer_exponential", "aten.exponential_")),
    ("elementwise", ("aten.add", "aten.mul", "aten.fill_")),
    ("transfer", ("aten.copy_", "aten.to")),
    (
        "view",
        (
            "aten.as_strided",
            "aten.reshape",
            "aten.view",
            "aten.slice",
            "aten.select",
            "aten.expand",
            "aten.split_with_sizes",
            "aten.detach",
            "aten.lift_fresh",
            "aten.movedim",
        ),
    ),
    (
        "allocation",
        ("aten.empty", "aten.empty_like", "aten.scalar_tensor", "aten.zeros"),
    ),
    (
        "bookkeeping",
        (
            "prim.device",
            "profiler._record_function_enter_new",
            "profiler._record_function_exit",
        ),
    ),
)

# The families that decide what a step costs. A capture in which these three
# carry no free symbol is a capture of one shape, whatever its totals say.
COST_BEARING_FAMILIES = ("gemm", "attention", "normalisation")

# The applicability of the step-symbol pass: where the traced graph stops being
# valid. The upper bounds are staging buffers' capacities read against the
# step's width -- ATOM's `max_num_batched_tokens`, its largest attention batch
# and its `max_num_seqs` -- so the graph covers every decode step up to the
# narrowest of them and declines to claim anything above it. None is an
# equality, which is the same fact as `replacements` being empty.
#
# TP2 installs one fewer of them. The per-token bound is the one that goes: the
# width halves each rank's share of the rows that bound reads, and what is left
# is implied by the per-sequence bounds that both widths install. Asserted per
# width rather than as a shared set, because the difference is a property of
# the step and worth failing on if it changes.
EXPECTED_GUARDS = {
    1: ("28*<axis> <= 8192", "<axis> < 128", "<axis> <= 512"),
    2: ("<axis> < 128", "<axis> <= 512"),
}


def expected_guards(tp, width):
    """The guards at TP `tp` traced at step width `width`.

    **The guard set is hint-dependent, and it is the one artifact of this
    capture that is.** At every hint but 2 a *lower* bound appears as well,
    `<axis> + 1 > <hint>`, measured at TP1 at step widths 3, 8 and 16 and at
    TP2 at step width 8. It is not a specialisation: it is a `__bool__`
    comparison installing an inequality on a live symbol, which is the boundary
    this pass deliberately left alive -- so it is positive evidence that
    `__bool__` is untouched rather than a leak. At hint 2 it is elided because
    a size symbol's default range is `[2, inf)`, which makes `s + 1 > 2`
    vacuously true, so torch does not record it.

    What it costs is the applicability sentence: at any hint but 2 the traced
    graph carries a lower bound as well as the upper ones, and a reader
    comparing two widths should be told which artifact differs between them and
    why the digest does not.
    """
    guards = set(EXPECTED_GUARDS[tp])
    if width > MIN_STEP_WIDTH:
        guards.add(f"<axis> + 1 > {width}")
    return guards


# How many of the step's staged buffers are copied to the device through
# `CpuGpuBuffer.copy_to_gpu` with the step's width as the bound. Five from
# `prepare_decode`'s own list, and the rest from the sampler's and the token
# processor's staging around it.
STAGED_COPIES = 10

# Every conversion of the step's width to a number that one traced forward
# makes, by the ATOM line it happens on. This is the log's declared set: the
# conversion is kept -- a host fill genuinely needs a number -- and what is
# replaced is the *recording*, so the log is the only thing standing between a
# conversion at a line nobody expected and a graph that silently describes one
# width. Asserted as a multiset, so a line that converts twice where it used to
# convert once fails too.
#
# Measured identical at TP1 and TP2 and at both widths: 20 conversions over 16
# lines, all of them host fills and slices. A conversion that reaches a *shape*
# would also show up as a digest that differs between two widths; one that only
# reaches a host value would not, which is why this is asserted rather than
# left to the digest.
EXPECTED_HOST_RESOLUTIONS = {
    "atom/model_ops/attentions/aiter_attention.py:1106 in prepare_decode": 2,
    "atom/model_ops/attentions/aiter_attention.py:1115 in prepare_decode": 1,
    "atom/model_ops/attentions/aiter_attention.py:1121 in prepare_decode": 1,
    "atom/model_ops/attentions/aiter_attention.py:1122 in prepare_decode": 1,
    "atom/model_ops/attentions/aiter_attention.py:1123 in prepare_decode": 2,
    "atom/model_ops/attentions/aiter_attention.py:1131 in prepare_decode": 1,
    "atom/model_ops/attentions/aiter_attention.py:1132 in prepare_decode": 2,
    "atom/model_engine/model_runner.py:2468 in prepare_inputs": 1,
    "atom/model_engine/model_runner.py:2479 in prepare_inputs": 2,
    "atom/model_engine/model_runner.py:2481 in prepare_inputs": 1,
    "atom/model_engine/model_runner.py:510 in prepare_input_ids": 1,
    "atom/model_engine/model_runner.py:513 in prepare_input_ids": 1,
    "atom/model_engine/model_runner.py:2564 in prepare_sample": 1,
    "atom/model_ops/attentions/backends.py:398 in _mrope_cpu_view": 1,
    "atom/model_ops/attentions/backends.py:400 in _mrope_cpu_view": 1,
    "atom/model_ops/attentions/gdn_attn.py:1251 in _attach_gdn_decode_metadata": 1,
}


def host_resolutions_by_line(record):
    """`record`'s host-resolution log as a `{ATOM line: conversions}` multiset.

    The innermost ATOM frame is the line that asked, which is the half that
    goes stale silently when a file is edited; the frames above it are the path
    that reached it and are not part of the declared set.
    """
    counted = collections.Counter()
    for event in record["host_resolutions"]:
        frames = event["frames"]
        counted[frames[-1] if frames else "outside atom"] += 1
    return dict(counted)


def op_family(name):
    """Which family `name` belongs to, matched whole rather than by prefix.

    `str(func)` is `namespace.operator.overload`; the overload says how an
    operator was called, not what it does, so it is dropped and the rest is
    matched exactly. Anything else lands in `unclassified` and fails.
    """
    parts = name.split(".")
    base = ".".join(parts[:2]) if len(parts) == 3 else name
    for family, names in OP_FAMILIES:
        if base in names:
            return family
    return "unclassified"


def _anonymise(text, axis):
    """`text` with the step's symbol written as `<axis>` rather than by name.

    A symbol is numbered by the order its ShapeEnv created it. That ordering is
    a property of the run and not of the step, so two records that agree about
    the step disagree about the name; anything compared across runs is compared
    with the name set aside.
    """
    name = str(axis)
    if not re.fullmatch(r"s\d+", name):
        return text
    return re.sub(rf"\b{name}\b", "<axis>", text)


# ---------------------------------------------------------------------------
# the capture driver -- everything below runs in the subprocess


_CUDA_READS_OUTSIDE_TORCH = set()
_UNCOUNTED_READERS = frozenset(
    {"torch", "importlib", "_frozen_importlib", "_frozen_importlib_external"}
)


def _declare_cuda():
    """Stub the `torch.cuda` names ATOM reads, and report what was stubbed.

    Two groups, because they are needed at different moments and a count that
    merges them hides which. The `import` group must be in place before ATOM is
    imported at all: aiter's Triton attention configs read
    `get_device_properties` at module scope. `is_available` has to report True
    -- ATOM branches on the device throughout, and on False takes paths nobody
    runs. `FakeTensorMode` needs the opposite answer; `_driverless_mode` says
    why, and how both are told what they need.

    Streams and events are not readings at all -- a capture has one order by
    construction -- so a null object is the whole of their content.
    """

    class _Props:
        multi_processor_count = CU_COUNT
        gcnArchName = ARCH
        name = ARCH
        major, minor = CAPABILITY
        total_memory = TOTAL_MEMORY_BYTES
        warp_size = 64
        max_threads_per_multi_processor = 2048
        L2_cache_size = 8 << 20
        regs_per_multiprocessor = 65536
        shared_memory_per_block = 65536

    class _Event:
        def record(self, *a, **k):
            return None

        def wait(self, *a, **k):
            return None

        def synchronize(self, *a, **k):
            return None

        def query(self, *a, **k):
            return True

        def elapsed_time(self, *a, **k):
            # Not 0.0: the caller appends this to a list of step durations, so a
            # zero is a confident, precise, entirely fictional measurement. A
            # capture runs no kernel and has no elapsed time to report.
            raise RuntimeError(
                "elapsed_time was read under a capture, which runs no kernel "
                "and will not invent a duration"
            )

    class _Stream:
        def __init__(self, *a, **k):
            self.cuda_stream = 0

        def record_event(self, *a, **k):
            return _Event()

        def wait_event(self, *a, **k):
            return None

        def wait_stream(self, *a, **k):
            return None

        def synchronize(self, *a, **k):
            return None

        def query(self, *a, **k):
            return True

    for_import = {
        "is_available": lambda: True,
        "device_count": lambda: 1,
        "_lazy_init": lambda *a, **k: None,
        "get_rng_state": lambda *a, **k: torch.zeros(16, dtype=torch.uint8),
        "set_rng_state": lambda *a, **k: None,
        "get_device_properties": lambda *a, **k: _Props(),
        "current_device": lambda: 0,
        "get_device_capability": lambda *a, **k: CAPABILITY,
    }
    for_runner = {
        # `Event` stays a type, not a lambda: `ModelRunner` evaluates
        # `torch.cuda.Event | None` in a class body at import, and
        # `function | None` is a TypeError.
        "Stream": _Stream,
        "Event": _Event,
        "current_stream": lambda *a, **k: _Stream(),
        "default_stream": lambda *a, **k: _Stream(),
        "set_device": lambda *a, **k: None,
        "synchronize": lambda *a, **k: None,
        "memory_stats": lambda *a, **k: {
            "allocated_bytes.all.current": 0,
            "allocated_bytes.all.peak": 0,
            "reserved_bytes.all.current": 0,
        },
        "memory_allocated": lambda *a, **k: 0,
        "stream": lambda s: contextlib.nullcontext(),
    }
    for group in (for_import, for_runner):
        for name, value in group.items():
            setattr(torch.cuda, name, value)

    # Every read of a stubbed name, counted. A stub is a claim that something
    # reads the name, and a list of names is not evidence of that: the list is
    # what this function wrote. The count is what the capture did with it.
    # Reads, not calls, because `Event` is read as a type in a class body and
    # never called.
    stubbed = frozenset(for_import) | frozenset(for_runner)
    reads = collections.Counter()

    # Separately, every name read outside torch, stubbed or not: an unstubbed
    # read is a real device reading on a GPU host and a nameless error without
    # one. The reader is the calling frame's module package. Torch's own and the
    # import system's reads bind names and are not counted. A reader that cannot
    # be named is recorded as "<name> (reader unknown)", so it fails by name.
    # Dunder names are module metadata, never a device reading, and are skipped.

    class _ReadCountingModule(type(torch.cuda)):
        def __getattribute__(self, name):
            if name in stubbed:
                reads[name] += 1
            if name.startswith("__"):
                return super().__getattribute__(name)
            try:
                reader = sys._getframe(1).f_globals["__name__"]
            except (ValueError, KeyError):
                reader = None
            if not isinstance(reader, str):
                _CUDA_READS_OUTSIDE_TORCH.add(f"{name} (reader unknown)")
            elif reader.partition(".")[0] not in _UNCOUNTED_READERS:
                _CUDA_READS_OUTSIDE_TORCH.add(name)
            return super().__getattribute__(name)

    torch.cuda.__class__ = _ReadCountingModule
    return {"import": sorted(for_import), "model_runner": sorted(for_runner)}, reads


def _decline_initialisers():
    """Decline the in-place RNG fills `nn.Module.reset_parameters` performs.

    A layer built on a CUDA device initialises its parameters through
    `Tensor.uniform_` / `Tensor.normal_`, and under the mode those reach a
    decomposition that asks the device for a generator -- `HIP error: no
    ROCm-capable device is detected`, from inside `nn.Conv3d.__init__` in the
    vision tower.

    Declining them costs nothing that is measured here. This capture reads no
    checkpoint, so every parameter value is arbitrary before it is arbitrary;
    what the inventory records is shape, dtype and device, and the fill changes
    none of the three. Returning `self` keeps the initialiser's contract.
    """
    filled = []
    for name in ("uniform_", "normal_"):
        filled.append(name)
        setattr(torch.Tensor, name, lambda self, *a, **k: self)
    return filled


def _driverless_mode(shape_env):
    """`FakeTensorMode` told the device is absent, while ATOM is told it is there.

    These two readers of `torch.cuda.is_available()` need opposite answers, and
    the same function cannot give both.

    ATOM needs True. ATOM branches on the device throughout, registers its
    operators at `dispatch_key="CUDA"`, and on False takes paths nobody runs.

    `FakeTensorMode` needs False, on a host with no driver at all. That one flag
    gates three accommodations it makes for an absent device, and every one of
    them is load-bearing here:

    * `_only_lift_cpu_tensors(True)`, which keeps `torch.tensor` on the host and
      moves it afterwards. `torch.tensor` reads its device eagerly, in C++,
      below anything the mode can intercept, so ATOM's own
      `torch.tensor([])` under a CUDA default device is otherwise
      `No HIP GPUs are available`;
    * `_ensureCUDADeviceGuardSet()`, which swaps the CUDA device guard for a
      no-op so CUDA kernels can be traced at all;
    * skipping constant propagation across a device conversion. Fake tensors
      small enough to carry their constant otherwise have their next operator
      run **for real** on the destination device.

    Overriding the property says the second without changing the first. The
    original measurement of this stub set was taken against a host whose driver
    was wedged rather than absent, where reporting False hangs inside
    `_ensureCUDADeviceGuardSet` and reporting True completes; with no driver at
    all the dependency runs the other way.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    class _DriverlessFakeTensorMode(FakeTensorMode):
        @property
        def avoid_device_init(self):
            return True

    return _DriverlessFakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)


def _decline_custom_all_gather():
    """Select the non-custom vocab-parallel gather, and say so.

    ATOM's default sends `embed_head.py`'s vocab-parallel gather down a custom
    path that reads `device_communicator.ca_comm` -- the device communicator
    this capture declines, because it opens a rendezvous that waits for ranks
    that do not exist. Left on, a TP2 forward dies with `'NoneType' object has
    no attribute 'ca_comm'` after roughly 2,500 operators; the non-custom form
    gathers the same shapes and is recorded.

    This is a configuration of the capture, not a stub, and belongs in the
    record beside the device readings: it is the whole difference between the
    run that refuses part-way and the one that completes.
    """
    os.environ["ATOM_USE_CUSTOM_ALL_GATHER"] = "0"
    return "ATOM_USE_CUSTOM_ALL_GATHER=0"


def _decline_context_priming():
    """Decline the one real allocation the fake-tensor machinery makes.

    `FakeTensor.__new__` primes a CUDA context for any device-tagged fake it
    builds -- a real `torch.zeros(1, device=...)`, guarded only by
    `torch.cuda.is_available()`, which the stubs above must report True. It
    exists so that backward through a CUDA fake does not error; nothing here
    runs backward, and there is no context to prime. Left in place it is the
    first thing the capture does and the first thing that fails.
    """
    from torch._subclasses import fake_tensor

    fake_tensor.init_gpu_context = lambda device: None
    return "fake_tensor.init_gpu_context"


def _declare_arch(tmpdir):
    """Declare the GPU architecture to the two things that go looking for it.

    aiter resolves it twice and only one of those reads `GPU_ARCHS`: the other,
    `get_gfx_runtime`, shells out to `rocminfo` unconditionally and raises when
    there is no `/dev/kfd`. A `rocminfo` of our own on PATH answers it. Triton's
    side is the same fact through a different door -- `triton.runtime.driver`
    asks the live device for its target, and aiter falls back to a jax import
    that is not installed when that raises.

    Both are the arch, configured. Neither is read from a device here, and the
    proxy answers nothing but the target: any other attribute raises rather than
    quietly standing in for a driver.
    """
    import triton
    from triton.backends.compiler import GPUTarget

    binpath = pathlib.Path(tmpdir) / "declared-bin"
    binpath.mkdir(parents=True, exist_ok=True)
    rocminfo = binpath / "rocminfo"
    rocminfo.write_text(f'#!/bin/sh\necho "  Name:                    {ARCH}"\n')
    rocminfo.chmod(0o755)
    os.environ["PATH"] = f"{binpath}:{os.environ.get('PATH', '')}"
    os.environ["GPU_ARCHS"] = ARCH

    class _DeclaredTarget:
        def get_current_target(self):
            return GPUTarget("hip", ARCH, 64)

        def __getattr__(self, name):
            raise RuntimeError(
                f"a device-free capture has no Triton driver; {name} was read"
            )

    triton.runtime.driver.set_active(_DeclaredTarget())
    return {"rocminfo": str(rocminfo), "triton_target": ARCH}


def _functional_collectives():
    """Route `c10d`'s legacy in-place collectives to their functional forms.

    ATOM's collective at TP>1 is aiter's `all_reduce_`, a registered custom
    operator with a registered fake implementation, and it needs nothing here.
    A handful of call sites reach `torch.distributed`'s own entry points
    instead, and on this stack those `c10d::*` operators carry a backend kernel
    and **neither a Meta nor a CompositeExplicitAutograd one**, so under the
    mode they raise `NotImplementedError` rather than producing a meta
    operation. Measured to be general rather than particular to one of them:
    `c10d::barrier` and `c10d::_allgather_base_` fail the same way.

    The functional forms do carry a kernel the mode can run and give the same
    shapes, so the collective is still recorded, with its real shapes, having
    communicated nothing. The output tensor is filled by the caller's own
    contract, so the shapes a reader sees are the shapes the legacy call would
    have produced.

    The one arrangement the substitution does change is recorded here rather
    than inferred anywhere. ATOM hands `all_gather_into_tensor` an output
    buffer of `(world_size,) + input_size`; the functional form concatenates
    along dim 0 into `(world_size * rows,) + rest` and the shim reshapes. Both
    are measured off the live call, so a reader of the record never has to
    guess which of the two a shape belongs to.
    """
    import torch.distributed as dist
    from torch.distributed._functional_collectives import (
        all_gather_tensor,
    )
    from torch.distributed._functional_collectives import (
        broadcast as functional_broadcast,
    )

    buffers = []

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        gathered = all_gather_tensor(input, 0, group or dist.group.WORLD)
        # aiter stages the gather into a `(world_size,) + input_size` buffer and
        # reshapes afterwards; the functional form concatenates along dim 0.
        # Same elements, same order, different arrangement.
        buffers.append(
            {
                "op": "all_gather_into_tensor",
                "input": [str(dim) for dim in input.shape],
                "atom_output": [str(dim) for dim in output.shape],
                "functional_output": [str(dim) for dim in gathered.shape],
            }
        )
        output.copy_(gathered.reshape(output.shape))

    def broadcast(tensor, src=0, group=None, async_op=False):
        tensor.copy_(functional_broadcast(tensor, src, group or dist.group.WORLD))

    dist.all_gather_into_tensor = all_gather_into_tensor
    dist.broadcast = broadcast
    return ["all_gather_into_tensor", "broadcast"], buffers


def _build_group(tp):
    """A process group of width `tp` inside one process, transport declined.

    The width is real: `get_tp_group().world_size` is `tp` because the group has
    `tp` ranks, so every shard size in the tree is the one a `tp`-way deployment
    computes, and every collective ATOM issues is dispatched and recorded. What
    one process cannot build is the transport -- a device communicator opens a
    rendezvous that waits for ranks that do not exist, as does the message-queue
    broadcaster, and a gloo sub-group's own rendezvous does the same. All three
    are declined here; none of them carries a shape.

    The collective ATOM issues at TP>1 does not need any of them. aiter's
    `all_reduce_` is a registered custom operator with a registered fake
    implementation, so under the mode the fake answers and the body that wants a
    communicator is unreachable: the operator is recorded with its real shapes
    having allocated and communicated nothing.
    """
    import torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore

    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(tp)
    os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp))
    dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=tp)

    from aiter.dist import parallel_state

    original_group = parallel_state.init_model_parallel_group
    original_new_group = dist.new_group

    def decline_transport(*args, **kwargs):
        kwargs["use_device_communicator"] = False
        kwargs["use_message_queue_broadcaster"] = False
        return original_group(*args, **kwargs)

    def decline_backend(ranks=None, **kwargs):
        # Every sub-group, including the `gloo` one the coordinator builds for
        # host-side coordination, takes the same peerless backend as the world.
        kwargs["backend"] = "fake"
        kwargs.pop("pg_options", None)
        return original_new_group(ranks, **kwargs)

    # The barrier in `allocate_kv_cache` is transport too, and it is the one
    # that has no fake implementation to answer with: `c10d::barrier` carries a
    # backend kernel and neither a Meta nor a CompositeExplicitAutograd one, so
    # under the mode it raises rather than producing a meta operation. It
    # carries no shape and moves no bytes; there is nothing for an inventory to
    # record and nobody to wait for.
    dist.barrier = lambda *args, **kwargs: None

    parallel_state.init_model_parallel_group = decline_transport
    dist.new_group = decline_backend
    try:
        parallel_state.init_distributed_environment(
            world_size=tp, rank=0, backend="fake", local_rank=0
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp, backend="fake"
        )
    finally:
        parallel_state.init_model_parallel_group = original_group
        dist.new_group = original_new_group
    functional, gather_buffers = _functional_collectives()
    return (
        parallel_state.get_tp_group().world_size,
        {
            "declined": [
                "new_group backend",
                "device_communicator",
                "message_queue_broadcaster",
                "barrier",
            ],
            "functional": functional,
        },
        gather_buffers,
    )


def _watch_simulated_tp(tree_root):
    """A sentinel on `apply_simulated_tp`, in place of a hard-coded `False`.

    A TP>1 inventory taken through `apply_simulated_tp` both erases and
    fabricates, so a record that came through it is not a TP>1 record at all.
    That used to be carried by a literal written into the record and asserted
    against itself, which cannot fail and cannot go stale.

    The sentinel records every call, with the ATOM frames that made it, and
    does not call through: a run in which it fires produces a record naming
    the site rather than a number nobody can check. Both bindings are taken,
    because `model_runner` imports the name rather than the module.

    **It is a forward guard, not an observation about ATOM at TP2 today.**
    ATOM calls `apply_simulated_tp` from one place,
    `ModelRunner._setup_device_and_distributed`, and `_build_runner` overrides
    that method, so no capture can reach the call and the list is empty by
    construction. Replacing the sentinel's body with `raise SystemExit` leaves
    every test that runs a capture passing, and fails the one that calls both
    bindings directly. It fires only if ATOM comes to call the function
    from a path this capture does execute. That ATOM has one call site, and
    that it is the overridden method, is held by a scan of ATOM's source in
    `test_apply_simulated_tp_is_called_only_where_the_capture_does_not_go`, so
    a second caller fails there even where the sentinel cannot see it.
    """
    from atom.distributed import simulated_tp
    from atom.model_engine import model_runner

    calls = []

    def sentinel(config):
        calls.append({"frames": _atom_frames(tree_root)})

    simulated_tp.apply_simulated_tp = sentinel
    model_runner.apply_simulated_tp = sentinel
    return calls


def _hint(value):
    """A `SymInt`'s trace-time hint, read without solving it.

    `int(sym)` and `sym.__index__()` both *specialise*: they record the hint as
    the symbol's value and the symbol stops being free. `sym.node.hint` reads
    the same number and records nothing, which is the whole difference between
    standing in for a repair and being the thing a repair is about.
    """
    if isinstance(value, torch.SymInt):
        hint = value.node.hint
        if hint is None:
            raise RuntimeError(f"{value} carries no hint to read")
        return int(hint)
    return value


def _hinted(key):
    """The same subscript with every `SymInt` in it replaced by its hint."""
    if isinstance(key, tuple):
        return tuple(_hinted(item) for item in key)
    if isinstance(key, slice):
        return slice(_hint(key.start), _hint(key.stop), _hint(key.step))
    return _hint(key)


class _HintSlicedView(numpy.ndarray):
    """A staging buffer's numpy view that does not solve a `SymInt` bound.

    This is the **simulated site-one repair**, and it is a probe rather than
    part of any capture. `prepare_decode` fills each staging buffer's numpy
    view with the row count it was handed -- `var["slot_mapping"].np[:running_
    tokens]` -- and numpy takes `__index__` of whatever it is given, which
    solves the symbol. A staging buffer that reads the bound's hint instead
    leaves it free. That is the shape of one repair to this site, applied from
    outside ATOM rather than by editing it.

    It exists to measure what happens *next*, because the answer is not what
    the two-site account assumed: the bound is not closed by repairing the
    site that solves it first. It is solved somewhere else, and that somewhere
    is in a third module.

    A `numpy.ndarray` subclass rather than a wrapper, because ATOM's own
    `pack_rows` takes `memoryview()` of the staging view -- a delegating
    wrapper is `a bytes-like object is required` there, and swapping the
    buffer protocol out is a change to the thing being measured rather than
    to the one line under test. `.view(_HintSlicedView)` shares the storage.
    """

    def __setitem__(self, key, value):
        super().__setitem__(_hinted(key), value)

    def __getitem__(self, key):
        return super().__getitem__(_hinted(key))


@contextlib.contextmanager
def _staged_allocators(fake_mode, symbolic, observed):
    """The three primitives `CpuGpuBuffer.__init__` needs staged, and no more.

    Substituted for the duration of one `__init__` body, so that what runs is
    ATOM's own body. The first version of this file replaced the method
    wholesale instead. The buffer built, but every line of ATOM's `__init__`
    was then unreachable: a `raise` as its first statement changed nothing
    anywhere in this file, and the repair route that makes `CpuGpuBuffer`
    symbolic -- an allocation change, and the
    allocation is `__init__` -- was the one route this test could not see.

    * `torch.zeros` for the host side runs outside the mode, so `self.cpu` is
      a real, numpy-backed, concrete tensor, which is the concrete half of
      the straddle. `pin_memory` is dropped: pinning is a real host allocation
      through the driver (`hipHostMalloc`), a property of the transfer rather
      than of the shape, and nothing traced here can observe it.
    * `torch.zeros_like` for the device side is substituted **only** in the
      symbolic pass. `static_shapes=False` is what puts a free symbol on each
      dimension; a tensor allocated inside the mode is already fake and
      *static*, with plain `int` shapes and no sign that anything was lost.
      The symbolic side is converted from a template that is then dropped, not
      from `self.cpu`: converting `self.cpu` memoises it as a symbolic fake,
      and the first operator ATOM performs on the CPU side directly then asks
      the converter for a concrete view of a tensor it has already given a
      symbolic meta storage -- `Trying to resize storage that is not
      resizable`. In the concrete pass ATOM's own `zeros_like` runs unaltered.
    * `Tensor.numpy` runs outside the mode. A `.numpy()` taken while the mode
      is active leaves the real storage marked not resizable, and the next
      operator on the CPU side then fails inside the converter with a
      deprecation warning about reading a FakeTensor data pointer as the only
      clue. The numpy view is a host alias, not an operation worth tracing.

    Each substitution counts its calls, and the counts go in the record. They
    are what says ATOM's body ran, and they move if its allocations are added
    to, removed or re-routed -- which is the other half of closing the hole.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    real_zeros = torch.zeros
    real_zeros_like = torch.zeros_like
    real_numpy = torch.Tensor.numpy
    reentered: list[bool] = []

    def zeros(*size, **kwargs):
        kwargs.pop("pin_memory", None)
        device = kwargs.get("device")
        if device is not None and torch.device(device).type != "cpu":
            return real_zeros(*size, **kwargs)
        observed["host_allocations"] += 1
        with unset_fake_temporarily():
            return real_zeros(*size, **kwargs)

    def zeros_like(tensor, **kwargs):
        if not symbolic:
            return real_zeros_like(tensor, **kwargs)
        device = kwargs.pop("device", None)
        with unset_fake_temporarily():
            template = real_zeros_like(tensor, device="cpu", **kwargs)
        observed["symbolic_device_allocations"] += 1
        staged = fake_mode.from_tensor(template, static_shapes=False)
        return staged if device is None else staged.to(device)

    def numpy(tensor, *args, **kwargs):
        # One view, counted once. `Tensor.numpy` is dispatched through the
        # torch-function mode `set_default_device` installs, which calls the
        # bound name again with dispatch off -- so an unguarded counter reads
        # two per buffer and the number stops meaning what it says.
        if not reentered:
            observed["numpy_views"] += 1
        reentered.append(True)
        try:
            with unset_fake_temporarily():
                return real_numpy(tensor, *args, **kwargs)
        finally:
            reentered.pop()

    torch.zeros = zeros
    torch.zeros_like = zeros_like
    torch.Tensor.numpy = numpy
    try:
        yield
    finally:
        torch.zeros = real_zeros
        torch.zeros_like = real_zeros_like
        torch.Tensor.numpy = real_numpy


def _stage_buffers(fake_mode, symbolic, tree_root, repair_site_one=False):
    """Run ATOM's own `CpuGpuBuffer.__init__`, staging only what has no device.

    `CpuGpuBuffer.__init__` allocates a CPU tensor, allocates the device side
    `zeros_like` it, and takes `.numpy()` of the first. Under the mode all
    three are faked and `.numpy()` raises -- `.numpy() is not supported for
    tensor subclasses` -- so no runner constructs without something being
    done here. What is done is the three substitutions in `_staged_allocators`
    above; the body between them is ATOM's, executed.

    That the body executes is what this test pins, and the record carries the
    evidence: which `__init__` ran, how many buffers it built, and how many of
    each staged allocation it asked for. `copy_to_gpu` is pinned by the
    specialisation site it produces; `__init__` is pinned by these counts,
    because a repair there changes what it allocates and not where a symbol is
    solved.

    `repair_site_one` wraps each numpy view in `_HintSlicedView` afterwards.
    That is the probe, not the capture: see the class.
    """
    from atom.utils import CpuGpuBuffer

    original = CpuGpuBuffer.__init__
    code = original.__code__
    path = pathlib.Path(code.co_filename).resolve()
    try:
        source = str(path.relative_to(tree_root))
    except ValueError:
        source = str(path)
    observed = {
        "source": f"{source}:{code.co_firstlineno}",
        "constructed": 0,
        "host_allocations": 0,
        "symbolic_device_allocations": 0,
        "numpy_views": 0,
        "hint_sliced_views": 0,
    }

    def staged_init(self, *size, dtype, device, pin_memory=True, with_numpy=True):
        observed["constructed"] += 1
        with _staged_allocators(fake_mode, symbolic, observed):
            original(
                self,
                *size,
                dtype=dtype,
                device=device,
                pin_memory=pin_memory,
                with_numpy=with_numpy,
            )
        if repair_site_one and with_numpy:
            observed["hint_sliced_views"] += 1
            self.np = self.np.view(_HintSlicedView)

    CpuGpuBuffer.__init__ = staged_init
    return original, observed


def _atom_frames(tree_root):
    """The ATOM source frames on the live stack, innermost last.

    A collective's name says which operator ran; only its call site says which
    of ATOM's twenty-odd communication paths issued it, and that is the half
    that goes stale silently when a file is edited.
    """
    frames = []
    for frame in traceback.extract_stack():
        path = pathlib.Path(frame.filename)
        try:
            rel = path.relative_to(tree_root)
        except ValueError:
            continue
        if rel.parts[0] != "atom":
            continue
        frames.append(f"{rel}:{frame.lineno} in {frame.name}")
    return frames


class _Recorder(TorchDispatchMode):
    """Every dispatched operator, with its shapes as the ShapeEnv reports them.

    Shapes are stringified rather than kept as `SymInt`s: an inventory is
    compared across widths and across runs, and a `SymInt` compares by the
    identity of its symbol, which differs between two ShapeEnvs that agree.
    """

    def __init__(self, tree_root, collective_pattern):
        super().__init__()
        self._tree_root = tree_root
        self._collective = re.compile(collective_pattern)
        self.ops = []
        self.collectives = []
        # Which ATOM lines produced an operator whose shapes are not all plain
        # integers. Only these walk the stack -- a concrete capture reaches none
        # of them and pays nothing, and a symbolic one reaches a few hundred.
        self.symbolic_sites = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        shapes_in = [self._shape(a) for a in self._tensors(args)]
        shapes_out = [self._shape(o) for o in self._tensors(self._flat(out))]
        self.ops.append((name, shapes_in, shapes_out))
        if self._carries_symbol(shapes_in, shapes_out):
            frames = _atom_frames(self._tree_root)
            self.symbolic_sites[(name, frames[-1] if frames else "outside atom")] += 1
        if self._collective.search(name):
            frames = _atom_frames(self._tree_root)
            self.collectives.append(
                {
                    "op": name,
                    "call_site": frames[-1] if frames else "outside atom",
                    "shapes": shapes_in,
                }
            )
        return out

    @staticmethod
    def _carries_symbol(shapes_in, shapes_out):
        for shape in (*shapes_in, *shapes_out):
            for dim in shape:
                if not re.fullmatch(r"-?\d+", dim):
                    return True
        return False

    def _tensors(self, items):
        return [x for x in items if isinstance(x, torch.Tensor)]

    def _flat(self, value):
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.extend(self._flat(item))
            return out
        return [value]

    @staticmethod
    def _shape(tensor):
        return [str(dim) for dim in tensor.shape]

    def distinct_ops(self):
        return sorted({name for name, _, _ in self.ops})

    def non_numeric_ops(self):
        """The operators carrying a shape entry that is not a plain integer."""
        carrying = set()
        for name, shapes_in, shapes_out in self.ops:
            for shape in (*shapes_in, *shapes_out):
                if any(not re.fullmatch(r"-?\d+", dim) for dim in shape):
                    carrying.add(name)
        return sorted(carrying)

    def shape_census(self):
        """`(shape entries recorded, entries that are not a plain integer)`.

        The second number separates a symbolic inventory from a concrete one: if
        every shape is an integer, the inventory describes only the shapes it was
        taken at and can be evaluated nowhere else.

        It is the total, and it is reported **only** beside `family_census`. On
        its own it says nothing worth acting on: sixty entries on `prim.device`,
        `as_strided` and `fill_` read exactly like sixty on the GEMMs.
        """
        entries = non_numeric = 0
        for _, shapes_in, shapes_out in self.ops:
            for shape in (*shapes_in, *shapes_out):
                for dim in shape:
                    entries += 1
                    if not re.fullmatch(r"-?\d+", dim):
                        non_numeric += 1
        return entries, non_numeric

    def family_census(self):
        """The census split by what each operator does.

        One row per family: how many operators it ran, how many shape entries
        they carry, how many of those are not plain integers, and which named
        operators in it carry one. The last is what makes the row auditable --
        a family can be "symbolic" because one view in it is, and naming the
        operators says which.
        """
        rows = {}
        for name, shapes_in, shapes_out in self.ops:
            row = rows.setdefault(
                op_family(name),
                {
                    "ops": 0,
                    "shape_entries": 0,
                    "non_numeric_shape_entries": 0,
                    "ops_carrying_a_symbol": collections.Counter(),
                },
            )
            row["ops"] += 1
            carrying = False
            for shape in (*shapes_in, *shapes_out):
                for dim in shape:
                    row["shape_entries"] += 1
                    if not re.fullmatch(r"-?\d+", dim):
                        row["non_numeric_shape_entries"] += 1
                        carrying = True
            if carrying:
                row["ops_carrying_a_symbol"][name] += 1
        for row in rows.values():
            row["ops_carrying_a_symbol"] = dict(row["ops_carrying_a_symbol"])
        return rows

    def concrete_dims(self):
        """Every dimension that stayed a number, by family and by value.

        The other half of the census, and the half that says whether the first
        half means anything. A concrete entry is either a fact about the model
        or the engine -- a hidden size, a head count, a staging buffer's
        capacity -- or a width that was quietly resolved somewhere this capture
        did not intercept, and only the value distinguishes them.
        """
        rows = {}
        for name, shapes_in, shapes_out in self.ops:
            row = rows.setdefault(op_family(name), collections.Counter())
            for shape in (*shapes_in, *shapes_out):
                for dim in shape:
                    if re.fullmatch(r"-?\d+", dim):
                        row[dim] += 1
        return {family: dict(counts) for family, counts in rows.items()}

    def graph_digest(self, axis):
        """A digest of the whole inventory, with the step's symbol anonymised.

        Two captures taken at two different widths produce the same digest
        exactly when their operator lists agree name for name and shape for
        shape once the symbol's *name* is set aside. That is the evidence that
        the symbol is free: if any dimension is carrying the trace-time hint
        rather than the symbol, it is a 2 in one inventory and an 8 in the
        other, and the digests differ.
        """
        lines = [
            _anonymise(f"{name}|{shapes_in}|{shapes_out}", axis)
            for name, shapes_in, shapes_out in self.ops
        ]
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()


class _Specialisations:
    """Where each free symbol stopped being free, and what it became.

    `ShapeEnv._set_replacement` is the single funnel every replacement goes
    through -- its own docstring says to use it rather than assigning into
    `replacements` -- so wrapping it catches each one at the moment it happens,
    with the live stack still standing. Reading `replacements` afterwards gives
    the same symbols and none of the sites, and the site is the claim: a symbol
    solved somewhere else is a different finding about a different line.
    """

    def __init__(self, shape_env, tree_root):
        self.events = []
        self._shape_env = shape_env
        self._tree_root = tree_root
        self._original = type(shape_env)._set_replacement

    def __enter__(self):
        recorder = self

        def watched(shape_env, symbol, target, msg):
            before = shape_env.replacements.get(symbol)
            result = recorder._original(shape_env, symbol, target, msg)
            after = shape_env.replacements.get(symbol)
            if after is not None and after != before:
                recorder.events.append(
                    {
                        "symbol": str(symbol),
                        "value": str(after),
                        "frames": _atom_frames(recorder._tree_root),
                    }
                )
            return result

        type(self._shape_env)._set_replacement = watched
        return self

    def __exit__(self, *exc):
        type(self._shape_env)._set_replacement = self._original
        return False


class _TritonLaunches:
    """Record `@triton.jit` launches and do not launch them -- a diagnostic.

    A raw `kernel[grid](...)` goes straight to the AMD driver. It never enters
    the torch dispatcher, so `TorchDispatchMode` cannot see it and
    `FakeTensorMode` cannot fake it: the launcher asks a `FakeTensor` for its
    `data_ptr` and the first one reached ends the trace. Skipping keeps the
    trace alive long enough to enumerate which kernels a step reaches, which is
    what deciding how to price them needs. It is not a capture: a skipped kernel
    writes nothing, so everything downstream of one reads uninitialised fake
    memory.
    """

    def __init__(self):
        self.launches = {}
        self._original = None

    def __enter__(self):
        from triton.runtime.jit import JITFunction

        self._original = JITFunction.run
        launches = self.launches

        def run(jit_function, *args, **kwargs):
            launches[jit_function.__name__] = launches.get(jit_function.__name__, 0) + 1

        JITFunction.run = run
        return self

    def __exit__(self, *exc):
        from triton.runtime.jit import JITFunction

        JITFunction.run = self._original
        self._original = None
        return False


def _decode_batch(width=DECODE_SEQS, axis=None):
    """Two sequences of one token each, built the way ATOM builds a dummy step.

    Two, not one: a dimension whose trace-time hint is 1 is silently specialised
    to a constant, and a decode step has exactly one token per sequence, so a
    one-sequence trace yields a fully constant graph with no warning.

    `axis` is the step's width as a free symbol, for the step-symbol pass, or
    `None` everywhere else. Everything else about the batch -- the sequences,
    their block tables, the per-sequence token counts -- is identical either
    way, because the symbol is the *width*, not the content.

    The batch is built at the concrete width and the four count fields are
    rebound afterwards, rather than the symbol being passed to the constructor.
    `ScheduledBatch.__init__` checks that the token array it staged is exactly
    `total_tokens_num` long -- a check about host values, and a correct one --
    and comparing a real length against a symbol resolves the symbol there,
    before the step has begun. Rebinding leaves that check running on the
    numbers it is about.
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
        seq = Sequence([0], block_size=BLOCK_SIZE, id=index)
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.DECODE
        seq.block_table = new_block_table([index])
        seqs[seq.id] = seq
    batch = ScheduledBatch(
        seqs=seqs,
        num_scheduled_tokens=np.ones(width, dtype=np.int32),
        total_tokens_num=width,
        total_tokens_num_decode=width,
        total_seqs_num=width,
        total_seqs_num_decode=width,
        is_dummy_run=True,
    )
    if axis is not None:
        for name in STEP_WIDTH_FIELDS:
            setattr(batch, name, axis)
    return batch


def _build_runner(config, fake_mode):
    """ATOM's own `ModelRunner`, constructed inside the mode, reading nothing.

    Two of the base class's own override points do all of the work. The model is
    built -- the real class, from `support_model_arch_dict`, in the model's own
    dtype -- and no checkpoint is read; at ATOM's fp32 default the fake tensors
    would trace kernels real hardware rejects, so the graph would not be the one
    that runs. Warmup is skipped because it runs a forward from inside
    `__init__`, and this capture drives its own.

    The distributed setup is skipped too: the group already exists, built at the
    honest width with its transport declined, and ATOM's own path would rebuild
    it through `init_dist_env` and then reach for `apply_simulated_tp`.
    """
    from atom.model_engine.model_runner import ModelRunner, support_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    architecture = config.hf_config.architectures[0]
    if architecture not in support_model_arch_dict:
        raise RuntimeError(
            f"{architecture} is not in ATOM's support_model_arch_dict, so this "
            "capture has no model class to build"
        )
    model_class = resolve_obj_by_qualname(support_model_arch_dict[architecture])

    class _CapturedRunner(ModelRunner):
        def _setup_device_and_distributed(self, rank, config):
            self.device = torch.device("cuda:0")

        def _build_and_load_model(self, built_class):
            self.model = built_class(config)
            torch.set_default_device(None)

        def _maybe_warmup(self):
            return

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    try:
        with fake_mode:
            runner = _CapturedRunner(0, config)
    finally:
        torch.set_default_dtype(previous_dtype)
    return architecture, model_class.__name__, runner


def _symbolic_bound(runner, fake_mode):
    """Hand `prepare_decode` its row count as a free symbol instead of an int.

    This is the caller's half of the specialisation. `prepare_decode` takes one
    count per staged buffer, uses it to fill the buffer's numpy view and then
    again as the copy's bound, and anything that needs an `int` takes
    `__index__` of a `SymInt` and gets its hint -- recording the symbol as a
    constant with no error and no warning. It is not a numpy behaviour: a bare
    `__index__()` and a plain list slice do the same.

    The symbol is made from a tensor of exactly the count the caller passed, so
    its hint is that count and the step traced is the step ATOM asked for.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    builder = runner.attn_metadata_builder
    original = builder.build
    injected = []

    def build(batch, running_bs, running_tokens, max_seqlen_q):
        # Outside the mode. A tensor allocated inside it is already fake and
        # *static*, `from_tensor` is then a no-op, and the bound comes back a
        # plain `int` with nothing to say it was meant to be a symbol.
        with unset_fake_temporarily():
            template = torch.zeros(int(running_tokens), dtype=torch.int32)
        bound = fake_mode.from_tensor(template, static_shapes=False).shape[0]
        injected.append({"symbol": str(bound), "hint": int(running_tokens)})
        return original(batch, running_bs, bound, max_seqlen_q)

    builder.build = build
    return injected


def _step_axis(fake_mode, hint):
    """One free symbol standing for the width of a decode step.

    It is handed to the `ScheduledBatch` as its token and sequence counts, and
    ATOM derives everything else from there: `ForwardMode.decide` settles
    `running_bs` and `running_tokens` off it, `prepare_inputs` writes the
    `cu_seqlens_q` boundary at `running_bs + 1`, and `prepare_decode` uses it as
    every staged buffer's bound. None of that arithmetic is re-done here --
    supplying a bound per buffer, which is what `_symbolic_bound` does for the
    probe above, is a width this capture chose rather than one ATOM ran.

    **One symbol, not two.** A decode step has one query row per sequence, so
    its token count and its sequence count are the same number for a structural
    reason and not by coincidence: `running_tokens` is `running_bs * q` with
    `q == 1`. Minting a symbol for each would produce a graph in which the
    hidden states and the attention metadata carry unrelated widths, and ATOM's
    own shape contract -- which asserts the two agree -- would then have to
    equate them, which is the specialisation again by a longer route.

    The symbol comes from a tensor of exactly `hint` rows, so its hint is the
    count ATOM would have computed and the step traced is the step ATOM asked
    for. It is built outside the mode: a tensor allocated inside it is already
    fake and *static*, `from_tensor` is then a no-op, and what comes back is a
    plain `int` with nothing to say it was meant to be a symbol.

    **And a plain `int` is what comes back at a hint of 0 or 1**, silently:
    torch specialises those two sizes to constants by design. A capture that
    went on from there would produce a fully concrete record wearing
    `step_symbol: true`, which is the failure mode this whole file exists to
    detect, so this refuses instead of returning it. `main` refuses the width
    earlier, at the command line; this is the backstop that does not depend on
    the caller having come through `main`.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        template = torch.zeros(hint, dtype=torch.int32)
    axis = fake_mode.from_tensor(template, static_shapes=False).shape[0]
    # Tested the way every record in this file is read -- a free symbol prints
    # as `s<n>` and a specialised one prints as its value -- rather than by
    # type, so the check and the assertions downstream of it agree by
    # construction.
    if not re.fullmatch(r"s\d+", str(axis)):
        raise AssertionError(
            f"the step axis came back as {str(axis)!r}, not a symbol: a hint "
            f"of {hint} is specialised by torch, so there is no free width to "
            f"trace. Trace at a width of at least {MIN_STEP_WIDTH}."
        )
    return axis


def _resolve_on_the_host(tree_root):
    """Let a symbolic count answer a host question with its hint, and record it.

    This is what makes the step-symbol pass possible, and it is aimed at the
    root the three ordered sites are symptoms of: **`SymInt.__index__` and
    `SymInt.__int__` are `guard_int`.** ATOM's staging has to fill `self.np[:n]`
    and slice a Python list, and both of those need a number. The conversion is
    not the problem. The problem is the *recording*: `guard_int` installs
    `Eq(s, hint)` and every shape downstream of the symbol is a constant from
    there, with no error and no warning.

    So the conversion is kept and the recording is replaced by this record.
    `__index__` and `__int__` return the symbol's hint, which is the count ATOM
    computed, and every one of them is logged with the ATOM frames it happened
    through. Nothing is hidden, and nothing is taken on trust: the log is
    compared, as a multiset of `(ATOM line, conversions)`, against the declared
    `EXPECTED_HOST_RESOLUTIONS`, at both group widths and both step widths. A
    conversion at a line this capture did not expect fails that comparison
    rather than joining the log unremarked -- which is the whole of what makes
    this an instrument rather than a decoration, and which this docstring
    claimed before anything implemented it.

    What is *not* touched is `__bool__` and the comparisons that reach it. A
    branch on a symbol still installs its guard, so a step whose shape decides
    which path ATOM takes still records that it did.

    `_HintSlicedView` above is the narrow form of the same idea, applied to one
    buffer's numpy view to simulate the first site closed. This is the general
    form, and it is why the sites did not have to be closed one at a time.
    """
    original_index = torch.SymInt.__index__
    original_int = torch.SymInt.__int__
    events = []

    def resolved(self):
        hint = self.node.hint
        if hint is None:
            # No hint to answer with -- an unbacked symbol. Let torch do what
            # it does, which is raise rather than guess.
            return original_index(self)
        events.append(
            {"value": str(self), "hint": int(hint), "frames": _atom_frames(tree_root)}
        )
        return int(hint)

    torch.SymInt.__index__ = resolved
    torch.SymInt.__int__ = resolved
    return events, (original_index, original_int)


def _capture(
    tp, tmpdir, symbolic, repair_site_one=False, step_symbol=False, width=DECODE_SEQS
):
    """Trace one decode step of the published model at width `tp`.

    Three passes, because they answer different questions and cannot be one
    run. The capture ATOM produces is `symbolic=False`: the staged buffers are
    concrete on both sides, as ATOM builds them, and what comes out is the
    inventory -- the operators, the collectives, and a shape census that is the
    claim about the inventory rather than about the mechanism.

    `symbolic=True` is the probe. It gives every staged buffer's device side a
    free symbol per dimension and hands `prepare_decode` its row count as a
    symbol too, then records where each one stopped being free. Symbols that
    survive are not evidence of anything here: a handful reach `as_strided`,
    `reshape` and `slice` on views that no compute operator consumes.

    `repair_site_one=True` is the same probe with the first of those places
    simulated closed, from outside ATOM (`_HintSlicedView`). It exists because
    the specialisation sites are **ordered**, not independent: closing the one
    that solves the bound first does not close the bound, it moves it. The
    record that pass produces is the source for the third site.

    `step_symbol=True` is the fourth pass, and the only one that specialises
    nowhere. It does not symbolise the staged buffers at all -- their
    dimensions are engine capacities and are not a function of the step -- and
    it hands no bound to `prepare_decode`. The step's *width* is a free symbol
    on the `ScheduledBatch` and ATOM derives every bound from it, while
    `_resolve_on_the_host` lets the symbol answer a host fill with its hint
    instead of with a guard. `width` is the hint, and tracing the same step at
    two of them is what says the symbol is free rather than assumed free."""
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    declared_cuda, cuda_reads = _declare_cuda()
    declared_arch = _declare_arch(tmpdir)
    declared_priming = _decline_context_priming()
    declared_init = _decline_initialisers()
    declared_gather = _decline_custom_all_gather()

    tree_root = pathlib.Path(__file__).resolve().parents[2]
    model_dir = pathlib.Path(tmpdir) / "published-config"
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = CONFIG_JSON.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONFIG_SHA256:
        raise RuntimeError(
            f"{CONFIG_JSON.name} is not the published config: sha256 {digest}, "
            f"expected {CONFIG_SHA256} for Qwen/Qwen3.8-27B at {CONFIG_REVISION}"
        )
    (model_dir / "config.json").write_bytes(payload)

    group_width, declared_transport, gather_buffers = _build_group(tp)

    from atom.config import Config, set_current_atom_config

    config = Config(
        model=str(model_dir),
        tensor_parallel_size=tp,
        load_dummy="empty",
        enforce_eager=True,
        kv_cache_block_size=BLOCK_SIZE,
    )
    set_current_atom_config(config)
    if config.tp_world_size != tp:
        raise RuntimeError(
            f"tp_world_size is {config.tp_world_size} at tensor_parallel_size "
            f"{tp}: the width has to be the group's, not a simulated one"
        )

    simulated_tp_calls = _watch_simulated_tp(tree_root)
    shape_env = ShapeEnv()
    fake_mode = _driverless_mode(shape_env)
    original_buffer_init, buffer_init = _stage_buffers(
        fake_mode, symbolic, tree_root, repair_site_one
    )
    resolutions, original_symint = ([], None)
    try:
        architecture, model_class_name, runner = _build_runner(config, fake_mode)
        bounds = _symbolic_bound(runner, fake_mode) if symbolic else []
        axis = _step_axis(fake_mode, width) if step_symbol else width
        if step_symbol:
            # Before the batch, not after it. `ScheduledBatch.__init__` already
            # asks its own counts for a number, and a conversion that happens
            # before this is in place takes the guard and makes the axis a
            # constant before the step has begun.
            resolutions, original_symint = _resolve_on_the_host(tree_root)
        recorder = _Recorder(tree_root, COLLECTIVE_OPS)
        triton = _TritonLaunches()
        specialised = _Specialisations(shape_env, tree_root)
        with specialised:
            batch = _decode_batch(width, axis if step_symbol else None)
            with fake_mode:
                runner.allocate_kv_cache(KV_BLOCKS)
                with torch._C._EnablePythonDispatcher(), triton, recorder:
                    runner.forward(batch)
    finally:
        from atom.utils import CpuGpuBuffer

        CpuGpuBuffer.__init__ = original_buffer_init
        if original_symint is not None:
            torch.SymInt.__index__, torch.SymInt.__int__ = original_symint

    import atom

    entries, non_numeric = recorder.shape_census()
    return {
        # Which `atom` package this record came from. A capture that cannot
        # name the tree it traced is not an observation about that tree, and
        # an `atom` resolved from somewhere else fails silently wherever both
        # trees have the symbol.
        "atom_package": atom.__file__,
        "tp": tp,
        "tp_group_world_size": group_width,
        "symbolic_staging": symbolic,
        "site_one_repair_simulated": repair_site_one,
        # The step-symbol pass declares itself, so a reader of the record never
        # has to infer which of the four arrangements produced it.
        "step_symbol": step_symbol,
        "step_axis": str(axis),
        "step_axis_hint": width,
        # Every call the sentinel saw, with its ATOM frames. Empty today by
        # construction, not by observation: see `_watch_simulated_tp`.
        "apply_simulated_tp_calls": simulated_tp_calls,
        # Which `CpuGpuBuffer.__init__` ran, and what it asked for. ATOM's own
        # body executes here, so a change to it moves one of these counts.
        "buffer_init": buffer_init,
        # Diagnostic because a raw Triton kernel was reached and skipped, and
        # everything downstream of one read uninitialised fake memory. Derived
        # from the launches rather than written, so it says what happened.
        "diagnostic_inventory": bool(triton.launches),
        "model": {
            "config_sha256": digest,
            "config_revision": CONFIG_REVISION,
            "architecture": architecture,
            "model_class": model_class_name,
            "dtype": str(config.torch_dtype),
        },
        "declared": {
            "cuda": declared_cuda,
            "cuda_reads": dict(cuda_reads),
            "arch": declared_arch,
            "context_priming": declared_priming,
            "initialisers": declared_init,
            "transport": declared_transport,
            "config": declared_gather,
        },
        "ops": len(recorder.ops),
        "distinct_ops": recorder.distinct_ops(),
        "collectives": recorder.collectives,
        # The two arrangements of the vocab gather, both measured off the live
        # call: the buffer ATOM passes, and the one the functional substitute
        # produces before the shim reshapes into it.
        "gather_buffers": gather_buffers,
        "shape_entries": entries,
        "non_numeric_shape_entries": non_numeric,
        "non_numeric_ops": recorder.non_numeric_ops(),
        "family_census": recorder.family_census(),
        "concrete_dims": recorder.concrete_dims(),
        "graph_digest": recorder.graph_digest(axis),
        "symbolic_sites": [
            {"op": op, "call_site": site, "n": n}
            for (op, site), n in sorted(recorder.symbolic_sites.items())
        ],
        "host_resolutions": resolutions,
        "injected_bounds": bounds,
        "specialisations": specialised.events,
        "shape_env_replacements": {
            str(k): str(v) for k, v in shape_env.replacements.items()
        },
        # Every guard, not a count of them: a guard is the record of where the
        # traced graph stops being valid, so one that nobody intended is a
        # narrower artifact than it looks and has to be read rather than
        # totalled. The symbol's name is anonymised for the same reason it is
        # in the digest -- it is the order a ShapeEnv created it in, not a
        # property of the step.
        "shape_env_guards": sorted(
            {_anonymise(str(g.expr), axis) for g in shape_env.guards}
        ),
        "triton_launches": triton.launches,
    }


def main(argv):
    import argparse
    import atexit
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--symbolic", action="store_true")
    parser.add_argument("--repair-site-one", action="store_true")
    parser.add_argument("--step-symbol", action="store_true")
    parser.add_argument("--width", type=int, default=DECODE_SEQS)
    args = parser.parse_args(argv)
    if args.repair_site_one and not args.symbolic:
        parser.error("--repair-site-one is a probe on the symbolic pass")
    if args.step_symbol and args.symbolic:
        parser.error(
            "--step-symbol and --symbolic are two arrangements of the staging, "
            "not two options on one"
        )
    if args.width < MIN_STEP_WIDTH:
        # Refusing rather than emitting. A width of 1 traces without error and
        # exits 0, and the record it writes says `"step_symbol": true` over a
        # `step_axis` of `"1"`, no non-numeric shape entries, an empty host
        # resolution log and an empty guard list -- a family census identical
        # to the concrete control's, wearing the label of the symbolic one.
        # Torch's 0/1 specialisation is documented and is not a leak in this
        # capture, but a pass that reports success while producing a concrete
        # record is the shape of the thing this file was built to catch, so it
        # is the one outcome the capture will not print.
        parser.error(
            f"--width {args.width} traces no symbol: torch specialises a size "
            f"hint of 0 or 1 to a constant, so the record would be concrete "
            f"and would say it was symbolic. The narrowest traceable width is "
            f"{MIN_STEP_WIDTH}."
        )
    # At exit, so it is printed even when a new device read ends the capture.
    atexit.register(
        lambda: print(READS_MARKER + json.dumps(sorted(_CUDA_READS_OUTSIDE_TORCH)))
    )
    with tempfile.TemporaryDirectory(prefix="compass-capture-") as tmpdir:
        record = _capture(
            args.tp,
            tmpdir,
            args.symbolic,
            args.repair_site_one,
            args.step_symbol,
            args.width,
        )
    sys.stdout.write(RECORD_MARKER + json.dumps(record) + "\n")
    return 0


# ---------------------------------------------------------------------------
# the tests -- everything below runs under pytest, in the parent process


RECORD_MARKER = "CAPTURE-RECORD "
READS_MARKER = "CAPTURE-CUDA-READS "


_RECORDS: dict[tuple, dict] = {}
_READS: dict[tuple, frozenset] = {}


def capture(
    tp, symbolic=False, repair_site_one=False, step_symbol=False, width=DECODE_SEQS
):
    """Run this file as a script at width `tp` and read back its record.

    Memoised: six of these would otherwise be nine, and each one builds the
    64-layer module tree.
    """
    import pytest

    key = (tp, symbolic, repair_site_one, step_symbol, width)
    if key in _RECORDS:
        return _RECORDS[key]
    tree_root = pathlib.Path(__file__).resolve().parents[2]
    argv = [sys.executable, str(pathlib.Path(__file__).resolve()), "--tp", str(tp)]
    if symbolic:
        argv.append("--symbolic")
    if repair_site_one:
        argv.append("--repair-site-one")
    if step_symbol:
        argv.append("--step-symbol")
    if width != DECODE_SEQS:
        argv += ["--width", str(width)]
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tree_root)},
        cwd=str(tree_root),
    )
    for line in completed.stdout.splitlines():
        if line.startswith(READS_MARKER):
            _READS[key] = frozenset(json.loads(line[len(READS_MARKER) :]))
        if line.startswith(RECORD_MARKER):
            _RECORDS[key] = json.loads(line[len(RECORD_MARKER) :])
    if key in _RECORDS:
        return _RECORDS[key]
    pytest.fail(
        f"the capture at TP{tp} (symbolic={symbolic}, "
        f"repair_site_one={repair_site_one}, step_symbol={step_symbol}, "
        f"width={width}) produced no record; the "
        f"subprocess exited {completed.returncode}.\n"
        f"--- stderr tail ---\n{completed.stderr[-4000:]}"
    )


def cuda_reads(
    tp, symbolic=False, repair_site_one=False, step_symbol=False, width=DECODE_SEQS
):
    """Every `torch.cuda` name read outside torch, even by a capture that failed.

    So a new device read is reported by its name, not by the error it raised.
    """
    import pytest

    key = (tp, symbolic, repair_site_one, step_symbol, width)
    try:
        capture(*key)
    except pytest.fail.Exception:
        if key not in _READS:
            raise
    return _READS[key]


def row_parallel_reduces():
    """How many row-parallel reduces one forward owes, from the config alone.

    Derived from the published config's geometry rather than read off the
    inventory, so the count has a source that is not the thing it checks. Every
    weight sharded along its input dimension reduces once per forward through
    `tensor_model_parallel_all_reduce`: each layer's `mlp.down_proj`, each
    full-attention layer's `self_attn.o_proj`, and each linear-attention
    layer's `linear_attn.out_proj`. The vision tower shards none.

    What the expression is sensitive to, stated rather than implied: given the
    assertion below that the two attention kinds account for every layer, the
    sum is `2 x num_hidden_layers` and does **not** depend on how the layers
    divide between them. The reduce per layer is the `mlp.down_proj`; the
    second is one attention output projection whichever kind the layer is. The
    split is asserted because a third layer kind would break the identity, not
    because the total counts it.
    """
    text = json.loads(CONFIG_JSON.read_bytes())["text_config"]
    layer_types = text["layer_types"]
    assert len(layer_types) == text["num_hidden_layers"]
    full = layer_types.count("full_attention")
    linear = layer_types.count("linear_attention")
    assert full + linear == len(layer_types)
    return len(layer_types) + full + linear


def raw_triton_launches():
    """The raw Triton kernels one decode step reaches, from the config alone.

    Each full-attention layer normalises its query and key and applies the
    multimodal rotary embedding through one raw kernel each -- `qk_norm` and
    `try_mrope_qk_fused` in `Qwen3NextAttention.forward` -- and a linear-
    attention layer launches neither. The step converts its block tables to KV
    indices once, whatever the layer count. So the counts follow the config's
    layer split rather than being read off the inventory they check.
    """
    layer_types = json.loads(CONFIG_JSON.read_bytes())["text_config"]["layer_types"]
    full = layer_types.count("full_attention")
    return {
        "_fused_qk_norm_single_kernel": full,
        "_mrope_qk_kernel": full,
        "kv_indices_generate_kernel": 1,
    }


def site(event, depth):
    """One specialisation as `(value, the innermost `depth` ATOM frames)`.

    Only the innermost frames are the site. The path that reaches it runs
    through the runner and the eplb wrapper, whose line numbers move for
    reasons that have nothing to do with where a symbol was solved.
    """
    return event["value"], tuple(event["frames"][-depth:])


def test_the_published_config_is_the_one_that_was_published():
    """The fixture is the checkpoint's own file, not a description of it.

    A synthetic config is the thing this tree deleted once already, and the
    difference between one and the published file is invisible in every number
    downstream of it.
    """
    payload = CONFIG_JSON.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == CONFIG_SHA256
    assert json.loads(payload)["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert json.loads(payload)["text_config"]["max_position_embeddings"] == 262144


def test_a_decode_step_traces_at_both_widths():
    """The forward completes at TP1 and TP2 and yields an inventory.

    The assertion is the **distinct**-operator count, not the total. A total
    moves with any change to ATOM's forward -- a fused kernel, one more view --
    and a test that pins one is a test that is edited every time it fails. The
    distinct set moves when the *kinds* of work change, which is the thing worth
    holding.

    Both inventories are diagnostic, and the record says so because it counted
    the raw Triton kernels it skipped, not because a literal says it: 33
    launches over three kernels, the same at both widths.
    """
    tp1 = capture(1)
    tp2 = capture(2)
    tree_root = str(pathlib.Path(__file__).resolve().parents[2])
    for record in (tp1, tp2):
        assert record["atom_package"].startswith(tree_root)
        assert record["ops"] > 0
        assert record["triton_launches"] == raw_triton_launches()
        assert record["diagnostic_inventory"] is True
        assert record["model"]["architecture"] == "Qwen3_5ForConditionalGeneration"
    assert len(tp1["distinct_ops"]) == TP1_DISTINCT_OPS
    assert len(tp2["distinct_ops"]) == TP2_DISTINCT_OPS
    assert set(tp1["distinct_ops"]) - set(tp2["distinct_ops"]) == set(TP1_ONLY_OPS)
    assert set(tp2["distinct_ops"]) - set(tp1["distinct_ops"]) == set(TP2_ONLY_OPS)


def test_the_width_is_the_group_s_and_nothing_simulated_it():
    """The group's width is measured; `apply_simulated_tp` is guarded, not observed.

    `tp_group_world_size` is read from `get_tp_group().world_size`, so it is
    what the group has rather than what the config asked for -- the one width
    figure in the record that a substitution could not fake, and the
    measurement half of this test.

    `apply_simulated_tp_calls` is the other half, and **it is a forward guard,
    not an observation about ATOM at TP2 today.** ATOM's one call site -- held
    by the source scan in the next test -- is in
    `ModelRunner._setup_device_and_distributed`, which `_build_runner`
    overrides, so the list cannot be anything but empty: a sentinel whose body
    is `raise SystemExit` leaves this test passing. It earns its place against
    the future -- an ATOM that calls the function from a path the capture does
    run fails here with the frames that called it -- and it replaced
    `assert record["apply_simulated_tp"] is False` against a literal written
    into the record, which could not fail at all. It matters because a TP>1
    inventory taken through `apply_simulated_tp` both erases and fabricates:
    129 `all_reduce` become the identity and appear nowhere, and one
    `all_gather` becomes six dispatched operators over a half-zeros tensor.
    """
    for record in (capture(1), capture(2), capture(1, symbolic=True)):
        assert record["tp_group_world_size"] == record["tp"]
        assert record["apply_simulated_tp_calls"] == []


def test_apply_simulated_tp_is_called_only_where_the_capture_does_not_go():
    """The premise under the sentinel, held by reading ATOM rather than stated.

    The sentinel in `_watch_simulated_tp` is a forward guard because ATOM calls
    `apply_simulated_tp` from exactly one method, and `_build_runner` overrides
    that method. Nothing checked the first half: a second call site in another
    overridden method would leave every capture test green and the account
    false. So every call to the name, anywhere under `atom/`, is listed with the
    function it sits in, and the list is held.

    Calls are matched by name, as a bare name or as an attribute, which is how
    both bindings of it are spelled. A call through an alias under another name
    would not be seen.
    """
    tree_root = pathlib.Path(__file__).resolve().parents[2]
    scanned = 0
    callers = set()
    for path in sorted((tree_root / "atom").rglob("*.py")):
        scanned += 1
        module = ast.parse(path.read_text(), filename=str(path))
        for function in ast.walk(module):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                called = getattr(node.func, "id", getattr(node.func, "attr", None))
                if called == "apply_simulated_tp":
                    rel = path.relative_to(tree_root).as_posix()
                    callers.add((rel, function.name))
    assert scanned > 100
    assert callers == {
        ("atom/model_engine/model_runner.py", "_setup_device_and_distributed")
    }
    # And that method is the one the capture's runner replaces.
    assert "def _setup_device_and_distributed" in inspect.getsource(_build_runner)


def test_a_call_through_either_binding_reaches_the_sentinel():
    """A call through either binding of `apply_simulated_tp` lands in the sentinel.

    No capture reaches ATOM's call site, so this calls both names on a `None`
    config. A binding left out, or a sentinel that calls through, runs the real
    function, which raises on `None.tensor_parallel_size`. The child also
    refuses any `atom` package but the one this process imported.
    """
    probe = (
        "import importlib.util, pathlib, sys, tempfile\n"
        "spec = importlib.util.spec_from_file_location('capture', sys.argv[1])\n"
        "capture = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(capture)\n"
        "capture._declare_cuda()\n"
        "with tempfile.TemporaryDirectory() as tmpdir:\n"
        "    capture._declare_arch(tmpdir)\n"
        "    calls = capture._watch_simulated_tp(pathlib.Path(sys.argv[2]))\n"
        "    from atom.distributed import simulated_tp\n"
        "    from atom.model_engine import model_runner\n"
        "    model_runner.apply_simulated_tp(None)\n"
        "    simulated_tp.apply_simulated_tp(None)\n"
        "from atom import __file__ as atom_file\n"
        "assert pathlib.Path(atom_file).resolve() == pathlib.Path(sys.argv[3]), atom_file\n"
        "print('SENTINEL-CALLS', len(calls))\n"
    )
    tree_root = pathlib.Path(__file__).resolve().parents[2]
    import atom

    atom_init = pathlib.Path(atom.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, "-c", probe, __file__, str(tree_root), str(atom_init)],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tree_root)},
        cwd=str(tree_root),
    )
    lines = completed.stdout.splitlines()
    assert "SENTINEL-CALLS 2" in lines, completed.stderr[-4000:]


def test_atom_s_own_buffer_constructor_is_what_runs():
    """`CpuGpuBuffer.__init__` executes here, and the record says how.

    The pin on the second specialisation site is worth only as much as the
    code it lets run. An earlier version of this file replaced `__init__`
    wholesale, and a `raise` as its first statement then changed nothing
    anywhere in this file -- so a repair to site two, a symbolic
    `CpuGpuBuffer`, could have gone in and this test would still have reported
    the site unrepaired. It is ATOM's body that runs now,
    with three primitives staged around it, and these counts are what says so.

    The counts are also the pin on the body itself, which no specialisation
    site covers: one host allocation and one numpy view per buffer, and in the
    symbolic pass one device-side allocation per buffer through
    `torch.zeros_like`. A repair that allocates differently moves one of them.
    """
    tree_root = pathlib.Path(__file__).resolve().parents[2]
    concrete = capture(1)
    symbolic = capture(1, symbolic=True)
    for record in (concrete, symbolic):
        init = record["buffer_init"]
        assert init["source"] == BUFFER_INIT
        assert (tree_root / init["source"].split(":")[0]).exists()
        assert init["constructed"] == BUFFER_COUNT
        # ATOM allocates the host side once per buffer and takes one numpy
        # view of it; both run outside the mode. A buffer that stopped doing
        # either, or one more buffer, moves one of these.
        assert init["host_allocations"] == BUFFER_COUNT
        assert init["numpy_views"] == BUFFER_COUNT
    # The device side is staged only in the symbolic pass; in the concrete one
    # ATOM's own `torch.zeros_like` runs unaltered and nothing counts it.
    assert concrete["buffer_init"]["symbolic_device_allocations"] == 0
    symbolic_init = symbolic["buffer_init"]
    assert symbolic_init["symbolic_device_allocations"] == symbolic_init["constructed"]


def test_the_stubbed_device_names_are_the_ones_the_capture_reads():
    """Which `torch.cuda` names a capture reads, counted rather than listed.

    The stub list is what `_declare_cuda` wrote, so asserting its length would
    assert the function against itself. What is measured is every read of a
    stubbed name over a whole capture, and that says which stubs are doing
    anything: all 17, the same at both widths and in every pass. The stub list
    is held equal to that read set, so a stub nothing reads fails here as
    surely as a stub that stops being read. Every name read outside torch,
    stubbed or not, is held equal to its own measured set, so a new device read
    fails here by its name on any host, including one the capture does not
    survive.
    """
    for tp in (1, 2):
        for arrangement in (
            {},
            {"symbolic": True},
            {"symbolic": True, "repair_site_one": True},
            {"step_symbol": True},
            {"step_symbol": True, "width": SECOND_WIDTH},
        ):
            if tp == 2 and arrangement.get("symbolic"):
                continue
            reads = cuda_reads(tp, **arrangement)
            assert reads == CUDA_NAMES_READ_OUTSIDE_TORCH, (tp, arrangement)
            record = capture(tp, **arrangement)
            cuda = record["declared"]["cuda"]
            declared = set(cuda["import"]) | set(cuda["model_runner"])
            assert set(record["declared"]["cuda_reads"]) == CUDA_NAMES_READ, (
                tp,
                arrangement,
            )
            assert declared == CUDA_NAMES_READ


def test_the_collectives_at_tp2_are_recorded_by_name_and_call_site():
    """TP2 records ATOM's real collectives, at the lines that issue them.

    By name and by site, never by a total. A count is the figure that goes stale
    first and says least: it cannot distinguish one collective moving to a
    different call site from the layer count changing, and it has gone stale
    twice on this project already.

    Six rows, not four: the two `wait_tensor` entries are the functional
    substitution's own operators rather than ATOM's, which is a reason to
    label them and not a reason to leave them out of a decomposition.

    TP1 issues none, which is the control: a group of width 1 shortcuts every
    reduce, so a collective appearing there would mean the recorder was
    counting something else.
    """
    assert capture(1)["collectives"] == []

    by_site = collections.Counter(
        (entry["op"], entry["call_site"]) for entry in capture(2)["collectives"]
    )
    assert dict(by_site) == {
        ("aiter.all_reduce_.default", ROW_PARALLEL): row_parallel_reduces(),
        ("aiter.all_reduce_.default", VOCAB_EMBEDDING): 1,
        ("_c10d_functional.all_gather_into_tensor.default", VOCAB_LM_HEAD): 1,
        ("_c10d_functional.wait_tensor.default", VOCAB_LM_HEAD): 1,
        ("_c10d_functional.broadcast.default", SAMPLER): 1,
        ("_c10d_functional.wait_tensor.default", SAMPLER): 1,
    }
    # The gather shapes say what the width did: one rank's vocab slice arrives
    # as the whole vocabulary.
    gathered = [
        entry["shapes"][0]
        for entry in capture(2)["collectives"]
        if entry["op"].endswith("all_gather_into_tensor.default")
    ]
    assert gathered == [["2", "124160"]]


def test_the_two_arrangements_of_the_vocab_gather_are_both_measured():
    """ATOM's gather buffer and the substitute's, neither one inferred.

    The dispatched operator carries only its input, so the destination shape
    is not in the inventory at all and stating one from the width would be
    arithmetic wearing a measurement's clothes. Both are read off the live
    call instead: ATOM stages the gather into `(world_size,) + input_size`,
    and the functional form the substitution routes to concatenates along
    dim 0 before the shim reshapes into ATOM's buffer.

    Which arrangement a shape belongs to is the distinction, because only the
    first is a fact about ATOM at TP2 and only the second is a fact about this
    capture's substitution.
    """
    assert capture(1)["gather_buffers"] == []
    assert capture(2)["gather_buffers"] == [
        {
            "op": "all_gather_into_tensor",
            "input": ["2", "124160"],
            "atom_output": ["2", "2", "124160"],
            "functional_output": ["4", "124160"],
        }
    ]


def test_the_inventory_is_concrete_at_both_widths():
    """No shape entry in either inventory is anything but a plain integer.

    That is the finding, not a defect in the tracing: the mechanism keeps free
    symbols perfectly well, and the symbolic probe below puts one on every
    staged dimension and watches ATOM's own path solve them. A concrete
    inventory is valid only at the shapes it was taken at, so this is the
    sentence that stops anyone evaluating one anywhere else.

    A zero is a fraction, and both halves are held. The denominator is held by
    family: which families there are, that each carries entries, and that they
    add up to the total, so "none of the GEMM entries is symbolic" is about
    GEMM entries that were recorded. The total's floor stays beside it, so a
    recorder that stopped recording cannot read as a concrete one. The counts
    themselves are not held; `CONCRETE_FAMILIES` says why and what that costs.
    The numerator's zero is held by the test after this one, which shows the
    same census reading a number other than zero when there is a symbol to read.
    """
    for tp in (1, 2):
        record = capture(tp)
        assert record["shape_entries"] > 10000
        census = record["family_census"]
        by_family = {family: row["shape_entries"] for family, row in census.items()}
        assert set(by_family) == CONCRETE_FAMILIES[tp], tp
        assert all(entries > 0 for entries in by_family.values()), (by_family, tp)
        assert sum(by_family.values()) == record["shape_entries"], tp
        assert record["non_numeric_shape_entries"] == 0
        assert record["non_numeric_ops"] == []
        for family, row in census.items():
            assert row["non_numeric_shape_entries"] == 0, (family, tp)


def test_the_census_counts_the_symbols_the_probe_leaves():
    """The concreteness detector, read where there is something to detect.

    The concrete inventory's census is zero, and zero is also what a census
    returns when it cannot tell a symbol from a number. So the census is read
    on the symbolic probe, which leaves a handful of symbols free on staging
    views, and its four loops over the inventory are held against each other:
    the total (`shape_census`), the family split (`family_census`), the
    operators carrying one (`non_numeric_ops`) and the call sites where they
    were dispatched (`symbolic_sites`). Each is counted separately, so a loop
    that stops discriminating disagrees with the others, or reads zero where
    the kinds below say there is something, and fails here by name rather than
    agreeing with the concrete zero.

    Kinds are held and counts are not: which families and which operators
    carry a symbol, not how many entries. `PROBE_FAMILIES` says why.
    """
    for repaired in (False, True):
        record = capture(1, symbolic=True, repair_site_one=repaired)
        census = record["family_census"]
        by_family = {
            family: row["non_numeric_shape_entries"]
            for family, row in census.items()
            if row["non_numeric_shape_entries"]
        }
        assert set(by_family) == PROBE_FAMILIES[repaired], repaired
        total = record["non_numeric_shape_entries"]
        assert total == sum(by_family.values()) > 0, repaired
        assert record["non_numeric_ops"] == PROBE_NON_NUMERIC_OPS[repaired]
        # Operator by operator, the family loop and the call-site loop count
        # the same dispatches.
        from_families = collections.Counter()
        for row in census.values():
            from_families.update(row["ops_carrying_a_symbol"])
        from_sites = collections.Counter()
        for entry in record["symbolic_sites"]:
            from_sites[entry["op"]] += entry["n"]
        assert from_families == from_sites, repaired
        assert sorted(from_families) == record["non_numeric_ops"], repaired


def test_the_census_counts_an_expression_as_a_symbol():
    """A dimension is numeric only if it is an integer; an expression is not.

    Which expressions the probe happens to produce is a property of ATOM's
    staging, not of the census, so the rule is held here on a recorder built
    by hand, with no capture: a negative integer is a number, and each
    expression -- `2*s0`, `s0 + 1` -- is one non-numeric entry. A detector
    narrowed to "starts with s" fails here whatever the probe produces.
    """
    recorder = _Recorder(pathlib.Path(__file__).resolve().parents[2], COLLECTIVE_OPS)
    recorder.ops = [
        ("aten.view.default", [["2*s0", "4"]], [["s0 + 1"]]),
        ("aten.add.Tensor", [["-1", "8"]], []),
    ]
    assert recorder.shape_census() == (5, 2)
    assert recorder.non_numeric_ops() == ["aten.view.default"]
    census = recorder.family_census()
    assert census["view"]["non_numeric_shape_entries"] == 2
    assert census["elementwise"]["non_numeric_shape_entries"] == 0
    assert _Recorder._carries_symbol([["2*s0", "4"]], [])
    assert not _Recorder._carries_symbol([["-1", "8"]], [])


def test_the_first_two_specialisation_sites_are_where_they_were_measured():
    """The two places a free symbol stops being free first.

    Not two independent sites. They are the first two in an order: the bound
    the caller passes is solved at site one because that is the first line to
    take `__index__` of it, and the buffer's own dimension is solved in
    `copy_to_gpu` because that is the first copy whose slice does not cover
    it. Both are pinned by value and by the frames they happened through, so a
    repair to either one fails here. What lies behind the first of them is the
    test below.
    """
    record = capture(1, symbolic=True)
    injected = record["injected_bounds"]
    assert len(injected) == 1 and injected[0]["hint"] == DECODE_SEQS
    # A bound that arrived as a plain `int` would specialise nothing and this
    # test would pass by measuring the absence of a question.
    assert re.fullmatch(r"s\d+", injected[0]["symbol"])
    assert record["site_one_repair_simulated"] is False

    one, two = record["specialisations"]
    assert site(one, 1) == SITE_ONE
    assert site(two, 2) == SITE_TWO
    # Two symbols solved at two places: the buffer's dimension is not the
    # caller's bound, so repairing the bound leaves the buffer as it is.
    assert one["symbol"] == injected[0]["symbol"]
    assert two["symbol"] != one["symbol"]


def test_closing_site_one_moves_the_bound_to_a_third_site():
    """Repairing the first site does not close the bound; it relocates it.

    The two sites were recorded as independent, with a symbolic bound closing
    the first and leaving the second as it is. Half of that is true. This is
    the other half, and it is the reason the sites are an order rather than a
    set: with the numpy view reading the bound's hint instead of solving it --
    the simulated site-one repair, applied from outside ATOM -- the bound
    survives site one and is solved in another file and a later phase of the
    step, inside ATOM's own shape-contract assertion: `assert_shape_contract`
    in `atom/utils/forward_context.py`, reached from the `run_model` call in
    `ModelRunner.forward`, with `prepare_inputs` already returned -- not
    further down `prepare_decode`.

    The second site is untouched by the repair, exactly as recorded.

    **What the third site turned out to be.** It was recorded one frame deeper,
    at `_rows`'s `int(t.shape[0])`, which converted a dimension before the
    assertion compared it. `_rows` no longer converts -- it returns
    `t.shape[0]` as it is -- and the assertion is still reached, because this
    probe hands the caller a bound that is a *different symbol* from the one the
    staged buffer carries, and ATOM's contract is that the two are one number.
    Equating them is the contract working. It is not what made the capture
    concrete, and it does not fire when the whole step is one symbol.
    """
    record = capture(1, symbolic=True, repair_site_one=True)
    assert record["site_one_repair_simulated"] is True
    assert record["buffer_init"]["hint_sliced_views"] > 0

    two, three = record["specialisations"]
    # Site two, unchanged by the repair -- which is the half of the two-site
    # account that holds.
    assert site(two, 2) == SITE_TWO
    assert site(three, 1) == SITE_THREE
    # Still the caller's bound, solved somewhere else: the value is the same
    # and the site is not.
    assert three["value"] == SITE_ONE[0]
    assert record["injected_bounds"][0]["symbol"] == three["symbol"]
    # And `_rows` is not where it happens any more: nothing on the path
    # converts a dimension to a number before the comparison.
    assert not any("in _rows" in frame for frame in three["frames"])


def test_nothing_specialises_under_the_step_symbol_capture():
    """No symbol is solved anywhere in a traced step, and here are the guards.

    `ShapeEnv._set_replacement` is the one funnel every replacement goes
    through -- its own docstring says to use it rather than assigning into
    `replacements` -- so an empty `replacements` at the end is not a statement
    about the routes this capture happened to think of.

    **But in this pass it is a weaker statement than that makes it sound, and
    the weakness is worth naming rather than leaving for a reader to find.**
    Two of the routes into that funnel, `__index__` and `__int__`, have been
    replaced by `_resolve_on_the_host` for the duration of the capture, so they
    cannot reach `_set_replacement` at all: for those two the assertion below
    is guaranteed by construction rather than tested. What it still tests is
    every other route -- a comparison, a hash, a format string, a `guard_size_
    oblivious`, anything inside torch that decides it needs a number -- and
    those are live, untouched and able to fail this line.

    **The load-bearing evidence is the two-width digest**, in
    `test_the_symbol_is_free_across_the_step_width` below, not this assertion.
    A symbol can be *lost* without ever being solved, by code reading its hint
    and building a tensor of that size, and no replacement is recorded when
    that happens. That class shows up only as an inventory that differs between
    two widths.

    The guards are asserted as a set and not as a count. Each is the record of
    where the traced step stops being valid -- the staging buffers' capacities,
    expressed against the step's width -- and a guard nobody intended is a
    narrower artifact than it looks. The set is a function of the width as well
    as of `tp`; see `expected_guards`.
    """
    for tp in (1, 2):
        record = capture(tp, step_symbol=True)
        assert record["step_symbol"] is True
        # The staged buffers are ATOM's own, at ATOM's own capacities: this
        # pass repairs nothing in the buffer, which is the finding.
        assert record["symbolic_staging"] is False
        assert record["buffer_init"]["symbolic_device_allocations"] == 0, tp
        assert record["buffer_init"]["constructed"] == BUFFER_COUNT, tp
        assert record["injected_bounds"] == [], tp
        # A capture whose axis never became a symbol would satisfy everything
        # below by having no question to answer.
        assert re.fullmatch(r"s\d+", record["step_axis"]), tp
        assert record["step_axis_hint"] == DECODE_SEQS
        assert record["shape_env_replacements"] == {}, tp
        assert record["specialisations"] == [], tp
        assert set(record["shape_env_guards"]) == expected_guards(tp, DECODE_SEQS), tp


def test_the_capture_refuses_a_width_that_torch_would_specialise():
    """`--width 1` is refused, because the record it would print reads as a pass.

    Traced at a hint of 1 the step axis is never a symbol -- torch specialises
    0 and 1 by design -- so the capture completes, exits 0 and prints
    `"step_symbol": true` over `"step_axis": "1"`, 0 non-numeric shape entries,
    an empty host-resolution log, empty guards and a family census identical to
    the concrete control's. Nothing in that record says it is concrete except
    the numbers a reader would have to know to check.

    That is the failure mode this file exists to detect wearing the label of
    the result, so the width is refused at the command line rather than
    reported. This test is what makes the refusal fail-able: it runs the real
    entry point and reads the real exit status.
    """
    argv = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--tp",
        "1",
        "--step-symbol",
        "--width",
        "1",
    ]
    tree_root = pathlib.Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tree_root)},
        cwd=str(tree_root),
    )
    # Without `main`'s refusal, `_step_axis` still refuses, but as an uncaught
    # AssertionError: a traceback, and exit 1. 2 is argparse's usage error.
    assert "Traceback" not in completed.stderr
    assert completed.returncode == 2
    assert "traces no symbol" in completed.stderr
    # And no record at all: a refusal that still printed one would be worse
    # than the emission it replaced. Either refusal satisfies this, so it
    # fails only with both gone; the next test holds the second on its own.
    assert RECORD_MARKER not in completed.stdout


def test_the_step_axis_refuses_a_hint_torch_specialises():
    """`_step_axis` refuses a hint of 1 itself; `main` never lets one reach it.

    Held by the raise, not the message: with the check disabled the call
    returns a plain `1`. The narrowest width is the control a symbol passes.
    """
    import pytest
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    fake_mode = FakeTensorMode(shape_env=ShapeEnv())
    assert re.fullmatch(r"s\d+", str(_step_axis(fake_mode, MIN_STEP_WIDTH)))
    with pytest.raises(AssertionError, match="not a symbol"):
        _step_axis(fake_mode, 1)


def test_the_symbol_reaches_the_work_that_decides_the_cost():
    """The census, by operator family, and never as a total.

    A total is the figure two earlier attempts on this property reported: sixty
    shape entries carrying a symbol, every one of them on `prim.device`,
    `as_strided`, `fill_` or `reshape`, and not a single GEMM, attention or
    normalisation among them. Sixty of twelve thousand and five thousand of
    twelve thousand look the same in a total and are not the same result.

    So the claim is per family, the three families that decide what a step
    costs are named, and the operators carrying the symbol inside each of them
    are named too -- a family is not "symbolic" because one view in it is.

    **The empty `unclassified` bucket is the guard that makes the breakdown a
    measurement rather than a description**, and it only works if the matcher
    cannot classify an operator nobody declared. `op_family` matched by prefix
    until this revision, which meant `aten.add` swallowed `aten.addmm` and
    `aten.addbmm` -- two GEMMs -- into `elementwise`, `aten.slice` swallowed
    `aten.slice_scatter` and `aten.select` swallowed `aten.select_scatter`, and
    the bucket could never have fired for exactly the cases it was written for.
    It matches whole operator names now. Nothing this step dispatches is one of
    those four, so every number below is unchanged by the repair; what changed
    is that a future ATOM routing a projection through `addmm` fails here
    instead of quietly understating the three families that decide the cost.
    """
    # Both widths. TP2 is a width that is real rather than simulated, which is
    # where a capture has most to lose, and TP1 is what says the width added
    # nothing.
    for tp in (1, 2):
        at_width = capture(tp, step_symbol=True)["family_census"]
        # An operator nobody has classified would otherwise land in whichever
        # family reads best. There is no such family; there is a bucket that
        # fails.
        assert "unclassified" not in at_width, tp
        for family in COST_BEARING_FAMILIES:
            assert at_width[family]["non_numeric_shape_entries"] > 0, (family, tp)
        assert set(at_width["gemm"]["ops_carrying_a_symbol"]) == {
            "aiter.gemm_a16w16.default"
        }, tp
        assert set(at_width["attention"]["ops_carrying_a_symbol"]) == {
            "aiter.unified_attention_with_output_base.default",
            "aiter.linear_attention_with_output_base.default",
        }, tp
        assert set(at_width["normalisation"]["ops_carrying_a_symbol"]) == {
            "aiter._fused_qk_rmsnorm_group_quant_kernel.default",
            "aten.mean.dim",
            "aten.pow.Tensor_Scalar",
            "aten.rsqrt.default",
        }, tp
    # At TP2 the collectives carry it too, which is the width showing up in
    # the census rather than only in the operator list.
    assert (
        capture(2, step_symbol=True)["family_census"]["collective"][
            "non_numeric_shape_entries"
        ]
        > 0
    )

    # The control, in the same shape: with the step's width a plain integer the
    # same three families carry nothing, which is what makes the rows above a
    # result rather than a description of the model.
    control = capture(1)
    for family in COST_BEARING_FAMILIES:
        row = control["family_census"][family]
        assert row["non_numeric_shape_entries"] == 0, family
    # And the two inventories are the same inventory. Same operators, same
    # count of them, same number of shape entries -- the step traced is the
    # step ATOM traces, and what this changed is what those entries say, not
    # which operators ran.
    symbolic = capture(1, step_symbol=True)
    assert symbolic["ops"] == control["ops"]
    assert symbolic["shape_entries"] == control["shape_entries"]
    # One operator differs, and it is a *value* rather than a shape.
    # `gdn_attn.prepare_decode` writes the step's token count into the tail of
    # a staging tensor; a Python integer is lifted as a constant and a symbol
    # is materialised instead. Same line, same one call, nothing about shape.
    assert set(symbolic["distinct_ops"]) - set(control["distinct_ops"]) == {
        "aten.scalar_tensor.default"
    }
    assert set(control["distinct_ops"]) - set(symbolic["distinct_ops"]) == {
        "aten.lift_fresh.default"
    }
    # Every family runs the same number of operators on both sides, save that
    # the one operator above is an allocation where it used to be a view.
    moved = {"allocation": 1, "view": -1}
    ran = {family: row["ops"] for family, row in symbolic["family_census"].items()}
    assert ran == {
        family: row["ops"] + moved.get(family, 0)
        for family, row in control["family_census"].items()
    }


def test_the_three_sites_are_where_they_were_and_carry_the_symbol_instead():
    """The three ordered sites, held by what now happens at each of them.

    None has moved and none has been removed. What changed is the outcome:

    **Site one**, the bound at the caller. `prepare_decode` still fills a
    staged buffer's numpy view with the count, and a host fill still needs a
    number, so the symbol is still converted there. It is converted to its hint
    and logged as a host resolution, where it used to be converted to a guard
    that made every shape downstream a constant. The line appears in
    `host_resolutions` and in nothing else.

    **Site two**, inside the buffer. `copy_to_gpu` runs the line it always ran,
    and the staged buffer is ATOM's own, unsymbolised. The bound carries the
    symbol, so both slices carry it and the copy dispatches with it; the line
    appears as the call site of operators whose shapes are not numbers.

    **Site three**, `assert_shape_contract`'s `_rows`, the only one that could not
    be closed from outside ATOM: `int(t.shape[0])` converts a dimension the assertion
    then compares against a symbolic width. It no longer converts, so it
    appears nowhere.

    **And the whole log is held, not just those three lines.** The point of
    replacing the recording with a log is that a conversion nobody expected
    fails rather than joining the log unremarked, and a membership test on one
    site does not do that: a seventeenth ATOM line converting the width would
    add an entry and nothing would notice. The multiset of `(ATOM line,
    conversions)` is asserted against `EXPECTED_HOST_RESOLUTIONS` instead, at
    both widths of the group and both step widths, because that is the property
    the instrument is claimed to have.

    This test cannot be passed by removing a site: a `copy_to_gpu` that stopped
    dispatching would lose its `symbolic_sites` entry, and a `prepare_decode`
    that stopped filling the view would lose its `host_resolutions` entry.
    """
    record = capture(1, step_symbol=True)

    # The declared set, on every arrangement this file traces. 20 conversions
    # over 16 ATOM lines, the same at TP1 and TP2 and at both step widths: the
    # lines are host fills and slices, and neither the group's width nor the
    # step's changes which of them run.
    for tp in (1, 2):
        for width in (DECODE_SEQS, SECOND_WIDTH):
            at = capture(tp, step_symbol=True, width=width)
            assert host_resolutions_by_line(at) == EXPECTED_HOST_RESOLUTIONS, (
                tp,
                width,
            )

    resolved_at = {
        frame for event in record["host_resolutions"] for frame in event["frames"][-1:]
    }
    assert SITE_ONE[1][0] in resolved_at
    # ... and nowhere in the specialisations, which is the whole change.
    assert record["specialisations"] == []

    at_the_copy = {
        entry["op"]: entry["n"]
        for entry in record["symbolic_sites"]
        if entry["call_site"] == SITE_TWO[1][-1]
    }
    # One copy per staged buffer, two slices per copy -- the destination and
    # the source, which is the pair that used to decide the shape between them
    # and now agrees on it -- and the device read the copy does per call.
    assert at_the_copy == {
        "aten.copy_.default": STAGED_COPIES,
        "aten.slice.Tensor": 2 * STAGED_COPIES,
        "prim.device.default": STAGED_COPIES,
    }

    # Site three converts nothing now. It is ATOM's own `_rows`, which returns
    # `t.shape[0]` unconverted, and it is the only site of the three that is not
    # reachable from a capture-time substitution.
    assert not [
        event
        for event in record["host_resolutions"]
        if any("in _rows" in frame for frame in event["frames"])
    ]


def test_the_symbol_is_free_across_the_step_width():
    """Two widths, one inventory: the evidence that the symbol is not a hint.

    This is the load-bearing evidence for the whole result, and it is the only
    thing here that can see a symbol *lost* rather than solved -- code reading
    a hint and building a tensor of that size records no replacement and
    installs no guard.

    ATOM computes its host values from the symbol's hint -- the slot mapping,
    the block tables, the cumulative sequence lengths are all real numbers for
    one concrete width. A graph with symbolic shapes built that way would still
    be a graph about one width if any dimension had quietly taken the hint
    instead of the symbol, and nothing in the census would say so.

    So the step is traced twice, at two widths, and the two inventories are
    compared operator for operator and shape for shape with the symbol's *name*
    set aside. They are identical. Anything carrying the hint would be a 2 in
    one and an 8 in the other.

    That is also what the concrete half of the census rests on: every dimension
    that stayed a number is the same number at both widths, so none of them is
    a step width in disguise. They are the model's and the engine's -- hidden
    sizes, head counts, projection widths, and the staging buffers' capacities.

    **One of those constants is a 2, and it is not the step's.** At TP2 the
    value 2 appears 73 times in the concrete half of the census -- it is the
    *group's* width, which this step really does have. The identity above is
    what separates the two readings: the concrete dimensions are the same at
    step width 2 and at step width 8, so a 2 that survives a step eight rows
    wide is an engine constant and not a width that leaked. At TP1 no 2 appears
    at all.

    **At TP2 as well as TP1.** TP2 is the width where a capture has most to
    lose, so the comparison that carries the result is run there too rather
    than inferred from TP1.

    **What the digest covers, and the one artifact that differs.** The digest
    is over operator names and tensor shapes: not scalar arguments, not dtypes,
    not strides. The single control-versus-symbol difference this file reports,
    `lift_fresh` becoming `scalar_tensor`, is a *value*-level difference and
    was found by comparing `distinct_ops`, which is the instrument for that.
    And one artifact genuinely is not identical between the two widths -- the
    guard set, which acquires a lower bound `<axis> + 1 > <hint>` at every hint
    but 2. It is asserted here rather than left out of the comparison, because
    a reader told five rows are identical should be told which row is not and
    why it does not disturb the conclusion: it is a `__bool__` comparison on a
    live symbol, which is the boundary this pass left alive on purpose.
    """
    for tp in (1, 2):
        narrow = capture(tp, step_symbol=True)
        wide = capture(tp, step_symbol=True, width=SECOND_WIDTH)
        assert narrow["step_axis_hint"] == DECODE_SEQS, tp
        assert wide["step_axis_hint"] == SECOND_WIDTH, tp
        assert wide["graph_digest"] == narrow["graph_digest"], tp
        assert wide["ops"] == narrow["ops"], tp
        assert wide["shape_entries"] == narrow["shape_entries"], tp
        assert (
            wide["non_numeric_shape_entries"] == narrow["non_numeric_shape_entries"]
        ), tp
        assert wide["concrete_dims"] == narrow["concrete_dims"], tp
        # The second width appears nowhere as a dimension, at either TP: the
        # same claim read the other way round, and the direction that can
        # actually fail, because a dimension that took the hint would be an 8
        # in the wide capture and a 2 in the narrow one.
        values = {
            int(value)
            for counts in narrow["concrete_dims"].values()
            for value in counts
        }
        assert SECOND_WIDTH not in values, tp
        if tp == 1:
            assert DECODE_SEQS not in values, tp
        else:
            # At TP2 the constant 2 *is* present -- 73 entries, across
            # allocations, views, transfers and one device read -- and it is
            # the **group's** width, not the step's. What says so is the line
            # above it: `concrete_dims` is identical at step widths 2 and 8, so
            # a 2 that is still a 2 when the step is eight wide is not the
            # step. The group width is an engine constant here in the way a
            # head count is.
            assert values & {DECODE_SEQS} == {DECODE_SEQS}, tp
        # The one artifact that is width-dependent, stated rather than omitted.
        assert set(narrow["shape_env_guards"]) == expected_guards(tp, DECODE_SEQS), tp
        assert set(wide["shape_env_guards"]) == expected_guards(tp, SECOND_WIDTH), tp
        assert f"<axis> + 1 > {SECOND_WIDTH}" in wide["shape_env_guards"], tp
        assert f"<axis> + 1 > {SECOND_WIDTH}" not in narrow["shape_env_guards"], tp


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
