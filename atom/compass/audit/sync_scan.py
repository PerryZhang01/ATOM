# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Find every call on ATOM's serving path that can stop a thread.

The simulator substitutes predicted durations for real work, so any call that
parks a thread on the real clock has to be accounted for: some become an
advance of simulated time, some carry a message from one logical process to
another, some are timeouts that move onto the simulated clock, and some are
invisible to simulated time and are left alone.
Deciding that per call site is a reading job; keeping the *list* honest as
ATOM changes is not, and that is what this module does.

It reports candidates; it does not judge them. ``sync_sites.json`` beside this
file carries one classified row per candidate, and a test asserts the two
agree, so a blocking call added to ATOM later fails that test instead of being
missed.

Two detectors:

``scan_calls``
    Parses every file under ``atom/`` outside :data:`UNSCANNED_ROOTS` and
    reports calls whose shape can park a thread -- sleeps, socket receives,
    pollers, queue operations, joins, lock and event waits, collectives,
    device synchronization, sends.
    Shapes are matched on the call's own text and argument form, never on what
    a name happens to be bound to at runtime, so the match is local and stable.

``scan_spin_loops``
    Reports ``while`` loops that go around without calling anything that can
    park -- a spin that burns a core instead of waiting.

Argument-form rules exist where a name alone is ambiguous. ``d.get(key)`` and
``q.get()`` are the same attribute on unrelated objects; so are ``",".join(x)``
and ``thread.join()``. Rather than guess the receiver's type, each shape states
the argument form that only the blocking reading can take -- ``get`` with no
positional argument, ``join`` with none or a single number. The rules are in
:data:`SHAPES` and each carries the reason it is written that way.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Every Python file under `atom/` is scanned unless an entry here names it, a
# directory (ending in `/`) or a single file, with the reason it is not on the
# path a simulated run executes. The scanned set is read off the tree, so a
# package added under `atom/` is scanned the day it lands and its blocking
# calls fail the inventory test until they are classified or it is named here.
UNSCANNED_ROOTS: tuple[tuple[str, str], ...] = (
    (
        "atom/benchmarks/",
        "client scripts that load a server from outside; they are not the server",
    ),
    (
        "atom/diffusion/",
        (
            "the video generation engine, a server of its own; a simulated "
            "run starts the language-model server"
        ),
    ),
    (
        "atom/entrypoints/atomesh/",
        (
            "the standalone router's Python adapter, an alternative front end; "
            "a simulated run posts to the OpenAI server above"
        ),
    ),
    (
        "atom/examples/",
        (
            "scripts that drive the offline engine or a server from outside "
            "it, not the serving path"
        ),
    ),
    (
        "atom/kv_transfer/offload/",
        (
            "CPU and NVMe offload backends, selected by their own transfer "
            "config and not by the prefill/decode path the simulator substitutes"
        ),
    ),
    (
        "atom/mesh/",
        (
            "the Rust router; its one Python file refreshes vendored protocol "
            "sources at build time, and the router's bounds are text anchors"
        ),
    ),
    (
        "atom/model_loader/",
        (
            "checkpoint loading, before the simulated window; a simulated run "
            "builds its model on meta tensors and reads no weights"
        ),
    ),
    (
        "atom/model_ops/",
        (
            "kernels and layers below the forward pass, which the simulator "
            "replaces whole rather than entering"
        ),
    ),
    (
        "atom/models/",
        "the model definitions, the forward pass the simulated runner replaces",
    ),
    (
        "atom/mono/",
        (
            "fused per-layer decode kernels and their tensor-parallel "
            "consensus, inside the forward pass the simulated runner replaces"
        ),
    ),
    (
        "atom/plugin/",
        (
            "the vLLM, SGLang and rtpllm backends, which serve ATOM's models "
            "from another engine's processes; a simulated run starts ATOM's own"
        ),
    ),
    (
        "atom/quantization/",
        "quantized layers inside the forward pass the simulated runner replaces",
    ),
    (
        "atom/utils/clock.py",
        (
            "the simulated clock's client: its waits are how a grant reaches "
            "an LP, the mechanisms the inventory assigns, not sites they apply to"
        ),
    ),
    (
        "atom/utils/compass_loop.py",
        (
            "the frontend event loop's clock driver: its waits are the idle "
            "points and receives the inventory assigns, not sites they apply to"
        ),
    ),
    (
        "atom/compass/clock/",
        (
            "the clock authority: its waits are the grant protocol itself, not "
            "sites a mechanism applies to"
        ),
    ),
    (
        "atom/compass/clock_transport/",
        (
            "the wire between an LP and the clock authority: the grant "
            "protocol's transport, not a site a mechanism applies to"
        ),
    ),
)


@dataclass(frozen=True)
class Shape:
    """One call form that can park a thread."""

    name: str
    # Attribute or bare-function names this shape matches.
    calls: frozenset[str]
    why: str
    # Optional extra predicate over the ast.Call node.
    form: object = None


def _no_positional(node: ast.Call) -> bool:
    """``q.get()`` and ``q.get(timeout=...)`` block; ``d.get(key)`` cannot.

    A mapping ``get`` always carries the key positionally, and calling it with
    none is a TypeError, so "no positional argument" selects the queue reading
    exactly. A leading boolean literal is the ``block`` flag spelled
    positionally and is kept too.
    """
    if any(k.arg in ("timeout", "block") for k in node.keywords):
        return True
    if not node.args:
        return True
    first = node.args[0]
    return isinstance(first, ast.Constant) and isinstance(first.value, bool)


def _join_form(node: ast.Call) -> bool:
    """``t.join()`` and ``t.join(5)`` park; ``sep.join(seq)`` and path joins do not.

    Separating on the argument form rather than the receiver keeps the rule
    local: a string join always passes exactly one iterable, a path join two or
    more strings, and neither is a bare call or a lone number.
    """
    if any(k.arg == "timeout" for k in node.keywords):
        return True
    if not node.args:
        return True
    if len(node.args) == 1:
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
            return True
        # `proc.join(remaining)` — a lone plain name that reads as a duration.
        if isinstance(arg, ast.Name) and arg.id in ("timeout", "remaining", "seconds"):
            return True
    return False


def _put_form(node: ast.Call) -> bool:
    """A put parks only on a bounded queue or an awaited one.

    ``put_nowait`` is excluded by name. A plain ``q.put(x)`` on an unbounded
    queue returns at once, so only an explicit bound is taken.
    """
    return any(k.arg in ("timeout", "block") for k in node.keywords)


def _worker_rpc_form(node: ast.Call) -> bool:
    """A worker RPC parks only when the caller asks for the reply.

    ``call_func`` without ``wait_out`` returns as soon as the request is on the
    shared-memory ring, and that ring's own hand-off is already a site of its
    own. The aggregating form always collects every worker's reply.
    """
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name == "call_func_with_aggregation":
        return True
    return any(
        k.arg == "wait_out"
        and isinstance(k.value, ast.Constant)
        and k.value.value is True
        for k in node.keywords
    )


SHAPES: tuple[Shape, ...] = (
    Shape(
        "sleep",
        frozenset({"sleep"}),
        "parks for a stated duration, whether that duration means anything or not",
    ),
    Shape(
        "socket_recv",
        frozenset(
            {
                "recv",
                "recv_multipart",
                "recv_pyobj",
                "recv_json",
                "recv_string",
                "recv_into",
            }
        ),
        "parks until a peer sends; unbounded unless the socket carries a timeout",
    ),
    Shape(
        "poll",
        frozenset({"poll"}),
        "parks up to a bound waiting for a socket to become readable",
    ),
    Shape(
        "queue_get",
        frozenset({"get"}),
        "parks until another thread or process puts an item",
        _no_positional,
    ),
    Shape(
        "queue_put",
        frozenset({"put"}),
        "parks when the queue is full",
        _put_form,
    ),
    Shape(
        "join",
        frozenset({"join"}),
        "parks until a thread or process exits",
        _join_form,
    ),
    Shape(
        "lock_acquire",
        frozenset({"acquire"}),
        "parks until the holder releases",
    ),
    Shape(
        "event_wait",
        frozenset({"wait", "wait_for", "result"}),
        "parks until another thread signals, or a future resolves",
    ),
    Shape(
        "barrier",
        frozenset({"barrier"}),
        "parks until every rank arrives",
    ),
    Shape(
        "collective",
        frozenset(
            {
                "all_reduce",
                "all_gather",
                "all_gather_into_tensor",
                "reduce_scatter",
                "reduce_scatter_tensor",
                "all_to_all",
                "broadcast",
                "gather",
                "scatter",
                "reduce",
            }
        ),
        "parks until every rank in the group arrives",
    ),
    Shape(
        "p2p",
        frozenset({"send", "send_multipart", "isend", "irecv", "batch_isend_irecv"}),
        "hands a message to a peer; parks when the transport's buffer is full",
    ),
    Shape(
        "device_sync",
        frozenset({"synchronize"}),
        "parks until the device stream or event completes",
    ),
    Shape(
        "task_wait",
        frozenset({"dequeue", "enqueue"}),
        "parks on the shared-memory ring until every reader or the writer moves",
    ),
    Shape(
        "executor_handoff",
        frozenset({"run_in_executor", "submit"}),
        "runs a callable on a thread pool; an awaited hand-off parks the caller "
        "until it returns, and a full pool parks it longer",
    ),
    Shape(
        "worker_rpc",
        frozenset({"call_func", "call_func_with_aggregation"}),
        "sends work to the worker processes and parks until their reply arrives",
        _worker_rpc_form,
    ),
)

_SHAPE_BY_CALL: dict[str, list[Shape]] = {}
for _shape in SHAPES:
    for _call in _shape.calls:
        _SHAPE_BY_CALL.setdefault(_call, []).append(_shape)

# Names that read like a blocking call but never park.
NON_BLOCKING_NAMES = frozenset(
    {"get_nowait", "put_nowait", "recv_nowait", "acquire_nowait"}
)


# How much of a call expression the identity carries. Long enough that two
# calls to one method in one function differ by their arguments; short enough
# that a wrapped multi-line call does not make the key unreadable.
EXPR_CHARS = 160


@dataclass
class Site:
    """One candidate call site."""

    file: str
    line: int
    symbol: str
    call: str
    expr: str
    shape: str
    ordinal: int = 0

    @property
    def id(self) -> str:
        """What names this site across edits.

        Deliberately not the line: an edit anywhere above a call would
        otherwise re-open every classification below it. Deliberately *with*
        the arguments: the callee alone cannot tell two calls to one method in
        one function apart, and the ordinal that separated them was positional,
        so inserting a call above another silently moved every row below it
        onto the wrong site. Two sites that still collide are textually the
        same call in the same function, which the inventory is required to
        classify the same way.
        """
        return f"{self.file}::{self.symbol}::{self.expr}::{self.shape}#{self.ordinal}"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "symbol": self.symbol,
            "call": self.call,
            "expr": self.expr,
            "shape": self.shape,
        }


class _ScopedVisitor(ast.NodeVisitor):
    """Tracks the enclosing qualified name a site or an anchor carries.

    ``spans`` maps each such name to the line spans of the definitions it
    names; ``<module>`` spans the whole file.
    """

    def __init__(self, rel_path: str, source: str) -> None:
        self.rel_path = rel_path
        self.source = source
        self._scope: list[str] = []
        self.sites: list[Site] = []
        self.spans = {"<module>": [(1, len(source.splitlines()))]}

    @property
    def symbol(self) -> str:
        return ".".join(self._scope) or "<module>"

    def _push(self, node):
        self._scope.append(node.name)
        self.spans.setdefault(self.symbol, []).append((node.lineno, node.end_lineno))
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _push
    visit_AsyncFunctionDef = _push
    visit_ClassDef = _push


class _CallVisitor(_ScopedVisitor):
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            name = None
        if name is not None and name not in NON_BLOCKING_NAMES:
            for shape in _SHAPE_BY_CALL.get(name, ()):
                if shape.form is not None and not shape.form(node):
                    continue
                text = ast.get_source_segment(self.source, func) or name
                text = " ".join(text.split())
                whole = ast.get_source_segment(self.source, node) or text
                whole = " ".join(whole.split())[:EXPR_CHARS]
                self.sites.append(
                    Site(
                        file=self.rel_path,
                        line=node.lineno,
                        symbol=self.symbol,
                        call=text,
                        expr=whole,
                        shape=shape.name,
                    )
                )
                break
        self.generic_visit(node)


class _SpinVisitor(_ScopedVisitor):
    """Find ``while`` loops that go around without calling anything that parks."""

    def visit_While(self, node: ast.While) -> None:
        if self._is_spin(node):
            text = ast.get_source_segment(self.source, node.test) or "while"
            whole = f"while {' '.join(text.split())}"[:EXPR_CHARS]
            self.sites.append(
                Site(
                    file=self.rel_path,
                    line=node.lineno,
                    symbol=self.symbol,
                    call=whole,
                    expr=whole,
                    shape="spin_loop",
                )
            )
        self.generic_visit(node)

    @staticmethod
    def _is_spin(node: ast.While) -> bool:
        has_continue = False
        for child in ast.walk(node):
            if isinstance(child, ast.Continue):
                has_continue = True
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(child, ast.Call):
                func = child.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None
                )
                if name in _SHAPE_BY_CALL and name not in NON_BLOCKING_NAMES:
                    return False
        return has_continue


def _number(sites: list[Site]) -> list[Site]:
    """Give sites that share a symbol and call text a stable ordinal.

    Line numbers move with any edit above them, so identity is the enclosing
    symbol plus the whole call expression plus its position among *textually
    identical* siblings. Siblings that share an ordinal are the same call
    written the same way in one function, so the inventory has to give them the
    same answer, and a test asserts it does. The line only orders the sites.
    """
    seen: dict[tuple[str, str, str], int] = {}
    for site in sites:
        key = (site.file, site.symbol, f"{site.expr}::{site.shape}")
        site.ordinal = seen.get(key, 0)
        seen[key] = site.ordinal + 1
    return sites


def _excluded(rel: str) -> bool:
    return any(
        rel.startswith(root) if root.endswith("/") else rel == root
        for root, _why in UNSCANNED_ROOTS
    )


def iter_scanned_files(tree_root: str | os.PathLike) -> list[str]:
    """Every file under ``atom/`` the scanner reads, relative to ``tree_root``.

    Read off the tree, minus :data:`UNSCANNED_ROOTS`, and sorted.
    """
    root = Path(tree_root)
    found = (str(p.relative_to(root)) for p in (root / "atom").rglob("*.py"))
    return sorted(f for f in found if not _excluded(f))


def scan(tree_root: str | os.PathLike) -> list[Site]:
    """Every candidate site :func:`iter_scanned_files` reads, in file and line order."""
    root = Path(tree_root)
    sites: list[Site] = []
    for rel in iter_scanned_files(root):
        source = (root / rel).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel)
        calls = _CallVisitor(rel, source)
        calls.visit(tree)
        spins = _SpinVisitor(rel, source)
        spins.visit(tree)
        sites.extend(calls.sites)
        sites.extend(spins.sites)
    sites.sort(key=lambda s: (s.file, s.line, s.call, s.shape))
    return _number(sites)


INVENTORY_PATH = Path(__file__).with_name("sync_sites.json")


def load_inventory(path: str | os.PathLike | None = None) -> dict:
    """The checked-in classification, as written."""
    return json.loads(Path(path or INVENTORY_PATH).read_text(encoding="utf-8"))


def repo_root_from_here() -> Path:
    """The checkout this module was imported from."""
    return Path(__file__).resolve().parents[3]
