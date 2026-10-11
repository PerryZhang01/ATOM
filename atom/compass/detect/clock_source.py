# SPDX-License-Identifier: MIT
"""A static pass for reads of a real clock on the simulated path.

Time on the simulated path comes from the run's own clock. A call to the
machine's clock returns real seconds, and real seconds put into a record of
modelled ones are not visibly different from the modelled ones -- the number
has the right magnitude, the table still adds up, and the only sign of trouble
is that two timestamps on one timeline disagree by a margin nobody is looking
for. So this is a static check: the read is a failure on the day it is written,
not on the day a number taken off it is argued about.

Reads are found by parsing rather than by matching text, because the forms that
matter are not all spelled the same. `from time import monotonic` followed by a
bare `monotonic()` is the same read as `time.monotonic()`, and a text search for
the second does not find the first. Imports and the aliases assigned from them
are resolved to the name they came from, and the call target is compared against
that -- so `now = time.monotonic` followed by `now()` is found as well, a form a
text search for the assignment *would* have found and an earlier parse of only
the imports did not.

**What this pass does not see.** Naming the blind spots is part of the check,
because a check whose limits are unstated gets read as a guarantee:

* a read reached through a string -- `getattr(time, "perf_counter")()`, or
  `importlib.import_module("time")` -- because the name is not in the syntax;
* `from time import *`, because the names it binds are in the module it imports
  and not in this one's text;
* a name bound to the result of a call rather than to a name, since only a
  dotted name on the right of an assignment is followed;
* scope: a name bound anywhere in the file is treated as bound everywhere in
  it, so a local alias in one function is resolved in another;
* **a clock handed around as a value rather than assigned to a name.** Only an
  import and a plain assignment are followed, so a parameter default
  (`def step(wall_clock=time.monotonic)`), a walrus, a tuple unpacking, a class
  attribute read back through the class, a dict entry read back by subscript
  and a `functools.partial` all carry the clock past this pass. They are one
  class -- the callable is bound somewhere the parser does not follow and
  called later -- and the parameter default is the form that appears in real
  code, which is also why a bare attribute load is not treated as a read.

Each of those is a read this pass would miss and a reviewer would not, which is
the trade the pass is worth making and not a reason to trust it alone. The list
is what has been tried against it, not an enumeration of what Python allows.

**The allow-list is part of the check, not an escape from it.** A read that is
meant to stay real gets an entry naming its file, or its site and class, and the
reason, so the list is reviewable, and adding to it is a decision somebody
writes down rather than a silence.
"""

import ast
import os
from collections import Counter
from dataclasses import dataclass

#: The calls that return real seconds. An asyncio timer is not one: it runs on
#: the event loop's clock, which a simulated run replaces. The `_ns` forms are
#: the same machine clocks in different units, and a number divided by a
#: billion after the fact is not distinguishable from one taken in seconds, so
#: they are listed beside them rather than under them.
#:
#: **Every entry is the fully qualified name, because the comparison is exact.**
#: A tail match would hand the whole list to any receiver whose attribute
#: happens to be spelled the same -- `self.time.monotonic()`, `row.datetime.now()`,
#: a `time` imported from somewhere that is not the standard library. The two
#: `datetime` entries are written out to `datetime.datetime` for the same
#: reason; `from datetime import datetime` resolves to exactly that, so the
#: short spelling costs nothing and the long one cannot be reached by accident.
CLOCK_READS = (
    "time.time",
    "time.time_ns",
    "time.monotonic",
    "time.monotonic_ns",
    "time.perf_counter",
    "time.perf_counter_ns",
    "time.clock_gettime",
    "time.clock_gettime_ns",
    "datetime.datetime.now",
    "datetime.datetime.utcnow",
)

#: Files whose real-clock reads are deliberate, and why. A path matches on whole
#: directory names ending at the entry, so the same entry works from any root
#: the check is run from and a directory that merely ends in the same letters
#: does not collect the exemption.
DEFAULT_ALLOW_LIST: dict[str, str] = {
    "atom/compass/run/__init__.py": (
        "the run summary's wall seconds, from the run's start to its finish: "
        "the speed result, which no simulated time reads"
    ),
    "atom/compass/parity/runner.py": (
        "a real runner's step timestamps; a simulated run runs its own runner "
        "and never imports this module"
    ),
}

#: Packages under `atom/` a simulated run never executes, or that keep their
#: own gate: `compass` its own, `mesh` is Rust, `benchmarks` and `examples` are
#: scripts, and `diffusion` and `plugin` are never run by a simulation.
NOT_CORE = ("compass", "mesh", "benchmarks", "examples", "diffusion", "plugin")

_ATOM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: ATOM's core: every module and package under `atom/` outside `NOT_CORE`, read
#: off the tree so a new package is gated the day it lands. Paths are relative
#: to the repository root; the clock-source and set-iteration gates both scan
#: these.
CORE_ROOTS = tuple(
    f"atom/{name}"
    for name in sorted(os.listdir(_ATOM))
    if name not in NOT_CORE + ("__pycache__",)
    and (name.endswith(".py") or os.path.isdir(os.path.join(_ATOM, name)))
)

_REPLACED_RUNNER = "the real model runner, which a simulated run replaces"
_REPLACED_TRANSFER = "a transfer backend the simulator replaces with a priced one"
_LOG_ONLY = "times real work for a log line; nothing in the simulated record reads it"
_COMPILE = "torch.compile at startup, before the simulated window"
_LOAD = "loads or warms up the model at startup, before the simulated window"

#: Real-clock reads in ATOM's core that stay real, by site: (file, the innermost
#: def holding the read, the call) -> (class, reason, how many such reads the
#: def holds). A def holding a different number fails, so a new read beside a
#: kept one is not excused by it. K8 stays real on purpose, out of the clock
#: authority's reach; K9 is outside the model -- outside the simulated window,
#: in code a simulated run replaces, or a timing only a log line reads. Every
#: other read takes the LP clock through `atom.utils.clock.now`.
# fmt: off
_KEPT_REAL = (
    ("atom/model_engine/engine_core.py", "_drain_kv_work_at_exit", "time.monotonic",
     "K9", "bounds the KV drain at shutdown, after the last step is charged", 2),
    ("atom/model_engine/engine_core.py", "_process_engine_step", "time.perf_counter",
     "K9", _LOG_ONLY, 4),
    ("atom/model_engine/engine_core_mgr.py", "close", "time.monotonic",
     "K9", "the engine processes' shutdown grace period, after the simulated window", 2),
    ("atom/model_engine/engine_utility.py", "_execute_utility_command", "time.monotonic",
     "K9", _LOG_ONLY, 2),
    ("atom/model_engine/model_runner.py", "_build_and_load_model", "time.perf_counter",
     "K9", _REPLACED_RUNNER, 2),
    ("atom/model_engine/model_runner.py", "_on_trace_ready", "time.time",
     "K9", _REPLACED_RUNNER, 1),
    ("atom/model_engine/model_runner.py", "_on_trace_ready", "time.monotonic",
     "K9", _REPLACED_RUNNER, 2),
    ("atom/model_engine/model_runner.py", "stop_profiler", "time.monotonic",
     "K9", _REPLACED_RUNNER, 2),
    ("atom/model_engine/model_runner.py", "warmup_model", "time.time",
     "K9", _REPLACED_RUNNER, 2),
    ("atom/model_engine/model_runner.py", "capture_cudagraph", "time.time",
     "K9", _REPLACED_RUNNER, 2),
    ("atom/model_engine/model_runner.py", "_disagg_collect_rank_files", "time.monotonic",
     "K9", _REPLACED_RUNNER, 2),
    ("atom/kv_transfer/disaggregation/moriio/moriio_connector.py", "_execute_handshake",
     "time.perf_counter", "K9", _REPLACED_TRANSFER, 3),
    ("atom/kv_transfer/offload/dense/connector.py", "_do_load_req", "time.perf_counter",
     "K9", _REPLACED_TRANSFER, 4),
    ("atom/kv_transfer/offload/dense/connector.py", "_do_save_req", "time.perf_counter",
     "K9", _REPLACED_TRANSFER, 4),
    ("atom/kv_transfer/offload/hybrid/dsv4/connector.py", "_load_page", "time.perf_counter",
     "K9", _REPLACED_TRANSFER, 4),
    ("atom/kv_transfer/offload/hybrid/dsv4/connector.py", "_do_save_req", "time.perf_counter",
     "K9", _REPLACED_TRANSFER, 4),
    ("atom/model_loader/expert_staging.py", "_ensure_staging", "time.perf_counter",
     "K9", _LOAD, 2),
    ("atom/model_loader/loader.py", "load_model", "time.perf_counter", "K9", _LOAD, 2),
    ("atom/model_loader/loader.py", "_save_online_quant_info", "time.time_ns",
     "K9", "names a file the loader writes at startup, before the simulated window", 1),
    ("atom/model_loader/loading_core.py", "load_weights_into_model", "time.perf_counter",
     "K9", _LOAD, 8),
    ("atom/model_loader/weight_iterator.py", "_run", "time.perf_counter", "K9", _LOAD, 2),
    ("atom/model_loader/weight_utils.py", "download_weights_from_hf", "time.perf_counter",
     "K9", _LOAD, 2),
    ("atom/spec_decode/drafter.py", "warmup_draft_graphs", "time.time", "K9", _LOAD, 2),
    ("atom/model_ops/eplb.py", "_execute_runtime_rebalance", "time.perf_counter",
     "K9", "moves expert weights inside the real forward, which a simulated run replaces", 4),
    ("atom/rollout/weight_sync.py", "load_weights_via_ipc", "time.time",
     "K9", "pushes real weights to the GPU, which a simulated run never holds", 2),
    ("atom/utils/__init__.py", "shutdown_all_processes", "time.monotonic",
     "K9", "the process shutdown grace period, after the simulated window", 2),
    ("atom/utils/backends.py", "compile", "time.time", "K9", _COMPILE, 3),
    ("atom/utils/backends.py", "__call__", "time.time", "K9", _COMPILE, 1),
    ("atom/utils/compass_loop.py", "select", "time.monotonic", "K9",
     "times a stall's diagnostic in wall seconds; the LP clock stands still", 1),
    ("atom/utils/decorators.py", "start_monitoring_torch_compile", "time.time",
     "K9", _COMPILE, 1),
    ("atom/utils/gc_utils.py", "_log", "time.perf_counter", "K9", _LOG_ONLY, 2),
)
# fmt: on
CORE_ALLOW_LIST: dict[tuple[str, str, str], tuple[str, str, int]] = {
    (path, scope, call): (kind, why, count)
    for path, scope, call, kind, why, count in _KEPT_REAL
}


def _names(path: str, entry: str) -> bool:
    """Does `entry` name `path`? Whole path segments only, from any root."""
    posix = path.replace(os.sep, "/")
    return posix == entry or posix.endswith("/" + entry)


@dataclass(frozen=True)
class ClockRead:
    """One call to a real clock: where it is, what it calls, and inside what."""

    path: str
    line: int
    call: str
    scope: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}  {self.call}  in {self.scope}"


class ClockSourceLint:
    """Parses the simulated path and reports every real-clock read it finds."""

    def __init__(self, allow_list=DEFAULT_ALLOW_LIST, sites=CORE_ALLOW_LIST) -> None:
        self.allow_list = dict(allow_list)
        self.sites = dict(sites)

    def allowed(self, path: str) -> str | None:
        """The recorded reason this file may read a real clock, if it may.

        The match is on whole path segments. A suffix on the text alone would
        hand the exemption to any directory whose name happens to end in the
        first segment of an entry, which is a file nobody reviewed.
        """
        for entry in sorted(self.allow_list):
            if _names(path, entry):
                return self.allow_list[entry]
        return None

    def site(self, read: ClockRead) -> tuple[str, str, str] | None:
        """The entry naming this read's site, if one does."""
        for path, scope, call in self.sites:
            if (scope, call) == (read.scope, read.call) and _names(read.path, path):
                return path, scope, call
        return None

    def scan_source(self, source: str, path: str) -> tuple[ClockRead, ...]:
        """Every real-clock read in one module's text, in line order.

        A call counts when its resolved name is exactly an entry in the list.
        Matching a tail instead would accuse any receiver whose attribute is
        spelled like a module.
        """
        if self.allowed(path) is not None:
            return ()
        tree = ast.parse(source, filename=path)
        origins = _origins(tree)
        spans = scopes(tree)
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call = _resolve(_dotted(node.func), origins)
            if call in CLOCK_READS:
                found.append(
                    ClockRead(path, node.lineno, call, scope_at(spans, node.lineno))
                )
        return tuple(sorted(found, key=lambda read: (read.path, read.line, read.call)))

    @staticmethod
    def modules(root: str) -> tuple[str, ...]:
        """Every module under `root`, in a fixed order on every machine."""
        if os.path.isfile(root):
            return (root,)
        found = []
        for directory, subdirectories, names in os.walk(root):
            subdirectories[:] = sorted(
                name for name in subdirectories if name != "__pycache__"
            )
            found.extend(
                os.path.join(directory, name)
                for name in sorted(names)
                if name.endswith(".py")
            )
        return tuple(found)

    def report(self, reads, scanned: int, allow_listed: int = 0, kept: int = 0) -> str:
        """What the check prints. Clean is a line; dirty is the list and the fix.

        `allow_listed` is how many of the files in this scan were passed over,
        and `kept` how many reads a site entry kept real, not how long either
        list is. The line reads as a statement about the scan, so it has to be
        one: a tree containing none of the listed files is a tree where nothing
        was exempted.
        """
        if not reads:
            by_site = f", {kept} read(s) kept real by site" if kept else ""
            return (
                f"clock-source lint: clean over {scanned} module(s), "
                f"{allow_listed} file(s) allow-listed{by_site}"
            )
        lines = [
            f"clock-source lint: {len(reads)} real-clock read(s) on the simulated path:"
        ]
        lines.extend(f"  {read}" for read in reads)
        lines.append(
            "A run charges simulated seconds for the work it models. A real read "
            "puts machine seconds into the same record, and the mixture is reported "
            "as one modelled number. Take the time from the run's clock, or add the "
            "file to the allow-list with the reason it stays real."
        )
        return "\n".join(lines)

    def check(self, root: str) -> tuple[int, str]:
        """Scan `root` and return an exit code beside the report. Non-zero fails CI.

        A root holding no module is refused: a scan of nothing has checked
        nothing, and a mistyped path would otherwise read as clean.
        """
        modules = self.modules(root)
        if not modules:
            return 1, f"clock-source lint: no module under {root}, nothing checked"
        reads = []
        for path in modules:
            with open(path, encoding="utf-8") as handle:
                reads.extend(self.scan_source(handle.read(), path))
        allow_listed = sum(1 for path in modules if self.allowed(path) is not None)
        unlisted = unkept(reads, self.sites, self.site)
        kept = len(reads) - len(unlisted)
        return (1 if unlisted else 0), self.report(
            unlisted, len(modules), allow_listed, kept
        )


def unkept(reads, sites, site) -> list:
    """The reads no entry keeps.

    `site(read)` names a read's entry, and the entry's last field is how many
    reads it keeps. A site holding any other number keeps none, so a new read
    beside a kept one is reported together with them.
    """
    keys = [site(read) for read in reads]
    held = Counter(keys)
    return [
        read
        for read, key in zip(reads, keys)
        if key is None or held[key] != sites[key][-1]
    ]


def _dotted(node) -> str | None:
    """`a.b.c` for an attribute or name chain, or None for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _origins(tree) -> dict[str, str]:
    """Local name -> the dotted name it stands for: imports, and aliases of them.

    An alias is an assignment whose right-hand side is a dotted name, and it is
    recorded whole -- `self._clock = time.perf_counter` binds `self._clock`, not
    `self` -- so that the call it enables resolves the same way an import does.
    The first binding a name gets is the one kept: a name assigned twice says
    two things, and the check is not an interpreter.
    """
    origins: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                origins.setdefault(alias.asname or alias.name.split(".")[0], alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                origins.setdefault(
                    alias.asname or alias.name, f"{node.module}.{alias.name}"
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            source = _dotted(node.value)
            if source is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                name = _dotted(target)
                if name is not None and name != source:
                    origins.setdefault(name, source)
    return origins


def _resolve(call: str | None, origins: dict[str, str]) -> str | None:
    """Rewrite a call's head through the imports and aliases that introduced it.

    An alias can stand on an alias, and `ast.walk` does not visit the bindings
    in the order they were written, so the chain is followed rather than
    stepped once. Each name is expanded at most once along a chain: a name can
    legitimately stand for one that contains it -- `from datetime import
    datetime` binds `datetime` to `datetime.datetime` -- and expanding that a
    second time would grow the name instead of resolving it.
    """
    if call is None:
        return None
    expanded = set()
    while True:
        origin = origins.get(call)
        if origin is not None:
            if call in expanded:
                return call
            expanded.add(call)
        else:
            head, _, rest = call.partition(".")
            head_origin = origins.get(head)
            if head_origin is None or head in expanded:
                return call
            expanded.add(head)
            origin = f"{head_origin}.{rest}" if rest else head_origin
        if origin == call:
            return call
        call = origin


def scopes(tree) -> tuple[tuple[int, int, str], ...]:
    """Every def and class as (first line, last line, name), outermost first."""
    return tuple(
        (node.lineno, node.end_lineno or node.lineno, node.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def scope_at(spans, line: int) -> str:
    """The innermost def or class containing `line`, or module level.

    `spans` is what `scopes` returns for the module holding `line`.
    """
    inner = [span for span in spans if span[0] <= line <= span[1]]
    return max(inner)[2] if inner else "<module>"
