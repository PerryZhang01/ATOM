# SPDX-License-Identifier: MIT
"""A static pass over the Rust router for clock reads and unseeded choices.

Under the router LP the router's choice of worker must depend on the run's
seed and its simulated history, never on the host. A wall-clock read, a draw
from an unseeded RNG, a map keyed by a random hasher and a `select!` that
polls its branches in random order all depend on the host. Each hit under
`atom/mesh/src/` is either routed through the router's determinism module
(which this pass then does not see) or kept by an entry below naming its file,
its pattern, how many hits the file holds, whether it can change a routing
decision, and why. A file holding a different number fails, so a new hit
beside a kept one is not excused by it. Entries of class `decision` are known
defects, not exemptions.

It matches tokens, not a parse, so cargo is not needed. Comments, string and
char literals are blanked first, and code under `#[cfg(test)]` is skipped,
including a module file a `#[cfg(test)] mod name;` declares.

**What this pass does not see:**

* a `use ... as` alias: `use std::time::Instant as Clock; Clock::now()`;
* a name imported once and called bare: `use tokio::time::sleep` is one hit,
  every later `sleep(...)` none;
* a macro that expands to a read, or a crate re-export reached under another
  path (`crate::util::now()` wrapping `Instant::now()` is found at the wrapper,
  not at its callers);
* a default-hasher map built through `Default::default()`, `#[derive(Default)]`
  or a `collect()` whose target type is written on the binding rather than as
  a turbofish;
* any other random source: `uuid::Uuid::new_v4`, `getrandom`, `OsRng`;
* a timer inside a dependency, set through its builder: reqwest
  `.timeout(...)` bounds every worker request by wall seconds and no pattern
  here matches it;
* `#[cfg(...)]` other than exactly `#[cfg(test)]`.

The list is what has been tried against it, not an enumeration of what Rust
allows.
"""

import os
import re
from collections import Counter

#: Pattern name -> regex over code with comments and literals blanked.
PATTERNS = {
    "Instant::now": re.compile(r"\bInstant::now\b"),
    "SystemTime::now": re.compile(r"\bSystemTime::now\b"),
    "Utc::now": re.compile(r"\b(?:Utc|Local)::now\b"),
    "tokio::time": re.compile(r"\btokio::time\b"),
    "rand::": re.compile(r"\brand::"),
    "RandomState": re.compile(
        r"\bRandomState\b"
        r"|\b(?:HashMap|HashSet|DashMap|DashSet)::(?:new|default|from|from_iter"
        r"|with_capacity|with_shard_amount|with_capacity_and_shard_amount)\b"
        r"|collect::<\s*(?:HashMap|HashSet|DashMap|DashSet)\b"
    ),
    "unbiased select!": re.compile(r"\bselect!\s*[({\[](?!\s*biased\s*;)"),
    "thread::sleep": re.compile(r"\bthread::sleep\b"),
}

#: The determinism module: the host reads it switches off under the router LP.
ROUTER = "determinism.rs"

#: Classes. `decision`: can change which worker serves a request under the
#: router LP; a defect. `bound`: a wall-clock bound or failure detector that
#: stays real on purpose and that the simulated launch (`pd_sim.sh`) leaves
#: off. `off-lp`: code whose requests never pass the router LP's ingress.
#: `latency`: changes only when something happens, or what a log, metric, id
#: or response field says, never which worker serves a request.
DECISION, BOUND, OFF_LP, LATENCY = "decision", "bound", "off-lp", "latency"

_TIMING = "times a request for a log line or metric"
_STAMP = "a creation or storage timestamp in a response or record"
_LOOKUP = "a map read by key; nothing reads its order"
_STARTUP = "worker discovery at startup, before the first request"
_ID = "draws an id no policy reads"
_EXT_PROC = "the Envoy ext_proc ingress, which a simulated launch does not start"
_TOKENS = "the rate limiter, built only with --max-concurrent-requests"
_PARSER = "tool-call parsing: what a response says"

# fmt: off
_KEPT = (
    ("compass_clock.rs", "RandomState", LATENCY, "next seq per channel, " + _LOOKUP, 1),
    ("compass_router.rs", "tokio::time", LATENCY,
     "re-arms the wait for a released message and warns; the message ends it", 1),
    ("middleware.rs", "Instant::now", LATENCY, _TIMING, 1),
    ("server.rs", "Instant::now", LATENCY, _STARTUP, 1),
    ("server.rs", "tokio::time", LATENCY, _STARTUP, 1),
    ("server.rs", "unbiased select!", LATENCY, "waits for a shutdown signal", 2),
    ("core/admission.rs", "tokio::time", BOUND, (
        "the admission queue timeout; with no --max-concurrent-requests a request "
        "queues only for a lock"), 1),
    ("core/circuit_breaker.rs", "Instant::now", BOUND, "--disable-circuit-breaker", 1),
    ("core/job_queue.rs", "Instant::now", LATENCY, _STARTUP, 1),
    ("core/job_queue.rs", "RandomState", LATENCY, _LOOKUP, 2),
    ("core/job_queue.rs", "SystemTime::now", LATENCY, "expires old job statuses", 1),
    ("core/job_queue.rs", "tokio::time", LATENCY, "expires old job statuses", 1),
    ("core/placement/backend/sglang.rs", "rand::", LATENCY, _ID, 1),
    ("core/retry.rs", "rand::", LATENCY, "backoff jitter; pd_sim.sh passes --disable-retries", 2),
    ("core/retry.rs", "tokio::time", LATENCY, "backoff wait; pd_sim.sh passes --disable-retries", 1),
    ("core/steps/worker/local/detect_connection.rs", "tokio::time", LATENCY, _STARTUP, 2),
    ("core/steps/worker/local/discover_metadata.rs", "RandomState", LATENCY, _LOOKUP, 3),
    ("core/steps/worker/local/mod.rs", "RandomState", LATENCY, _LOOKUP, 3),
    ("core/token_bucket.rs", "Instant::now", BOUND, _TOKENS, 3),
    ("core/token_bucket.rs", "tokio::time", BOUND, _TOKENS, 4),
    ("core/token_bucket.rs", "unbiased select!", BOUND, _TOKENS, 1),
    ("core/worker.rs", "RandomState", LATENCY, "active routing keys; only the count is read", 1),
    ("core/worker_builder.rs", "RandomState", LATENCY, _LOOKUP, 4),
    ("core/worker_manager.rs", "RandomState", LATENCY, _LOOKUP, 2),
    ("core/worker_manager.rs", "tokio::time", DECISION,
     "polls worker loads on a wall interval into power_of_two's cached loads", 1),
    ("core/worker_registry.rs", "RandomState", DECISION, (
        "get_all lists workers in the order of a random hasher over random worker "
        "ids; a request naming no model gets its candidates in that order, and "
        "policy tie-breaks read it"), 7),
    ("core/worker_registry.rs", "tokio::time", BOUND, "the health check; --disable-health-check", 2),
    *((f"ext_proc/{name}", pattern, OFF_LP, _EXT_PROC, n) for name, pattern, n in (
        ("executor.rs", "Instant::now", 2), ("executor.rs", "RandomState", 1),
        ("executor.rs", "tokio::time", 2), ("executor.rs", "unbiased select!", 3),
        ("lifecycle.rs", "Instant::now", 1), ("routing.rs", "RandomState", 1),
        ("runtime.rs", "tokio::time", 4), ("runtime.rs", "unbiased select!", 3),
        ("session.rs", "Instant::now", 3), ("session.rs", "unbiased select!", 2))),
    ("observability/gauge_histogram.rs", "RandomState", LATENCY, _LOOKUP, 1),
    ("observability/inflight_tracker.rs", "Instant::now", LATENCY, _TIMING, 2),
    ("observability/inflight_tracker.rs", "RandomState", LATENCY, _LOOKUP, 1),
    ("observability/metrics/schema.rs", "RandomState", LATENCY, _LOOKUP, 1),
    ("observability/request.rs", "Instant::now", LATENCY, _TIMING, 1),
    ("observability/request_id.rs", "rand::", LATENCY, _ID, 2),
    ("observability/ttft.rs", "Instant::now", LATENCY, _TIMING, 1),
    ("policies/cache_aware.rs", "rand::", LATENCY,
     "imports the traits the routed draws call", 2),
    ("policies/cache_aware.rs", "RandomState", LATENCY,
     "a warn-once set, and a grouping by model where each model builds its own tree", 2),
    ("policies/dp_sticky.rs", "RandomState", LATENCY,
     "assignments read by key, and per-worker session counts", 2),
    ("policies/power_of_two.rs", "rand::", DECISION, "draws two candidates from the thread RNG", 2),
    ("policies/power_of_two.rs", "RandomState", LATENCY, _LOOKUP, 1),
    ("policies/random.rs", "rand::", DECISION, "draws the worker from the thread RNG", 2),
    ("policies/registry.rs", "RandomState", LATENCY, _LOOKUP, 2),
    ("policies/tree.rs", "RandomState", LATENCY,
     "per-tenant sizes read by key, and a debug dump", 2),
    ("policies/utils.rs", "thread::sleep", DECISION, (
        "cache_aware's tree eviction sleeps wall seconds between passes, and when "
        "it runs moves later prefix matches"), 1),
    ("routers/comm/conversations.rs", "Utc::now", LATENCY, _STAMP, 1),
    ("routers/comm/persistence_utils.rs", "Utc::now", LATENCY, _STAMP, 1),
    ("routers/grpc/completion_adapter.rs", "SystemTime::now", LATENCY, _STAMP, 2),
    ("routers/grpc/engine/mod.rs", "rand::", LATENCY, _ID, 1),
    ("routers/grpc/engine/worker_client_cache.rs", "RandomState", LATENCY, _LOOKUP, 1),
    ("routers/grpc/pipeline.rs", "Instant::now", LATENCY, _TIMING, 2),
    ("routers/http_pd_router.rs", "Instant::now", LATENCY, _TIMING, 1),
    ("routers/http_pd_router.rs", "RandomState", LATENCY, "startup bootstrap tables; " + _LOOKUP, 5),
    ("routers/http_pd_router.rs", "tokio::time", BOUND, (
        "aborts a prefill after 5 wall seconds and drops its load guard, on the "
        "vLLM backend only; pd_sim.sh passes --backend atom"), 1),
    ("routers/openai/responses/streaming.rs", "Utc::now", LATENCY, _STAMP, 1),
    ("routers/prepare/chat_template.rs", "RandomState", LATENCY, _LOOKUP, 1),
    ("routers/prepare/mod.rs", "SystemTime::now", LATENCY, _STAMP, 2),
    ("routers/render/chat_streaming.rs", "Instant::now", LATENCY, _TIMING, 2),
    ("routers/render/generate_aggregator.rs", "Instant::now", LATENCY, _TIMING, 1),
    ("routers/render/generate_streaming.rs", "Instant::now", LATENCY, _TIMING, 2),
    ("routers/router_manager.rs", "RandomState", LATENCY, _LOOKUP, 1),
    ("tool_parser/parsers/common.rs", "RandomState", LATENCY, _PARSER, 1),
    ("tool_parser/parsers/minimax.rs", "RandomState", LATENCY, _PARSER, 2),
    ("tool_parser/registry.rs", "RandomState", LATENCY, _PARSER, 3),
)
# fmt: on

#: (file under the scanned root, pattern) -> (class, reason, hits in the file).
ALLOW_LIST = {(path, pattern): rest for path, pattern, *rest in _KEPT}

_TEST_ATTR = re.compile(r"#\[cfg\(test\)\]")
_MOD_DECL = re.compile(r"\bmod\s+(\w+)\s*;")


_START = re.compile(r"""//|/\*|(?<!\w)b?r(#*)"|(?<!\w)b?"|'""")


def blank(source: str) -> str:
    """`source` with comments and string and char literals replaced by spaces.

    Newlines are kept, so a match's line is the line it had.
    """
    out, i, n = list(source), 0, len(source)
    while start := _START.search(source, i):
        i, token = start.start(), start.group()
        if token == "//":
            end = source.find("\n", i)
            end = n if end < 0 else end
        elif token == "/*":
            depth, end = 1, i + 2
            while end < n and depth:
                step = source[end : end + 2]
                depth += {"/*": 1, "*/": -1}.get(step, 0)
                end += 2 if step in ("/*", "*/") else 1
        elif "r" in token:
            close = '"' + start.group(1)
            end = source.find(close, start.end())
            end = n if end < 0 else end + len(close)
        elif token != "'":
            end = start.end()
            while end < n and source[end] != '"':
                end += 2 if source[end] == "\\" else 1
            end += 1
        elif source.startswith("\\", i + 1) or source.startswith("'", i + 2):
            end = source.find("'", i + 3 if source[i + 1] == "\\" else i + 2) + 1
        else:  # a lifetime or a label
            i += 1
            continue
        for k in range(i, min(end, n)):
            if out[k] != "\n":
                out[k] = " "
        i = max(end, i + 1)
    return "".join(out)


def _item_end(code: str, start: int) -> int:
    """Where the item starting at `start` ends: after its `{...}` body, after
    a `;` or `,` outside brackets (a `,` also outside `<...>`), or before an
    unmatched `}` (the last field or variant of its parent)."""
    parens = angles = 0
    for end in range(start, len(code)):
        char = code[end]
        if char in "([":
            parens += 1
        elif char in ")]":
            parens -= 1
        elif char == "<":
            angles += 1
        elif char == ">" and code[end - 1] != "-":
            angles = max(angles - 1, 0)
        elif parens:
            continue
        elif char == "}":
            return end
        elif char == ";" or (char == "," and not angles):
            return end + 1
        elif char == "{":
            depth = 0
            for close in range(end, len(code)):
                depth += {"{": 1, "}": -1}.get(code[close], 0)
                if depth == 0:
                    return close + 1
    return len(code)


def strip_tests(code: str) -> tuple[str, list[str]]:
    """`code` with every `#[cfg(test)]` item blanked, and the module names
    that `#[cfg(test)] mod name;` declares."""
    out, test_mods = list(code), []
    for attr in _TEST_ATTR.finditer(code):
        end = _item_end(code, attr.end())
        if code[end - 1] == ";" and (decl := _MOD_DECL.search(code, attr.end(), end)):
            test_mods.append(decl.group(1))
        for k in range(attr.start(), end):
            if out[k] != "\n":
                out[k] = " "
    return "".join(out), test_mods


def _module_files(path: str, name: str) -> tuple[str, str]:
    """The two files `mod name;` in `path` may name."""
    directory, base = os.path.split(path)
    if base not in ("mod.rs", "lib.rs", "main.rs"):
        directory = os.path.join(directory, base[: -len(".rs")])
    return os.path.join(directory, f"{name}.rs"), os.path.join(
        directory, name, "mod.rs"
    )


def hits(source: str) -> tuple[list[tuple[int, str]], list[str]]:
    """(line, pattern) of every hit in one file's text, in line order, and the
    test-only module names it declares."""
    code, test_mods = strip_tests(blank(source))
    found = [
        (code.count("\n", 0, match.start()) + 1, name)
        for name, pattern in PATTERNS.items()
        for match in pattern.finditer(code)
    ]
    return sorted(found), test_mods


def scan(root: str) -> list[tuple[str, int, str]]:
    """(file relative to `root`, line, pattern) of every hit under `root`."""
    texts, skipped = {}, set()
    for directory, subdirectories, names in os.walk(root):
        subdirectories.sort()
        for name in sorted(names):
            if name.endswith(".rs"):
                path = os.path.join(directory, name)
                with open(path, encoding="utf-8") as handle:
                    texts[path] = hits(handle.read())
                for module in texts[path][1]:
                    skipped.update(_module_files(path, module))
    return [
        (os.path.relpath(path, root).replace(os.sep, "/"), line, pattern)
        for path, (found, _) in texts.items()
        if path not in skipped
        for line, pattern in found
    ]


def check(root: str, allow_list=ALLOW_LIST) -> tuple[int, str]:
    """Scan `root`; return an exit code beside the report. Non-zero fails CI.

    The report counts every hit by pattern and disposition (`routed`, an
    entry's class, or `UNKEPT`), then lists the unkept hits and the stale
    entries. An entry naming no hit is stale and fails, so a fixed defect
    does not stay excused. A root holding no .rs file is refused.
    """
    found = scan(root)
    if not any(n.endswith(".rs") for _, _, names in os.walk(root) for n in names):
        return 1, f"rust router lint: no .rs file under {root}, nothing checked"
    held = Counter((path, pattern) for path, _, pattern in found)

    def disposition(path, pattern):
        if path == ROUTER:
            return "routed"
        kind, _, count = allow_list.get((path, pattern), ("UNKEPT", "", 0))
        return kind if count == held[(path, pattern)] else "UNKEPT"

    marked = [(hit, disposition(hit[0], hit[2])) for hit in found]
    unkept = [hit for hit, how in marked if how == "UNKEPT"]
    stale = sorted(key for key in allow_list if not held[key])
    tally = Counter((hit[2], how) for hit, how in marked)
    head = f"{len(found)} hit(s), {len(unkept)} unkept, {len(stale)} stale entries"
    lines = [f"rust router lint: {head}"]
    lines += [f"  {p:<17} {how:<8} {n}" for (p, how), n in sorted(tally.items())]
    lines += [f"  UNKEPT {path}:{line}  {pattern}" for path, line, pattern in unkept]
    lines += [f"  STALE {path}  {pattern}" for path, pattern in stale]
    if unkept or stale:
        lines.append(
            "Route the read or draw through crate::determinism, or add an "
            "ALLOW_LIST entry saying whether it can change a routing decision "
            "and why it stays; delete an entry with no hit left."
        )
    return (1 if unkept or stale else 0), "\n".join(lines)
