# SPDX-License-Identifier: MIT
"""The probe that needs no device: a tokenizer swept here, as a fragment.

`tokenizer_sweep` times a loaded tokenizer and fits `fixed + tokens / rate` on
relative error, refusing a fit whose worst residual exceeds the caller's bound.
`tokenizer_fragment` composes the entries a caller measured with the core
counts of the processor it runs on. Those counts are required fields of the
machine, so a tokenizer swept on another host than the one the device fragment
describes contradicts it there, and `merge` refuses the pair. The counts come
from the kernel's per-processor topology and are refused rather than guessed.
How the rates were obtained is the caller's to state: `origin` has no default.
"""

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from .fields import SCHEMA_VERSION
from .machine import entry_label
from .merge import Fragment
from .rules import Rule, SpecRefusal
from .tokenizers import table

#: Where the kernel publishes one directory per processor.
PROCESSORS = "/sys/devices/system/cpu"


def _unreadable(what: str) -> NoReturn:
    raise SpecRefusal(
        Rule.MEASURED,
        f"the processors of this host cannot be counted: {what}",
        "write `host.cpu.cores_physical` and `host.cpu.cores_logical` into the "
        "fragment by hand; the logical count is not the physical one, and on a "
        "host with simultaneous multithreading the two differ by the factor "
        "that makes the pair worth recording",
    )


def _reading(processor: pathlib.Path, name: str) -> int:
    """One topology number of one processor, or a refusal naming what it held."""
    try:
        published = (processor / "topology" / name).read_text()
    except OSError as unreadable:
        _unreadable(f"{processor.name} publishes no topology ({unreadable})")
    try:
        return int(published.strip())
    except ValueError:
        _unreadable(f"{processor.name} publishes {name} as {published!r}")


def cpu_counts(processors: str = PROCESSORS) -> dict[str, int]:
    """The `host.cpu` block of the machine this is running on, or a refusal."""
    root = pathlib.Path(processors)
    logical = sorted(path for path in root.glob("cpu[0-9]*") if path.name[3:].isdigit())
    if not logical:
        _unreadable(f"{root} lists no processor")
    cores = {
        (_reading(path, "physical_package_id"), _reading(path, "core_id"))
        for path in logical
    }
    return {"cores_physical": len(cores), "cores_logical": len(logical)}


def tokenizer_fragment(
    entries: Sequence[Mapping[str, Any]],
    *,
    machine: str,
    authored_by: str,
    date: str,
    origin: str,
    subject: Mapping[str, Any] | None = None,
    source: str = "tokenizer",
    processors: str = PROCESSORS,
    band: Mapping[str, Any] | None = None,
) -> Fragment:
    """A fragment of the entries a caller measured and the processor read here.

    `origin` is how those entries were obtained, and it has no default because
    nothing in this function establishes one; `subject` is what a probe ran on,
    and `band` the `p10` and `p90` of each rate over its sweeps.
    """
    if not entries:
        raise SpecRefusal(
            Rule.TOKENIZER_IDENTITY,
            f"the tokenizer fragment for {machine!r} carries no tokenizer",
            "sweep a tokenizer and emit its entry; a fragment carrying only "
            "this host's core counts states nothing probed about the thing "
            "the probe is for",
        )
    # Read for its refusals: an entry that is not a tokenizer entry is refused
    # here, beside the caller that supplied it, rather than after a merge.
    table(entries)
    origins = {
        entry_label(entry["id"], entry.get("backend")): {"origin": origin}
        | dict(subject or {})
        | dict(band or {})
        for entry in entries
    }
    counts = cpu_counts(processors)
    origins |= {f"host.cpu.{count}": {"origin": "probed"} for count in counts}
    document = {
        "schema_version": SCHEMA_VERSION,
        "name": machine,
        "provenance": {
            "authored_by": authored_by,
            "date": date,
            "origin": "mixed",
            "origins": origins,
        },
        "host": {"cpu": counts, "tokenizers": list(entries)},
    }
    return Fragment.from_mapping(document, source)


def _fit(what: str, points: Sequence[tuple[int, float]], bound: float):
    """`fixed + tokens / rate` fitted to `points` on relative error, or a refusal.

    Each point is weighted by its own time, so a long call does not drown the
    fixed cost a one-token call is made of.
    """
    uu = uv = vv = u1 = v1 = 0.0
    for tokens, seconds in points:
        u, v = 1.0 / seconds, tokens / seconds
        uu, uv, vv, u1, v1 = uu + u * u, uv + u * v, vv + v * v, u1 + u, v1 + v
    det = uu * vv - uv * uv
    fixed = (u1 * vv - v1 * uv) / det
    per_token = (uu * v1 - uv * u1) / det
    if fixed <= 0 or per_token <= 0:
        raise SpecRefusal(
            Rule.MEASURED,
            f"{what} fits a fixed cost of {fixed!r} s and {per_token!r} s per "
            f"token over {[tokens for tokens, _ in points]} tokens",
            "both are positive on a tokenizer that does work per call and per "
            "token; sweep lengths that reach below and above the point where "
            "the two costs are equal, on a quiet host",
        )
    tokens, seconds = max(
        points, key=lambda p: abs((fixed + p[0] * per_token) / p[1] - 1)
    )
    worst = (fixed + tokens * per_token) / seconds - 1
    if abs(worst) > bound:
        raise SpecRefusal(
            Rule.MEASURED,
            f"{what} at {tokens} tokens took {seconds!r} s and the fit says "
            f"{fixed + tokens * per_token!r} s, {worst:+.1%} against a bound "
            f"of {bound:.1%}",
            "the cost is not fixed plus linear over this sweep, so no rate "
            "describes it; sweep a range where it is, or find what makes that "
            "length different before writing a rate down",
        )
    return fixed, 1.0 / per_token


def tokenizer_sweep(
    tokenizer,
    text: str,
    lengths: Sequence[int],
    *,
    bound: float,
    clock,
    repeats: int = 5,
) -> dict[str, float]:
    """The rates of a tokenizer entry, swept through a loaded tokenizer.

    Each length is a prefix of `text`'s tokens, timed `repeats` times through
    `encode` and `decode` with the fastest kept. Encode is fitted against the
    token count `encode` returns, since that is what a run charges for. A fit
    whose worst relative residual exceeds `bound` is refused, naming the
    length. `clock` reads seconds, such as `time.perf_counter`.
    """
    ids = tokenizer.encode(text)
    if len(ids) < max(lengths):
        raise SpecRefusal(
            Rule.MEASURED,
            f"the sweep text is {len(ids)} tokens and the sweep asks for "
            f"{max(lengths)}",
            "give a text at least as long as the longest length swept",
        )

    def fastest(call, argument) -> float:
        times = []
        for _ in range(repeats):
            start = clock()
            call(argument)
            times.append(clock() - start)
        return min(times)

    encoded, decoded = [], []
    for n in lengths:
        piece = tokenizer.decode(ids[:n])
        count = len(tokenizer.encode(piece))
        encoded.append((count, fastest(tokenizer.encode, piece)))
        decoded.append((n, fastest(tokenizer.decode, ids[:n])))
    encode_fixed, encode_rate = _fit("encode", encoded, bound)
    decode_fixed, decode_rate = _fit("decode", decoded, bound)
    return {
        "encode_fixed_s": encode_fixed,
        "encode_tokens_per_s": encode_rate,
        "decode_fixed_s": decode_fixed,
        "decode_tokens_per_s": decode_rate,
    }
