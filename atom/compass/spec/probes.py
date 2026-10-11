# SPDX-License-Identifier: MIT
"""What a probe that needs no device emits, and which probe fills which constant.

The first half of this module is the tier that needs no device; the second is
the table of which probe supplies each runtime constant, and the one entry that
no probe supplies. The reading that makes two fragments comparable comes first.

A probe measures part of a machine and emits a fragment: the fields it measured
plus the stanza saying which machine they are for, by whom, when and how. What
is here is the tier that needs no device: `tokenizer_sweep` times a loaded
tokenizer, `tokenizer_fragment` composes whatever rates it is handed -- and a
tokenizer sweep needs no device, so it is the measurement most likely to be taken
somewhere other than the machine it is authored for -- a laptop, a build agent,
the login node in front of a cluster. That is not an abuse of it. Tokenization
is single-threaded, cache-resident integer work, so the rates transfer across
processors of comparable class, and refusing every spec not authored on its own
target would refuse most specs that will ever be written.

**So a fragment composed here writes down the processor this ran on, and that
is the whole of its second job.** `host.cpu.cores_physical` and
`host.cpu.cores_logical` are already required fields of the machine, so a
fragment that fills them in is claiming that the host that composed it is the
machine its stanza names. When it is not
-- the tokenizer measured on the laptop, the device fields measured on the node,
both authored for the node -- the two fragments then state different numbers for
one field of one machine, and the merge refuses the pair with both readings and
both stanzas. Before this they shared no field at all: nothing in a fragment's
shape says where it was measured, so nothing contradicted anything, and the pair
combined into a document that describes neither host. Two fields that were
already required, and nothing added to the schema, are what turn that silence
into a refusal.

This does not make the transfer illegitimate, and it does not record where the
probe ran -- no field of a fragment does that. A tokenizer measured on a
comparable processor is still the intended use. What changes is that an author
who wants it now has to write the target's own core counts into the document,
rather than have the question go unasked.

**The counts come from the kernel's per-processor topology, and are refused
rather than guessed.** The physical count is the number of distinct
package-and-core pairs; the logical count is the number of processors. They
differ by the simultaneous-multithreading factor, so deriving either from the
other would be wrong by exactly the factor that makes the pair worth recording.
A host that publishes no topology, or one that publishes a file holding nothing
a number can be read out of, gets a refusal naming what could not be read: a
count taken off an unreadable file is a reading with no content behind it, and
it would travel into a spec looking exactly like a measurement.

**The rates are the caller's, and so is the claim about how they were
obtained.** This function composes a fragment and reads a topology. It does not
encode or decode anything, so it cannot know whether the rates it was handed
were swept on a loaded tokenizer, copied out of an older spec or typed in -- and
`provenance.method` is what a merge refusal quotes and what a run artifact
carries as the numbers' source. A method stamped here would be this function
asserting a measurement it did not take, over data it received. So the caller
states it, with no default to fall through to, and the core counts read here
ride along under whatever the caller states about its own rates.

**What the entries are is checked here rather than several verbs later.** They
are read against the closed table a tokenizer entry has, the same one a spec is
read against, so an entry that is not a tokenizer, or one missing the derate its
two rates oblige, is refused while the probe that produced it is still the thing
in front of the author. Left to `validate`, the same entry is refused after a
merge, in a document whose other fields came from elsewhere, and the fragment
that carried it has to be worked back to.

**Which probe fills which runtime constant, and the one entry that has none.**
The runtime constants keyed by tensor-parallel width are the terms with no law
behind them, so a width nobody measured is refused by name rather than
interpolated, and the remedy such a refusal offers is to go and run a probe.
`FILLED_BY` records which probe that is. Two of them fill these terms: one
starts a single engine on one card and reads the device's own reserve off it,
and one starts an engine per width above one and reads both width-keyed terms
at each. Between them they do not cover the allocator's retained bytes at width
one, and that hole is written here as a hole. The single-card probe is
specified to fill the reserve term and not this one; the multi-rank probe is
specified for the widths above one. Handing the entry to either would be
writing a capability into the table that nothing states, and a table entry
invented to close a gap travels onward as a measurement -- so asking for that
entry's probe returns a refusal naming what has to be measured by hand instead.

**And the hole is not a tier boundary somebody forgot to move.** The retained
bytes measure 1.1e6 at width one and 2.17e9 flat at every width above it,
because what is retained above one card is a collective staging pool that does
not exist on a single card. Running the multi-rank probe at width one would
therefore not be the same probe measuring a smaller number; it would be
measuring a different quantity and writing it under the same name. Which of the
two probes should own this entry, or whether it stays hand-measured, is a
decision about what each probe is specified to fill, and it is not one this
table can take on its own.

**A width is a count of cards, so the domain starts at one.** `FILLED_BY` says
which probe fills a constant at width one and above it, and there is nothing
below that for either probe to describe: no engine starts on zero cards, and a
negative width is not a deployment. So a non-positive width is refused by name
rather than answered with the probe that happens to sit on the other side of
the width-one test. That test is the only place a width reaches this module
from a caller instead of from a document, where the schema has already refused
it.

**The sweep fits `fixed + tokens / rate` on relative error and refuses a fit
it cannot stand behind.** A rate is charged on every call, short and long, so
a fit that is good in absolute seconds and 40% off on a one-token call is not
a rate. The bound is the caller's, and the refusal names the length that
broke it.
"""

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from .fields import SCHEMA_VERSION
from .merge import Fragment
from .rules import Rule, SpecRefusal
from .tokenizers import table

#: Where the kernel publishes one directory per processor.
PROCESSORS = "/sys/devices/system/cpu"

#: The probe that starts one engine on one card.
SINGLE_CARD = "device-memory"
#: The probe that starts one engine per tensor-parallel width above one.
MULTI_RANK = "device-runtime-constants"

#: Which probe fills each width-keyed runtime constant, at width one and above
#: it. `None` is a hole in the tools, not a field with a default behind it.
FILLED_BY = {
    "driver_and_collective_reserve_bytes": (SINGLE_CARD, MULTI_RANK),
    "allocator_retained_after_load_bytes": (None, MULTI_RANK),
}


def probe_for(constant: str, tp_width: int) -> str:
    """The probe that fills one width-keyed constant at one width, or a refusal."""
    if constant not in FILLED_BY:
        raise SpecRefusal(
            Rule.NO_DEFAULTS,
            f"`{constant}` is not one of the constants measured per "
            f"tensor-parallel width: {sorted(FILLED_BY)}",
            "ask this of a width-keyed runtime constant; a term that is flat "
            "in width has no width to ask about, and one that is not a "
            "runtime constant is not a probe's to fill",
        )
    if tp_width < 1:
        raise SpecRefusal(
            Rule.NO_DEFAULTS,
            f"tensor-parallel width {tp_width} is not a number of cards an "
            f"engine runs on, so no probe fills `{constant}` there",
            "ask about a width of one or more; the probes here fill this term "
            "at one card and above it, and a width below that is a wrong "
            "question rather than a term nobody has measured yet",
        )
    at_one, above_one = FILLED_BY[constant]
    probe = at_one if tp_width == 1 else above_one
    if probe is None:
        raise SpecRefusal(
            Rule.NO_DEFAULTS,
            f"no probe here fills `{constant}` at tensor-parallel width {tp_width}",
            f"measure it on an engine at that width and write the entry down "
            f"by hand: `{SINGLE_CARD}` reads the device's own reserve and not "
            f"this term, `{MULTI_RANK}` starts one engine per width above "
            "one, and neither is documented to produce this number. running "
            "the second one here would not be the same measurement taken "
            "smaller: what is retained above one card includes a collective "
            "staging pool that a single card has none of, so it would put a "
            "different quantity under this name",
        )
    return probe


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
    method: str,
    source: str = "tokenizer",
    processors: str = PROCESSORS,
) -> Fragment:
    """A fragment of the entries a caller measured and the processor read here.

    `method` is how those entries were obtained, and it has no default because
    nothing in this function establishes one.
    """
    if not entries:
        raise SpecRefusal(
            Rule.TOKENIZER_IDENTITY,
            f"the tokenizer fragment for {machine!r} carries no tokenizer",
            "sweep a tokenizer and emit its entry; a fragment carrying only "
            "this host's core counts states nothing measured about the thing "
            "the probe is for",
        )
    # Read for its refusals: an entry that is not a tokenizer entry is refused
    # here, beside the caller that supplied it, rather than after a merge.
    table(entries)
    document = {
        "schema_version": SCHEMA_VERSION,
        "name": machine,
        "provenance": {"authored_by": authored_by, "date": date, "method": method},
        "host": {"cpu": cpu_counts(processors), "tokenizers": list(entries)},
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
    """The four rates of a tokenizer entry, swept through a loaded tokenizer.

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
