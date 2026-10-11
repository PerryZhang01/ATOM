# SPDX-License-Identifier: MIT
"""The machine specification: what Compass is told about the machine it simulates.

Compute capability, memory size, bandwidth and interconnect are configured, not
read from a device runtime, so this artifact is load-bearing rather than
convenient -- a simulated run is only as true as the document that describes the
hardware it is pretending to be.

One rule decides what may appear in it. The spec describes the **machine**;
ATOM's config describes the **deployment**; anything ATOM could configure
belongs to ATOM, and Compass reads it from there. A spec that carried a
tensor-parallel width or a block size would silently contradict the engine it
claims to describe, because the engine was launched with its own.

The rule is kept by the schema being closed rather than by a list of exclusions,
so it holds for knobs nobody has thought of yet. Everything else here is a
refusal: a missing runtime constant names itself and the width nobody measured,
a spec-peak number without its derate is declined, a stack pin that no longer
matches what is loaded warns and names both versions, and an unmeasured
tokenizer is refused rather than guessed. A refusal is read by whoever hit it,
so a path that does not resolve says which of the three it is: a key the schema
never declared, a declared field the spec in hand was assembled without, or a
block, which groups fields and holds no value of its own.

Nothing here imports a tensor library, a device runtime or the engine, so a spec
can be authored, read and checked anywhere Python runs. The document is a plain
mapping, and turning a file into one belongs to the tools built over this, which
can declare the parser they need.

Three verbs work over that mapping. `merge` combines the fragments separate
probes produce and refuses the ones that name different machines, which
is the hazard the schema cannot see: two tokenizers measured on two machines
contradict nothing in their shape. `validate` reports everything a document is
missing or inconsistent about, rather than completing it. `explain` takes a
predicted quantity back to the spec fields under it, which is what makes echoing
the whole spec into every artifact worth anything. `across_ranks` reduces the
one reading that is taken per rank, keeping the smallest and reporting how far
the ranks disagreed.

A probe fills part of the document for them to work over. `tokenizer_fragment`
is the tier that needs no device: it takes the rates somebody measured, checks
them against the table a tokenizer entry has, and emits them together with the
core counts of the processor it read here -- which is what makes a fragment
composed somewhere other than its target contradict the target rather than
merge quietly into it. How the rates were obtained is the caller's to state,
because `tokenizer_fragment` did not measure them; `tokenizer_sweep` is one
way to, timing a loaded tokenizer over a length sweep. `probe_for` answers which probe
fills a runtime constant at a width, and refuses for the one entry no probe
fills rather than naming a probe that would not produce it.

`DeviceMemory` is what a probe that does start an engine reads off each card,
kept as four numbers rather than as the one the spec wants: free, total,
reserved and the cache budget. `non_torch_across_ranks` takes those to the
single reserve a spec carries, declining a rank whose readings cannot have come
off one card, a rank whose cache was sized by what was free rather than by its
budget, a spread too wide to be one machine, and a reading too large for what
the collective terms predict at that width. The last of those is the only one
that sees a card every rank shares with the same neighbour, and it can be asked
only because free and total arrived apart.
"""

from .explain import QUANTITIES, Basis, Contribution, explain
from .fields import DECLARED, SCHEMA, SCHEMA_VERSION, Field, Kind
from .machine import MachineSpec
from .memory import ABSOLUTE_LIMIT, DeviceMemory, non_torch_across_ranks
from .merge import Fragment, Merge, merge
from .probes import probe_for, tokenizer_fragment, tokenizer_sweep
from .ranks import SPREAD_LIMIT, RankSpread, across_ranks
from .rules import (
    DEPLOYMENT_OWNED,
    FingerprintMismatch,
    Rule,
    SpecRefusal,
    StackMismatch,
)
from .tokenizers import Backend, TokenizerEntry, TokenizerKey, TokenizerTable
from .validate import Validation, validate

__all__ = [
    "ABSOLUTE_LIMIT",
    "DECLARED",
    "DEPLOYMENT_OWNED",
    "QUANTITIES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SPREAD_LIMIT",
    "Backend",
    "Basis",
    "Contribution",
    "DeviceMemory",
    "Field",
    "FingerprintMismatch",
    "Fragment",
    "Kind",
    "MachineSpec",
    "Merge",
    "RankSpread",
    "Rule",
    "SpecRefusal",
    "StackMismatch",
    "TokenizerEntry",
    "TokenizerKey",
    "TokenizerTable",
    "Validation",
    "across_ranks",
    "explain",
    "merge",
    "non_torch_across_ranks",
    "probe_for",
    "tokenizer_fragment",
    "tokenizer_sweep",
    "validate",
]
