# SPDX-License-Identifier: MIT
"""What a refusal says, and the one list that makes an unknown key legible.

A refusal here is a result, not an error path, so it is built to be read by
whoever hit it. Three parts, always: which rule declined, what in the document
tripped it, and what would satisfy it. A message that names only the first two
tells a user their spec is wrong and leaves them to guess the fix, and the fix
is the only part they need.

The same reading decides how many refusals there are. Three different things
can stop a dotted path resolving and they want three different actions, so
there are three refusals and not one. A key with no row in the field table is a
typo or a knob that belongs to the deployment. A declared field this object
holds no value for is the schema's own field in a spec assembled from parts. A
path that names a block names a group of fields rather than a value, which is
the one of the three a reader can act on without changing anything. A single
message would be true of whichever case it was written against and would send
the other two readers to the schema to find what they asked for sitting there.

The separation rule is the one that decides every inclusion question:

    the spec describes the machine; ATOM's config describes the deployment;
    anything ATOM could configure belongs to ATOM, and Compass reads it there.

It is enforced by the schema being **closed** rather than by the list below. A
document key that is not a declared field is refused wherever it sits, so a
spec cannot carry a tensor-parallel width, a block size or a memory-utilisation
fraction even though no rule anywhere names them -- and it equally cannot carry
the next such knob, which nobody has thought to list yet. A denylist would have
to be complete to work; a closed schema is complete by construction.

`DEPLOYMENT_OWNED` exists only to improve the message. When the unknown key is
one of the knobs a spec author is most likely to reach for, the refusal says
where ATOM configures it instead of saying merely that it is unrecognised. The
values name engine-argument fields rather than command-line flags because the
field names are what a test can check against the engine's own surface, and a
list that drifts away from that surface would be folklore inside a month.
"""

import enum
from typing import NoReturn


class Rule(enum.Enum):
    """The rules a spec is read against, as a refusal names them, and the one a
    probe filling it is held to.

    The last is not about a document. A probe that cannot take its reading has
    nothing wrong with its spec yet -- the host under it will not say what it
    is -- and naming a document rule in front of that describes something that
    did not happen, to a reader who then looks for it in the file.
    """

    SEPARATION = "the spec describes the machine, not the deployment"
    ONE_MACHINE = "one spec describes one host"
    RANK_AGREEMENT = "ranks of one group measure one machine"
    DEVICE_WIDE = "a device-wide reading is the whole card's, not one engine's"
    NO_DEFAULTS = "a runtime constant has no default"
    DERATE = "a spec-peak number carries a derate"
    PINNED_STACK = "the constants are pinned to a software stack"
    TOKENIZER_IDENTITY = "a tokenizer is probed or it is refused"
    SHAPE = "the document has the shape the schema declares"
    TOTALITY = "a spec resolves every field the document was required to state"
    ADDRESSING = "a field is asked for by the path of the field itself"
    VERSION = "the reader knows the schema version the document was written to"
    MEASURED = "a probe reports what it read, and refuses what it could not"


#: Knobs the engine already owns, mapped to where it owns them. Not the
#: mechanism -- the closed schema is -- only the part of the message that turns
#: "unrecognised" into "ATOM configures this, and Compass reads it from there".
DEPLOYMENT_OWNED = {
    "block_size": "EngineArgs.block_size",
    "cudagraph_capture_sizes": "EngineArgs.cudagraph_capture_sizes",
    "cudagraph_mode": "EngineArgs.cudagraph_mode",
    "gpu_memory_utilization": "EngineArgs.gpu_memory_utilization",
    "kv_cache_dtype": "EngineArgs.kv_cache_dtype",
    "level": "EngineArgs.level",
    "max_model_len": "EngineArgs.max_model_len",
    "max_num_seqs": "EngineArgs.max_num_seqs",
    "tensor_parallel_size": "EngineArgs.tensor_parallel_size",
}


class SpecRefusal(Exception):
    """A declined answer that names its rule and what would satisfy it."""

    def __init__(self, rule: Rule, what: str, remedy: str) -> None:
        self.rule = rule
        self.what = what
        self.remedy = remedy
        super().__init__(f"{rule.value}: {what}. {remedy}")


class StackMismatch(UserWarning):
    """The stack a spec was probed against is not the stack now loaded."""


class FingerprintMismatch(UserWarning):
    """The tokenizer now loaded is not the one whose rates were probed."""


def refuse_block(path: str, holds: tuple[str, ...]) -> NoReturn:
    """Decline a path that names a block, naming what the block groups instead.

    The cheapest of the three to act on, and the one that reads worst when it
    is lumped in with a typo: the path is in the schema, the spec is complete,
    and the only thing wrong is that a block groups fields and is not itself a
    value. So the remedy is the list of what it groups rather than a direction
    to go and look.

    The list is what the schema puts one segment under the block, which may
    include other blocks, so it is stated as what is there rather than as an
    instruction to ask for one of them -- for `device` several of those names
    would arrive back here.
    """
    raise SpecRefusal(
        Rule.ADDRESSING,
        f"`{path}` is a block of this schema, not one of its fields",
        "a block groups fields and holds no value of its own; the schema puts "
        + ", ".join(holds)
        + " under it, and a field is answered at its whole path",
    )


def refuse_absent_field(path: str, required: bool) -> NoReturn:
    """Decline a declared field this spec holds no value for, by how it is declared.

    The companion to `refuse_unknown_key`, and the reason the two are separate:
    a key with no row in the table is a typo or a deployment knob, and the
    reader is sent to the schema. A key with a row that this object does not
    carry is the schema's own field, and sending that reader to the schema
    sends them to find it there and stop.

    Which half of the rule applies is decided by how the field is declared,
    because the two have different fixes. A required field cannot be missing
    from a spec the reader produced -- it resolves every one of them or refuses
    -- so the object was assembled some other way, which is what a probe
    fragment is, and the fix is in whatever assembled it. An optional field the
    document did not state is missing from a spec that was read, and the fix is
    in the document.
    """
    if required:
        refuse_absent_fields((path,))
    raise SpecRefusal(
        Rule.TOTALITY,
        f"`{path}` is declared by this schema as optional, and this spec states "
        "no value for it",
        "write it in the document if a reader needs it; a field the document "
        "leaves out is left out here rather than invented",
    )


def refuse_absent_fields(paths: tuple[str, ...]) -> NoReturn:
    """Decline required fields this spec holds no value for, naming every one.

    The required half of `refuse_absent_field`, which reads one path through
    here, so a question over several fields and a question over one are
    declined in the same words. Every missing field is named rather than the
    first: each is a separate fragment to merge, and a refusal naming one of
    them sends the reader back once per field.
    """
    one = len(paths) == 1
    named = ", ".join(f"`{path}`" for path in paths)
    are, them, suppliers = (
        ("is", "it", "fragment that supplies this field")
        if one
        else ("are", "them", "fragments that supply these fields")
    )
    raise SpecRefusal(
        Rule.TOTALITY,
        f"{named} {are} declared by this schema, and this spec carries no value "
        f"for {them}",
        "a spec read with `MachineSpec.from_mapping` resolves every required "
        f"field, so this one was assembled from parts; merge the {suppliers} "
        f"before asking for {them}",
    )


def refuse_unknown_key(path: str) -> NoReturn:
    """Decline a key the schema does not declare, saying why it is not one."""
    owner = DEPLOYMENT_OWNED.get(path.rsplit(".", 1)[-1])
    if owner is not None:
        raise SpecRefusal(
            Rule.SEPARATION,
            f"`{path}` states how ATOM was launched, not what the machine is",
            f"remove it; {owner} carries it and Compass reads it from there",
        )
    raise SpecRefusal(
        Rule.SEPARATION,
        f"`{path}` is not a field of this schema",
        "the schema is closed: a property of the machine is added to the field "
        "table deliberately, and anything ATOM could configure stays in ATOM's "
        "config, where Compass reads it",
    )
