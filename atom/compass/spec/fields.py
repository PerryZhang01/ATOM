# SPDX-License-Identifier: MIT
"""The schema as data: every field the machine spec has, and what it may hold.

The table is the schema. Reading a document means checking it against this and
nothing else, which is what makes the schema closed -- a key with no row here is
refused wherever it sits, so the separation rule holds without anyone
maintaining a list of the knobs it excludes.

Four things about the table are load-bearing.

**A spec-peak number is marked, and its derate is derived from the mark.** No
row declares a derate. `closed()` adds one to the block of every field marked
`peak`, so a number that came off a datasheet cannot be added without also
requiring the haircut that turns it into an achievable one. Listing the derates
by hand would let the two drift: a new peak term would be accepted with no
derate, which is exactly the quiet datasheet-as-achievable the rule exists to
stop. Five derates come out of the four blocks here plus the one in a tokenizer
entry, and none of them is written down.

**A runtime constant keyed by tensor-parallel width is a table, not a law.** The
measured values at widths 1, 2, 4 and 8 fit no fixed-plus-per-peer form, so
there is nothing to interpolate along and a width that was not measured cannot
be produced from the widths that were. `WIDTH_TABLE` holds what was measured and
the accessor refuses the rest by name.

**A checked value is the spec's own.** Every kind returns a copy rather than
what the document handed over: `tuple` for a list of names, `dict` for a width
table, and a walk for a tokenizer entry, which holds mappings and lists of its
own where the other two hold scalars. A frozen spec that aliased its caller's
document would have a digest that moves afterwards, under an artifact that
already recorded it.

**Types are checked positively.** A count is an `int` that is not a `bool`, a
quantity is a positive real, a derate is a fraction in (0, 1]. `bool` is an
`int` in Python, so a check that only asked for an integer would accept `True`
as a core count; and a quantity that is allowed to be zero or negative turns a
division in a cost term into an infinity or a negative duration far from here.

Everything in the table is required. A machine has an interconnect whether or
not a particular run drives it, and the value of refusing a missing term is that
the refusal happens while a person is authoring the document rather than in the
middle of a simulated run. The exceptions are the provenance fields `merge`
fills in later, and the activation entries, whose absence falls back to a
declared form.
"""

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .rules import Rule, SpecRefusal


class Kind(enum.Enum):
    """What a field may hold."""

    TEXT = "text"
    COUNT = "count"
    QUANTITY = "quantity"
    DERATE = "derate"
    WIDTH_TABLE = "width_table"
    NAMES = "names"
    TOKENIZERS = "tokenizers"
    ACTIVATIONS = "activations"


@dataclass(frozen=True, slots=True)
class Field:
    """One field of the schema: where it sits, what it holds, and its role."""

    path: str
    kind: Kind
    peak: bool = False
    required: bool = True

    @property
    def block(self) -> str:
        """The dotted path of the block this field sits in, or the empty string."""
        return self.path.rpartition(".")[0]


def closed(fields: tuple[Field, ...]) -> tuple[Field, ...]:
    """The declared fields plus the derate each spec-peak number obliges."""
    derates = dict.fromkeys(
        f"{field.block}.derate" if field.block else "derate"
        for field in fields
        if field.peak
    )
    return fields + tuple(Field(path, Kind.DERATE) for path in derates)


#: The machine spec, one row per field, in the order a document reads.
DECLARED = (
    Field("schema_version", Kind.COUNT),
    Field("name", Kind.TEXT),
    Field("provenance.authored_by", Kind.TEXT),
    Field("provenance.date", Kind.TEXT),
    Field("provenance.method", Kind.TEXT),
    Field("provenance.fragments", Kind.NAMES, required=False),
    Field("provenance.notes", Kind.TEXT, required=False),
    Field("host.cpu.cores_physical", Kind.COUNT),
    Field("host.cpu.cores_logical", Kind.COUNT),
    Field("host.tokenizers", Kind.TOKENIZERS),
    Field("host.ipc.zmq_roundtrip_s", Kind.QUANTITY),
    Field("host.ipc.shm_broadcast_s", Kind.QUANTITY),
    Field("host.admission_fixed_s", Kind.QUANTITY),
    Field("device.name", Kind.TEXT),
    Field("device.arch", Kind.TEXT),
    Field("device.count_per_node", Kind.COUNT),
    Field("device.memory.capacity_bytes", Kind.QUANTITY),
    Field("device.memory.bandwidth_bytes_per_s", Kind.QUANTITY, peak=True),
    Field("device.compute.bf16_flops", Kind.QUANTITY, peak=True),
    Field("device.compute.fp8_flops", Kind.QUANTITY, peak=True),
    Field(
        "device.runtime_constants.driver_and_collective_reserve_bytes",
        Kind.WIDTH_TABLE,
    ),
    Field(
        "device.runtime_constants.allocator_retained_after_load_bytes",
        Kind.WIDTH_TABLE,
    ),
    Field("device.runtime_constants.persistent_forward_buffer_bytes", Kind.QUANTITY),
    Field("device.runtime_constants.cudagraph_pool.w1_base_bytes", Kind.QUANTITY),
    Field(
        "device.runtime_constants.cudagraph_pool.w1_bytes_per_captured_token",
        Kind.QUANTITY,
    ),
    Field("device.runtime_constants.cudagraph_pool.w_gt1_flat_bytes", Kind.QUANTITY),
    Field("device.activations", Kind.ACTIVATIONS, required=False),
    Field("device.software_pinned_to.rocm", Kind.TEXT),
    Field("device.software_pinned_to.aiter", Kind.TEXT),
    Field("device.software_pinned_to.rccl", Kind.TEXT),
    Field("interconnect.intra_node.topology", Kind.TEXT),
    Field(
        "interconnect.intra_node.link_bandwidth_bytes_per_s", Kind.QUANTITY, peak=True
    ),
    Field("interconnect.intra_node.link_latency_s", Kind.QUANTITY),
    Field(
        "interconnect.inter_node.link_bandwidth_bytes_per_s", Kind.QUANTITY, peak=True
    ),
    Field("interconnect.inter_node.link_latency_s", Kind.QUANTITY),
    Field("interconnect.router_relay_s", Kind.QUANTITY),
)

SCHEMA = closed(DECLARED)
BY_PATH = {field.path: field for field in SCHEMA}
#: Every dotted prefix of a declared path, so a document can be walked into.
BLOCKS = frozenset(
    path.rsplit(".", index + 1)[0]
    for path in BY_PATH
    for index in range(path.count("."))
)
#: The names the runtime constants are keyed under, without their block prefix.
RUNTIME_CONSTANTS = "device.runtime_constants"
#: The stack the constants are pinned to, one component per field.
PINNED = tuple(
    field.path.rsplit(".", 1)[-1]
    for field in SCHEMA
    if field.block == "device.software_pinned_to"
)
SCHEMA_VERSION = 1


def _refuse(path: str, wanted: str, value: object) -> None:
    raise SpecRefusal(
        Rule.SHAPE,
        f"`{path}` holds {value!r}, which is not {wanted}",
        f"write {wanted} there",
    )


def _detached(value: object) -> object:
    """A document value copied, so the document cannot move it afterwards.

    A list of names and a width table get this from `tuple` and `dict`, because
    what they hold is scalars and the constructor is the whole copy. A
    tokenizer entry holds mappings and lists, so the same guarantee is a walk.
    """
    if isinstance(value, Mapping):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_detached(item) for item in value]
    return value


def check(field: Field, value: object, path: str) -> object:
    """The value as the schema keeps it, or a refusal naming what was wanted."""
    kind = field.kind
    if kind is Kind.TEXT:
        if not isinstance(value, str) or not value.strip():
            _refuse(path, "a non-empty string", value)
        return value
    if kind is Kind.COUNT:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _refuse(path, "a positive whole number", value)
        return value
    if kind is Kind.QUANTITY:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            _refuse(path, "a positive number", value)
        return value
    if kind is Kind.DERATE:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _refuse(path, "a fraction in (0, 1]", value)
        if not 0 < value <= 1:
            raise SpecRefusal(
                Rule.DERATE,
                f"`{path}` is {value!r}, which is not a fraction in (0, 1]",
                "a derate is the declared gap between a spec-peak number and "
                "what a kernel reaches, so it can shrink a peak and never grow it",
            )
        return value
    if kind is Kind.NAMES:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            _refuse(path, "a list of names", value)
        for name in value:
            if not isinstance(name, str) or not name.strip():
                _refuse(path, "a list of non-empty strings", value)
        return tuple(value)
    if kind is Kind.WIDTH_TABLE:
        if not isinstance(value, Mapping) or not value:
            _refuse(path, "a non-empty table keyed by tensor-parallel width", value)
        for width, measured in value.items():
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                _refuse(f"{path}[{width!r}]", "a positive tensor-parallel width", width)
            check(Field(f"{path}.{width}", Kind.QUANTITY), measured, f"{path}[{width}]")
        return dict(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _refuse(path, f"a list of {kind.value} entries", value)
    return tuple(_detached(entry) for entry in value)
