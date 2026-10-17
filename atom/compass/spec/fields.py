# SPDX-License-Identifier: MIT
"""The schema as data: every field the machine spec has, and what it may hold.

The table is the schema, so a key with no row here is refused wherever it sits.

**A spec-peak number is marked, and its derate is derived from the mark.**
`closed()` adds a derate to the block of every field marked `peak`, so a peak
cannot be added without the haircut that makes it achievable.

**A checked value is the spec's own.** Every kind returns a copy, so a frozen
spec's digest cannot move when the caller's document does.

**Types are checked positively.** A count is an `int` that is not a `bool`, a
quantity is a positive real, a derate is a fraction in (0, 1].

Every field is required except the provenance fields `merge` fills in,
`fp4_flops`, the pipeline send terms, a link's `link_setup_s` (zero when
absent), the `host.http_intake` pair and the `host_link` pair. An engine
start's memory readings are no field here: `compass measure memory` publishes
them per configuration, as a `memory_readings` entry.

**Every field says where it came from.** `provenance.origin` is one of
`ORIGINS`. A document of several origins says `mixed`, and `merge` writes
`provenance.origins`: one entry per field, holding the field's origin and, on
a probed field, the subject the probe ran on (`SUBJECT`).
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
    NAMES = "names"
    TOKENIZERS = "tokenizers"
    ORIGIN = "origin"
    SUBJECT = "subject"
    ORIGINS = "origins"


#: Where a value came from; `mixed` describes a document, never one field.
ORIGINS = ("datasheet", "probed", "assumed", "mixed")
#: What a probe ran on, any of which it may state. `record` is the digest of
#: the record of the run it read, `sha256:` over the server log.
SUBJECT = {
    "model": Kind.TEXT,
    "width": Kind.COUNT,
    "source_root": Kind.TEXT,
    "record": Kind.TEXT,
}


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
FIELDS = (
    Field("schema_version", Kind.COUNT),
    Field("name", Kind.TEXT),
    Field("provenance.authored_by", Kind.TEXT),
    Field("provenance.date", Kind.TEXT),
    Field("provenance.origin", Kind.ORIGIN),
    Field("provenance.subject", Kind.SUBJECT, required=False),
    Field("provenance.origins", Kind.ORIGINS, required=False),
    Field("provenance.fragments", Kind.NAMES, required=False),
    Field("provenance.notes", Kind.TEXT, required=False),
    Field("host.cpu.cores_physical", Kind.COUNT),
    Field("host.cpu.cores_logical", Kind.COUNT),
    Field("host.tokenizers", Kind.TOKENIZERS),
    Field("host.ipc.zmq_roundtrip_s", Kind.QUANTITY),
    Field("host.ipc.shm_broadcast_s", Kind.QUANTITY),
    # The frontend event loop's time to take in one HTTP request.
    Field("host.http_intake.fixed_s", Kind.QUANTITY, required=False),
    Field("host.http_intake.per_byte_s", Kind.QUANTITY, required=False),
    Field("device.name", Kind.TEXT),
    Field("device.arch", Kind.TEXT),
    Field("device.count_per_node", Kind.COUNT),
    Field("device.memory.capacity_bytes", Kind.QUANTITY),
    Field("device.memory.bandwidth_bytes_per_s", Kind.QUANTITY, peak=True),
    Field("device.compute.bf16_flops", Kind.QUANTITY, peak=True),
    Field("device.compute.fp8_flops", Kind.QUANTITY, peak=True),
    Field("device.compute.fp4_flops", Kind.QUANTITY, peak=True, required=False),
    Field("device.software_pinned_to.rocm", Kind.TEXT),
    Field("device.software_pinned_to.aiter", Kind.TEXT),
    Field("device.software_pinned_to.rccl", Kind.TEXT),
    Field("device.software_pinned_to.torch", Kind.TEXT),
    Field("interconnect.intra_node.topology", Kind.TEXT),
    Field("interconnect.intra_node.domain_devices", Kind.COUNT),
    Field(
        "interconnect.intra_node.link_bandwidth_bytes_per_s", Kind.QUANTITY, peak=True
    ),
    Field("interconnect.intra_node.link_latency_s", Kind.QUANTITY),
    Field("interconnect.intra_node.link_setup_s", Kind.QUANTITY, required=False),
    # A pipeline stage's send, per carrier; a pipeline run requires them.
    *(
        Field(f"interconnect.intra_node.pp.{carrier}.{name}", kind, required=False)
        for carrier in ("rccl", "gloo")
        for name, kind in (
            ("eager_threshold_bytes", Kind.COUNT),
            ("latency_s", Kind.QUANTITY),
            ("overhead_s", Kind.QUANTITY),
            ("per_byte_s", Kind.QUANTITY),
            ("rendezvous_fixed_s", Kind.QUANTITY),
        )
    ),
    Field(
        "interconnect.inter_node.link_bandwidth_bytes_per_s", Kind.QUANTITY, peak=True
    ),
    Field("interconnect.inter_node.link_latency_s", Kind.QUANTITY),
    Field("interconnect.inter_node.link_setup_s", Kind.QUANTITY, required=False),
    Field("interconnect.router_relay_s", Kind.QUANTITY),
    # The device-to-host copy an lmcache server's stores and loads ride, as a
    # copy reaches it (not a peak); a run with an lmcache server requires it.
    Field(
        "interconnect.host_link.bandwidth_bytes_per_s", Kind.QUANTITY, required=False
    ),
    Field("interconnect.host_link.latency_s", Kind.QUANTITY, required=False),
)

SCHEMA = closed(FIELDS)
BY_PATH = {field.path: field for field in SCHEMA}
#: Every dotted prefix of a declared path, so a document can be walked into.
BLOCKS = frozenset(
    path.rsplit(".", index + 1)[0]
    for path in BY_PATH
    for index in range(path.count("."))
)
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

    A list of names gets this from `tuple`, because what it holds is scalars
    and the constructor is the whole copy. A tokenizer entry holds mappings and
    lists, so the same guarantee is a walk.
    """
    if isinstance(value, Mapping):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_detached(item) for item in value]
    return value


def origin(value: object, path: str, *, of_a_field: bool = False) -> str:
    """An origin as written, or a refusal; one field's origin is never `mixed`."""
    allowed = ORIGINS[:-1] if of_a_field else ORIGINS
    if value not in allowed:
        raise SpecRefusal(
            Rule.SHAPE,
            f"`{path}` holds {value!r}, which is not an origin",
            f"write one of {', '.join(allowed)}; an origin nobody defined says "
            "nothing about where a value came from",
        )
    return value


def _subject(value: object, path: str) -> dict:
    if not isinstance(value, Mapping) or not value or not set(value) <= set(SUBJECT):
        _refuse(path, f"a mapping of some of {', '.join(SUBJECT)}", value)
    return {k: check(Field(k, SUBJECT[k]), v, f"{path}.{k}") for k, v in value.items()}


def _origins(value: object, path: str) -> dict:
    if not isinstance(value, Mapping):
        _refuse(path, "a mapping from field to origin", value)
    origins = {}
    for label, entry in value.items():
        where = f"{path}[{label}]"
        if not isinstance(entry, Mapping) or "origin" not in entry:
            _refuse(where, "a mapping holding an `origin`", entry)
        stated = origin(entry["origin"], f"{where}.origin", of_a_field=True)
        subject = {key: held for key, held in entry.items() if key != "origin"}
        if subject and stated != "probed":
            _refuse(where, "a probed field, the only kind with a subject", entry)
        origins[label] = {
            "origin": stated,
            **(_subject(subject, where) if subject else {}),
        }
    return origins


def check(field: Field, value: object, path: str) -> object:
    """The value as the schema keeps it, or a refusal naming what was wanted."""
    kind = field.kind
    if kind is Kind.ORIGIN:
        return origin(value, path)
    if kind is Kind.SUBJECT:
        return _subject(value, path)
    if kind is Kind.ORIGINS:
        return _origins(value, path)
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
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _refuse(path, f"a list of {kind.value} entries", value)
    return tuple(_detached(entry) for entry in value)
