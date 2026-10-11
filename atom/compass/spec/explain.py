# SPDX-License-Identifier: MIT
"""Where a number came from: the spec fields under it, and who measured them.

A run artifact carries the whole resolved spec, which makes every number it
reports recoverable in principle. This is what makes it recoverable in practice.
The reader has one quantity and one question -- what is this built out of -- and
the answer is the fields that feed it, each with the value the document holds,
the haircut applied to it, and the fragment that supplied it.

**It is a decomposition and never a total.** A summed check over these terms
once read +13.8% while hiding three separate errors, two of which cancelled and
the largest of which was a quarter of a single term. So each field is reported
on its own line in its own unit, and nothing here adds them up. The arithmetic
that combines them belongs to whatever consumes them, and a sum taken across
seconds, bytes and bytes-per-second would be a fiction anyway.

**A spec-peak number is shown twice, with the derate between them.** That is the
one decomposition this module can always do, and it is the one most worth doing:
the gap between a datasheet figure and what a kernel reaches is a declared
judgement, not a measurement, and a reader tracing an over-optimistic prediction
wants to see the declared figure beside the derated one rather than a single
number that is silently either.

Two ways to name what to explain. A **field** or a block of fields is named by
its dotted path, and is exact -- there is nothing to maintain and nothing to
drift. A **quantity** is named out of `QUANTITIES`, which lists the spec fields
each predicted quantity is built from; that is the spec half of the relationship
and the only half this package can see, so it is checked for naming real fields
and nothing more. A consumer that reaches for a term this table does not list
for it is the thing to fix, in the table.

A width-keyed constant explains at the width asked for, or at every width that
was measured when no width is given; asking at a width nobody measured is
refused by name, here as everywhere else. Tokenizer rates explain per entry and
per rate, because a spec holding two tokenizers has two different answers and a
reader tracing a tokenization cost needs to see which of them was used.

A term with nothing to explain is refused one of two ways, because the two want
different actions. A term the schema does not know is asked for again by a real
path. A term the schema does know, with no field under it that this spec holds,
would come back empty however it was spelled, so it is refused as the absent
field it is -- the same refusal `MachineSpec.value` gives for that field.

A term naming several fields is refused when any required field under it is
absent, and the refusal names every one of them. Explaining from the fields
that are there would present part of a basis as the whole of it. An optional
field the document left out is not missing, so a block holding one still
explains from the rest.

A field the spec does hold is never refused here, even when it holds nothing:
a tokenizer table with no entries explains to no rows, as `MachineSpec.value`
answers it with none. It is still printed, as the empty value it holds, so a
reader can tell a field that contributed nothing from one that was left out.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .fields import BY_PATH, SCHEMA, Kind
from .machine import MachineSpec
from .merge import Merge, entry_label
from .rules import Rule, SpecRefusal, refuse_absent_field, refuse_absent_fields
from .tokenizers import ENTRY_FIELDS

#: Predicted quantities, and the spec fields each one is built out of.
QUANTITIES: Mapping[str, tuple[str, ...]] = {
    "kv_blocks": (
        "device.memory.capacity_bytes",
        "device.runtime_constants.driver_and_collective_reserve_bytes",
        "device.runtime_constants.allocator_retained_after_load_bytes",
        "device.runtime_constants.persistent_forward_buffer_bytes",
        "device.runtime_constants.cudagraph_pool.w1_base_bytes",
        "device.runtime_constants.cudagraph_pool.w1_bytes_per_captured_token",
        "device.runtime_constants.cudagraph_pool.w_gt1_flat_bytes",
        "device.activations",
    ),
    "admission": (
        "host.admission_fixed_s",
        "host.ipc.zmq_roundtrip_s",
        "host.ipc.shm_broadcast_s",
        "host.tokenizers",
    ),
    "collective": (
        "device.count_per_node",
        "interconnect.intra_node.topology",
        "interconnect.intra_node.link_bandwidth_bytes_per_s",
        "interconnect.intra_node.link_latency_s",
        "interconnect.inter_node.link_bandwidth_bytes_per_s",
        "interconnect.inter_node.link_latency_s",
    ),
    "kv_transfer": (
        "interconnect.inter_node.link_bandwidth_bytes_per_s",
        "interconnect.inter_node.link_latency_s",
        "interconnect.router_relay_s",
    ),
}


@dataclass(frozen=True, slots=True)
class Contribution:
    """One spec field under a quantity: what it holds and where it came from."""

    path: str
    value: Any
    derate: float | None
    effective: Any
    supplied_by: tuple[str, ...]

    def __str__(self) -> str:
        line = f"{self.path} = {self.value!r}"
        if self.derate is not None:
            line += f" x {self.derate!r} = {self.effective!r}"
        if self.supplied_by:
            line += f"  [{', '.join(self.supplied_by)}]"
        return line


@dataclass(frozen=True, slots=True)
class Basis:
    """What a quantity is built out of, and the spec that is stated in."""

    term: str
    digest: str
    contributions: tuple[Contribution, ...]
    #: Fields the spec holds that contribute no rows, printed so as not to
    #: read as fields left out.
    empty: tuple[Contribution, ...] = ()

    def __str__(self) -> str:
        return "\n".join(
            [f"{self.term}, from spec {self.digest}"]
            + [f"  {row}" for row in self.contributions + self.empty]
        )


def _supplied(origin: Merge | None, *labels: str) -> tuple[str, ...]:
    if origin is None:
        return ()
    for label in labels:
        if label in origin.sources:
            return origin.sources[label]
    return ()


def _row(
    path: str, value: Any, derate: float | None, supplied: tuple[str, ...]
) -> Contribution:
    effective = value
    if derate is not None and isinstance(value, (int, float)):
        effective = value * derate
    return Contribution(path, value, derate, effective, supplied)


def _rows(
    spec: MachineSpec, path: str, tp_width: int | None, origin: Merge | None
) -> list[Contribution]:
    field = BY_PATH[path]
    value = spec.values[path]
    derate = spec.values.get(f"{field.block}.derate") if field.peak else None
    if field.kind is Kind.WIDTH_TABLE:
        name = path.rsplit(".", 1)[-1]
        widths = (tp_width,) if tp_width is not None else tuple(sorted(value))
        return [
            _row(
                f"{path}[{width}]",
                spec.runtime_constant(name, width),
                None,
                _supplied(origin, f"{path}[{width}]", path),
            )
            for width in widths
        ]
    if field.kind is Kind.TOKENIZERS:
        rows = []
        for entry in spec.tokenizers.entries:
            label = entry_label(entry.key.id, entry.key.backend.value)
            for rate in ENTRY_FIELDS:
                if rate.kind is not Kind.QUANTITY:
                    continue
                rows.append(
                    _row(
                        f"{label}.{rate.path}",
                        getattr(entry, rate.path),
                        entry.derate if rate.peak else None,
                        _supplied(origin, label, path),
                    )
                )
        return rows
    return [_row(path, value, derate, _supplied(origin, path))]


def explain(
    spec: MachineSpec,
    term: str,
    *,
    tp_width: int | None = None,
    origin: Merge | None = None,
) -> Basis:
    """The spec fields a quantity is built from, each with its value and source."""
    paths = QUANTITIES.get(term)
    if paths is None:
        paths = tuple(
            field.path
            for field in SCHEMA
            if field.path == term or field.path.startswith(f"{term}.")
        )
    if not paths:
        raise SpecRefusal(
            Rule.ADDRESSING,
            f"`{term}` is not a field, a block of fields or a quantity this "
            "schema knows",
            "ask again by the whole dotted path of a field or a block of fields, "
            f"or by one of the quantities {sorted(QUANTITIES)}",
        )
    absent = [path for path in paths if path not in spec.values]
    missing = tuple(path for path in absent if BY_PATH[path].required)
    if missing:
        refuse_absent_fields(missing)
    if len(absent) == len(paths):
        refuse_absent_field(paths[0], False)
    contributions: list[Contribution] = []
    empty: list[Contribution] = []
    for path in paths:
        if path in spec.values:
            rows = _rows(spec, path, tp_width, origin)
            contributions += rows
            if not rows:
                held = spec.values[path]
                empty.append(_row(path, held, None, _supplied(origin, path)))
    return Basis(term, spec.digest(), tuple(contributions), tuple(empty))
