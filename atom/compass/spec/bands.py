# SPDX-License-Identifier: MIT
"""A probed number is published with the p10 and p90 its probe measured,
stated in `provenance.origins` beside its subject. Readers use the value. A
count or a text is read, not sampled, and so is a field marked `exact`."""

from collections.abc import Mapping

from .fields import SCHEMA, Kind
from .machine import MachineSpec, entry_label
from .rules import Rule, SpecRefusal
from .tokenizers import ENTRY_FIELDS

#: The kinds a probe samples.
BANDED = (Kind.QUANTITY, Kind.BYTES)


def refuse_unbanded(spec: MachineSpec) -> None:
    """Refuse to publish `spec` while a probed number states no band, naming
    each; a probed tokenizer entry states one per rate."""
    owed = {
        f.path: None
        for f in SCHEMA
        if f.kind in BANDED and not f.exact and f.path in spec.values
    }
    rates = {f.path for f in ENTRY_FIELDS if f.kind in BANDED}
    for e in spec.tokenizers.entries:
        owed[entry_label(e.key.id, e.key.backend.value)] = rates
    unbanded = []
    for label, keys in owed.items():
        found = spec.origin(label)
        low = found.get("p10")
        if found["origin"] == "probed" and (
            low is None or (set(low) if isinstance(low, Mapping) else None) != keys
        ):
            unbanded.append(f"`{label}`")
    if unbanded:
        raise SpecRefusal(
            Rule.BAND,
            f"{', '.join(unbanded)} probed with no p10 and p90",
            "write the band of the probe's records beside each in "
            "`provenance.origins`; a probed number alone reads as more certain "
            "than its probe measured it",
        )
