# SPDX-License-Identifier: MIT
"""Tokenizer rates, keyed by the tokenizer rather than by the model.

Models share tokenizers, so the rates are stored once per tokenizer and list the
architectures that resolve to them. The key is `id` (author-chosen), `backend`
(fast and slow run an order of magnitude apart on one `tokenizer.json`),
`fingerprint` (the sha256 of the file probed) and `applies_to`.

Resolution refuses an architecture no entry names on the loaded backend, rather
than defaulting: the error would land inside time to first token and grow with
the prompt. A fingerprint mismatch only warns, since the rates are still that
tokenizer's.
"""

import enum
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

from .fields import Field, Kind, check, closed
from .rules import FingerprintMismatch, Rule, SpecRefusal


class Backend(enum.Enum):
    """Which tokenizer implementation the rates were probed on."""

    FAST = "fast"
    SLOW = "slow"


#: A tokenizer entry, closed like the rest of the schema; the derate is derived
#: from the two rates being spec-peak numbers, not declared.
ENTRY_FIELDS = closed(
    (
        Field("id", Kind.TEXT),
        Field("backend", Kind.TEXT),
        Field("vocab_size", Kind.COUNT),
        Field("fingerprint", Kind.TEXT),
        Field("applies_to", Kind.NAMES),
        Field("encode_fixed_s", Kind.QUANTITY),
        Field("encode_tokens_per_s", Kind.QUANTITY, peak=True),
        Field("decode_fixed_s", Kind.QUANTITY),
        Field("decode_tokens_per_s", Kind.QUANTITY, peak=True),
    )
)


@dataclass(frozen=True, slots=True)
class TokenizerKey:
    """What identifies a probed tokenizer, and what a merge conflicts on."""

    id: str
    backend: Backend
    fingerprint: str
    applies_to: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TokenizerEntry:
    """One probed tokenizer: its identity and the rates that were probed."""

    key: TokenizerKey
    vocab_size: int
    encode_fixed_s: float
    encode_tokens_per_s: float
    decode_fixed_s: float
    decode_tokens_per_s: float
    derate: float


@dataclass(frozen=True, slots=True)
class TokenizerTable:
    """The probed tokenizers of one host, resolved by model architecture."""

    entries: tuple[TokenizerEntry, ...]

    def resolve(
        self, architecture: str, backend: Backend, fingerprint: str | None = None
    ) -> TokenizerEntry:
        """The entry probed for this architecture, or a refusal naming it."""
        for entry in self.entries:
            if architecture in entry.key.applies_to and entry.key.backend is backend:
                if fingerprint is not None and fingerprint != entry.key.fingerprint:
                    warnings.warn(
                        f"tokenizer {entry.key.id!r} was probed against "
                        f"{entry.key.fingerprint}, but the tokenizer.json now "
                        f"loaded hashes to {fingerprint}",
                        FingerprintMismatch,
                        stacklevel=2,
                    )
                return entry
        probed = ", ".join(
            f"{entry.key.id} ({entry.key.backend.value})" for entry in self.entries
        )
        raise SpecRefusal(
            Rule.TOKENIZER_IDENTITY,
            f"no tokenizer probed for {architecture!r} on the "
            f"{backend.value} backend; this host probed {probed or 'nothing'}",
            "probe that tokenizer on this host and add an entry whose "
            "applies_to names this architecture; an unprobed one is refused "
            "rather than defaulted because the whole of the error lands inside "
            "time to first token and grows with the prompt",
        )


def _entry(raw: object, index: int) -> TokenizerEntry:
    where = f"host.tokenizers[{index}]"
    if not isinstance(raw, Mapping):
        raise SpecRefusal(
            Rule.SHAPE,
            f"`{where}` holds {raw!r}, which is not a tokenizer entry",
            "write a mapping of the entry's fields there",
        )
    declared = {field.path: field for field in ENTRY_FIELDS}
    for name in raw:
        if name not in declared:
            raise SpecRefusal(
                Rule.SEPARATION,
                f"`{where}.{name}` is not a field of a tokenizer entry",
                "a tokenizer entry carries the identity of a tokenizer and the "
                "rates probed for it, and nothing about the deployment",
            )
    values = {}
    for path, field in declared.items():
        if path not in raw:
            rule = Rule.DERATE if field.kind is Kind.DERATE else Rule.SHAPE
            raise SpecRefusal(
                rule,
                f"`{where}.{path}` is missing",
                "every field of a tokenizer entry is required, including the "
                "derate the two rates oblige",
            )
        values[path] = check(field, raw[path], f"{where}.{path}")
    try:
        backend = Backend(values["backend"])
    except ValueError:
        raise SpecRefusal(
            Rule.TOKENIZER_IDENTITY,
            f"`{where}.backend` is {values['backend']!r}",
            "name the implementation the rates were probed on: "
            + " or ".join(member.value for member in Backend),
        ) from None
    return TokenizerEntry(
        key=TokenizerKey(
            id=values["id"],
            backend=backend,
            fingerprint=values["fingerprint"],
            applies_to=values["applies_to"],
        ),
        vocab_size=values["vocab_size"],
        encode_fixed_s=values["encode_fixed_s"],
        encode_tokens_per_s=values["encode_tokens_per_s"],
        decode_fixed_s=values["decode_fixed_s"],
        decode_tokens_per_s=values["decode_tokens_per_s"],
        derate=values["derate"],
    )


def table(raw: object) -> TokenizerTable:
    """Read the tokenizer list, refusing a key that two entries both answer."""
    entries = tuple(_entry(item, index) for index, item in enumerate(raw))
    seen: dict[tuple[str, Backend], str] = {}
    for entry in entries:
        for architecture in entry.key.applies_to:
            claim = (architecture, entry.key.backend)
            if claim in seen:
                raise SpecRefusal(
                    Rule.TOKENIZER_IDENTITY,
                    f"{architecture!r} on the {entry.key.backend.value} backend "
                    f"resolves to both {seen[claim]!r} and {entry.key.id!r}",
                    "one architecture resolves to one probed tokenizer; if "
                    "two entries really describe it, they disagree and the "
                    "disagreement is the thing to settle",
                )
            seen[claim] = entry.key.id
    return TokenizerTable(entries)
