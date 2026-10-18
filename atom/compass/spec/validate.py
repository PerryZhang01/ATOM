# SPDX-License-Identifier: MIT
"""What a spec document is checked for, every reason reported at once.

Reading a document stops at the first thing wrong with it; a check whose reader
is filling in a form reports the whole list, in the schema's order, so a missing
field or a peak without its derate is named beside every other one.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .fields import SCHEMA, check
from .machine import MachineSpec, _stated, _survey
from .rules import SpecRefusal


@dataclass(frozen=True, slots=True)
class Validation:
    """Everything wrong with a spec, rather than the first thing wrong with it."""

    refusals: tuple[SpecRefusal, ...]

    @property
    def ok(self) -> bool:
        return not self.refusals

    def __str__(self) -> str:
        return "\n".join(
            [f"{'ok' if self.ok else 'refused'}: {len(self.refusals)} refusal(s)"]
            + [f"  {refusal}" for refusal in self.refusals]
        )


def validate(document: Mapping) -> Validation:
    """Check a document and report every refusal."""
    found: dict = {}
    refusals = list(_survey(document, "", found))
    for field in SCHEMA:
        try:
            if _stated(field, found):
                check(field, found[field.path], field.path)
        except SpecRefusal as refusal:
            refusals.append(refusal)
    if not refusals:
        try:
            MachineSpec.from_mapping(document)
        except SpecRefusal as refusal:
            refusals.append(refusal)
    return Validation(tuple(refusals))
