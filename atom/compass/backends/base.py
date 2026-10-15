# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The interface a cost backend implements.

Two methods. `price` prices one step; `describe` says what this backend is,
in a line that goes into the run record so a result can be read a month later.

`batch_view` is a projection of the scheduled batch, prepared by the caller --
plain numbers and sequences, no engine objects. That is the whole reason this
package imports nothing from the rest of ATOM: a backend is then testable on a
machine with no accelerator and no engine, and the fields it depends on are
visible in the projection rather than reachable by attribute from anywhere in
the scheduler. The projection's shape belongs to the caller that builds it, so
it is not named here.

Whoever defines that shape inherits an obligation with it, because the
no-engine-imports property belongs to a package and not to a type. The test
that enforces it reads the files under this package and nothing else: a
projection defined here is covered automatically, and one defined anywhere
else is covered by nobody until its own package's tests assert the same thing
over its own sources. A projection that reaches an engine type for one field
is a projection that can only be built where the engine imports, which defeats
the reason it exists.

`granularity` says what the backend prices as one unit, a whole step or the
ops in it, which is a different axis from where each answer inside it came
from. An op-level backend still answers an unpriced leaf analytically; the
granularity records the unit and the provenance records what happened.
"""

from __future__ import annotations

import abc
import enum
from typing import Any

from atom.compass.backends.cost import StepCost


class Granularity(enum.Enum):
    """What a backend prices as one unit."""

    STEP = "step"
    OP = "op"


class CostBackend(abc.ABC):
    """Turns a projected batch into a duration with its decomposition."""

    @property
    @abc.abstractmethod
    def granularity(self) -> Granularity:
        """What this backend prices as one unit."""

    @abc.abstractmethod
    def price(self, batch_view: Any) -> StepCost:
        """Price one step.

        Returns a `StepCost`, which carries its breakdown by construction.
        Raises `CostRefused` when nothing can price the step: a backend asked
        about something it has no basis for refuses by name and does not
        substitute a number, however plausible one would look.
        """

    @abc.abstractmethod
    def describe(self) -> str:
        """One line naming this backend and what it is answering from."""
