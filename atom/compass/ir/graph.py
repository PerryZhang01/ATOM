# SPDX-License-Identifier: MIT
"""A recorded region, and the statement of where that record is valid."""

import abc
from dataclasses import dataclass

from .nodes import Region


class Applicability(abc.ABC):
    """Where a recorded graph is valid: a discrete key plus the trace's guards.

    Abstract, so a graph that claims to hold everywhere cannot be built.
    """

    __slots__ = ()

    @abc.abstractmethod
    def describe(self) -> str:
        """Where this graph is valid, in one line."""


@dataclass(frozen=True, slots=True)
class Graph:
    """One recorded region together with where it is valid."""

    applicability: Applicability
    region: Region

    def __post_init__(self) -> None:
        if not isinstance(self.applicability, Applicability):
            raise TypeError(
                "a graph states where its record is valid; expected an "
                f"Applicability, got {type(self.applicability).__name__}"
            )
        if self.region.free_indices:
            raise ValueError(
                f"nothing binds {sorted(self.region.free_indices)}, named by a "
                "context key in this graph"
            )
