# SPDX-License-Identifier: MIT
"""The one naming function, and the coordinates a per-rank file is named by.

Two incidents, one cause. A rank-coordinate function that reported only
`{"tp": rank}` under TP=2 x DP=2 made all four ranks resolve to one
`graph.tp0.json` and write it in turn -- three quarters of the evidence
discarded and the survivor looking complete at 807 operators with no error.
And at TP=2 a calibration wrote `steps.tp0.jsonl` / `steps.tp1.jsonl` while the
run looked for `steps.jsonl`; both workers died on a bare `FileNotFoundError`
and the manager reported a SHUTDOWN signal from a DP rank in a run with no DP.

Three properties here answer them, and the third is a departure from how the
incident's own code behaved.

**A coordinate is only valid against a topology.** `RankCoords` holds the
widths it was taken in, so a coordinate that names one axis of a four-axis
topology cannot be constructed at all. The four-ranks-one-file incident needs
`RankCoords` to be constructible from `{"tp": rank}` alone, and it is not.

**Every axis appears in every name, with its width.** A suffix reads
`.dp0of2.pp0of1.pcp0of1.tp1of2`, so the name states which rank of which
topology wrote the file without a reader having to know the run. The widths
are there because the coordinates alone are not enough: rank zero of a width-1
run and rank zero of a width-2 run share every coordinate, and a name that
stated only those would let one answer for the other.

**The suffix is applied at width one too.** *"A single rank cannot expose
this, because at width one no suffix is applied"* describes the code that
failed, not a requirement. Suffixing unconditionally costs a longer name and
buys two things: the read side that drops the coordinates asks for a name no
correct writer has produced at any width; and a file can be read back into the
topology that made it, so an entry can refuse a neighbour from another run
rather than answer from it.

The parse-back exists for the same reason. A file sitting in an entry whose
name this function did not produce is refused by name rather than ignored,
because an unrecognised neighbour is exactly what the width-2 read was about to
answer from.
"""

import itertools
import re
from dataclasses import dataclass

from .rules import ArtifactRefusal, Rule

#: The parallel axes, in the order ATOM's own rank arithmetic reshapes them
#: (`all_ranks.reshape(-1, dp, pp, pcp, tp)`).
AXES = ("dp", "pp", "pcp", "tp")
_AXIS = re.compile(r"^([a-z]+)([0-9]+)of([0-9]+)$")


@dataclass(frozen=True, slots=True)
class Topology:
    """The widths a per-rank artifact was produced at, every axis stated."""

    dp: int = 1
    pp: int = 1
    pcp: int = 1
    tp: int = 1

    def __post_init__(self) -> None:
        for axis in AXES:
            width = getattr(self, axis)
            if isinstance(width, bool) or not isinstance(width, int) or width < 1:
                raise ArtifactRefusal(
                    Rule.ONE_NAMING_FUNCTION,
                    f"topology axis `{axis}` is {width!r}",
                    "every axis has a width of at least one, stated even when "
                    "it is one, so a name says which topology produced it",
                )

    @property
    def widths(self) -> dict[str, int]:
        return {axis: getattr(self, axis) for axis in AXES}

    @property
    def rank_count(self) -> int:
        """How many ranks the topology holds, across every axis.

        Deliberately **not** called `width`: at `-tp 2 -dp 2` the
        tensor-parallel width is 2 and the rank count 4, which is why the
        artifact keys state every axis instead of one scalar.
        """
        total = 1
        for axis in AXES:
            total *= getattr(self, axis)
        return total

    @property
    def text(self) -> str:
        return ".".join(f"{axis}{getattr(self, axis)}" for axis in AXES)

    def ranks(self) -> tuple["RankCoords", ...]:
        """Every rank of this topology, in a fixed order."""
        spans = [range(getattr(self, axis)) for axis in AXES]
        return tuple(
            RankCoords(self, **dict(zip(AXES, point)))
            for point in itertools.product(*spans)
        )

    @classmethod
    def from_mapping(cls, widths: object) -> "Topology":
        if not isinstance(widths, dict) or set(widths) != set(AXES):
            raise ArtifactRefusal(
                Rule.ONE_NAMING_FUNCTION,
                f"{widths!r} does not state every axis",
                f"a topology states all of {', '.join(AXES)}",
            )
        return cls(**{axis: widths[axis] for axis in AXES})


@dataclass(frozen=True, slots=True)
class RankCoords:
    """One rank's position, and the topology the position is meaningful in."""

    topology: Topology
    dp: int = 0
    pp: int = 0
    pcp: int = 0
    tp: int = 0

    def __post_init__(self) -> None:
        for axis in AXES:
            coord = getattr(self, axis)
            width = getattr(self.topology, axis)
            if isinstance(coord, bool) or not isinstance(coord, int):
                raise ArtifactRefusal(
                    Rule.ONE_NAMING_FUNCTION,
                    f"coordinate `{axis}` is {coord!r}",
                    "a coordinate is a whole number",
                )
            if not 0 <= coord < width:
                raise ArtifactRefusal(
                    Rule.ONE_NAMING_FUNCTION,
                    f"coordinate `{axis}{coord}` is outside this topology, "
                    f"which is {self.topology.text}",
                    "take the coordinate from the same topology the artifact "
                    "is being written at; a coordinate that names fewer axes "
                    "than the run has is how four ranks resolved to one file",
                )

    @property
    def suffix(self) -> str:
        """The coordinates as they appear in a file name, every axis stated."""
        return "".join(
            f".{axis}{getattr(self, axis)}of{getattr(self.topology, axis)}"
            for axis in AXES
        )


def member_name(stem: str, coords: RankCoords, extension: str) -> str:
    """The one name a per-rank file has, on the write side and the read side."""
    for part, what in ((stem, "stem"), (extension, "extension")):
        if (
            not isinstance(part, str)
            or not part
            or any(character in part for character in "./\\ ")
        ):
            raise ArtifactRefusal(
                Rule.ONE_NAMING_FUNCTION,
                f"{what} {part!r} cannot appear in a name",
                "a stem and an extension are non-empty and carry no separator, "
                "so a name can be read back into the coordinates that made it",
            )
    if not isinstance(coords, RankCoords):
        raise ArtifactRefusal(
            Rule.ONE_NAMING_FUNCTION,
            f"{coords!r} is not a rank coordinate",
            "name a file through RankCoords, which carries the widths of every "
            "axis; a coordinate without them cannot tell four ranks apart",
        )
    return f"{stem}{coords.suffix}.{extension}"


def read_back(
    name: str, topology: Topology | None = None
) -> tuple[str, RankCoords, str]:
    """The stem, coordinates and extension a name was made from, or a refusal.

    The name states the width of every axis as well as the coordinate, so the
    topology is derived from the name rather than assumed. `topology`, when
    given, is cross-checked against the derived one: a file whose own name
    disagrees with the entry it sits in is refused rather than read.

    Used on every file found in an entry, so a neighbour this function did not
    write is declined by name instead of being available to answer.
    """
    parts = name.split(".")
    shape = ".".join(f"{axis}NofW" for axis in AXES)
    if len(parts) != len(AXES) + 2:
        raise ArtifactRefusal(
            Rule.ONE_NAMING_FUNCTION,
            f"`{name}` was not written by the naming function",
            f"a per-rank file reads <stem>.{shape}.<extension>; a file named "
            "otherwise states no rank, and a read that answered from it would "
            "be answering for the wrong one",
        )
    stem, *axes, extension = parts
    coords, widths = {}, {}
    for axis, part in zip(AXES, axes):
        matched = _AXIS.match(part)
        if matched is None or matched.group(1) != axis:
            raise ArtifactRefusal(
                Rule.ONE_NAMING_FUNCTION,
                f"`{name}` states {part!r} where axis `{axis}` belongs",
                f"a per-rank file reads <stem>.{shape}.<extension>",
            )
        coords[axis] = int(matched.group(2))
        widths[axis] = int(matched.group(3))
    derived = Topology(**widths)
    if topology is not None and derived != topology:
        raise ArtifactRefusal(
            Rule.ONE_NAMING_FUNCTION,
            f"`{name}` was written at {derived.text} and sits in an entry "
            f"published at {topology.text}",
            "a file states the topology it came from, so a rank of one run "
            "cannot answer for a rank of another that happens to share a "
            "coordinate",
        )
    return stem, RankCoords(derived, **coords), extension
