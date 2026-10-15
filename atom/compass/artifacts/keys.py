# SPDX-License-Identifier: MIT
"""What names an artifact: the artifact keys, as tuples that cannot be paths.

The table below is the whole of artifact identity. An entry is addressed by the
values that decide what it contains and by nothing else -- not by where someone
put the file, and not by a name a script composed. `price_list` is the row that
earns the rule: it is keyed by **(model, parallel layout, source-root
digest)**, and a bare or merged price file states no layout, so it silently
prices nothing while looking layered.

The parallel layout (`LAYOUT`) is the four rank widths and the three settings
that change kernels and groups at a fixed rank count. A `price_list`,
`region_terms` or `memory_readings` entry answers only at the layout it was
keyed at: a TP2 price list does not answer for `-tp 2 -dp 2`. A width is a
whole number of at least one and a switch is a `bool`. A `bool` is refused in
every other field, because `True == 1` would make two keys equal whose
canonical texts, and so directories, differ.

Two refusals follow from that, and between them they are why this module
exists rather than a dict of strings.

**A field the kind does not have is refused with the kind's whole tuple.** The
most likely mistake is reaching for `path=`, and the message that helps is not
"unknown field" but the whole tuple a `price_list` is keyed by.

**A value that is a path is refused where it sits.** Deterministically, without
asking the filesystem: an `os.PathLike`, an absolute path, anything with a
`..` segment, or anything whose last segment ends in a data-file suffix. A
model name like `Qwen/Qwen3-32B` contains a separator and is not a path, so the
separator alone cannot be the test.

A key renders to a directory name, and the rendering is deliberately lossy --
it is a label plus a digest of the canonical key, and the entry restates its
key inside itself. The path is a place, never the authority on what lives
there.
"""

import enum
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .naming import AXES
from .rules import ArtifactRefusal, Rule


class Kind(enum.Enum):
    """The artifacts this store holds. There are no others."""

    MACHINE_SPEC = "machine_spec"
    SHAPE_POPULATION = "shape_population"
    OP_GRAPH = "op_graph"
    PRICE_LIST = "price_list"
    REGION_TERMS = "region_terms"
    MEMORY_READINGS = "memory_readings"
    COVERAGE_HULL = "coverage_hull"
    STEP_GRID_BUNDLE = "step_grid_bundle"

    def __str__(self) -> str:
        return self.value


#: The layout switches: a `bool` each, and no other field takes one.
SWITCHES = ("enable_expert_parallel", "enable_dp_attention")
#: The parallel layout a key states: the rank widths, then what changes kernels
#: and groups at a fixed rank count.
LAYOUT = (*AXES, *SWITCHES, "decode_context_parallel_size")

#: What keys each artifact, in the order a key reads.
KEY_FIELDS: Mapping[Kind, tuple[str, ...]] = {
    Kind.MACHINE_SPEC: ("device", "software_stack"),
    Kind.SHAPE_POPULATION: ("model", "workload", "engine_config"),
    Kind.OP_GRAPH: ("structure",),
    Kind.PRICE_LIST: ("model", *LAYOUT, "source_root"),
    Kind.REGION_TERMS: ("model", *LAYOUT),
    Kind.MEMORY_READINGS: (
        "model",
        *LAYOUT,
        "utilization",
        "max_num_seqs",
        "max_model_len",
        "kv_dtype",
        "block_size",
    ),
    Kind.COVERAGE_HULL: ("price_list",),
    # `engine_config` names the one the grid was planned for; `grid` names
    # the grid definition with its margin.
    Kind.STEP_GRID_BUNDLE: (
        "model",
        "device",
        "parallel",
        "engine_config",
        "grid",
        "source_root",
    ),
}

#: The incident each kind's refusal quotes, where there is one to quote.
WHY: Mapping[Kind, str] = {
    Kind.PRICE_LIST: (
        "a bare or merged price file states no width and silently prices nothing"
    ),
    Kind.OP_GRAPH: "an op_graph is keyed by structure, never by shape",
    Kind.COVERAGE_HULL: (
        "a hull is keyed by the digest of the price list it was built from"
    ),
}

#: Suffixes that make a string a file name rather than a value.
DATA_SUFFIXES = frozenset(
    {".json", ".jsonl", ".yaml", ".yml", ".csv", ".txt", ".tar", ".pt", ".parquet"}
)
_SLUG = re.compile(r"[^0-9A-Za-z]+")


def _tuple_text(kind: Kind) -> str:
    return "(" + ", ".join(KEY_FIELDS[kind]) + ")"


def _remedy(kind: Kind) -> str:
    why = WHY.get(kind)
    keyed = f"a {kind} is keyed by {_tuple_text(kind)}"
    return f"{keyed}; {why}" if why else keyed


def _looks_like_a_path(value: object) -> bool:
    if isinstance(value, os.PathLike):
        return True
    if not isinstance(value, str):
        return False
    if os.path.isabs(value) or value.startswith("~"):
        return True
    parts = value.replace("\\", "/").split("/")
    if any(part in ("..", ".") for part in parts):
        return True
    return os.path.splitext(parts[-1])[1].lower() in DATA_SUFFIXES


@dataclass(frozen=True, slots=True)
class Key:
    """One artifact's identity: its kind and the values that key it."""

    kind: Kind
    fields: tuple[tuple[str, object], ...]

    @classmethod
    def of(cls, kind: Kind, **values: object) -> "Key":
        """A key, or a refusal naming the tuple the kind is keyed by."""
        if not isinstance(kind, Kind):
            raise ArtifactRefusal(
                Rule.KEY_IS_A_TUPLE,
                f"{kind!r} is not a Kind member, and only a Kind names an artifact",
                "pass one of " + ", ".join(f"Kind.{k.name}" for k in Kind),
            )
        declared = KEY_FIELDS[kind]
        unknown = sorted(set(values) - set(declared))
        if unknown:
            raise ArtifactRefusal(
                Rule.KEY_IS_A_TUPLE,
                f"{kind} has no key field {', '.join(repr(u) for u in unknown)}",
                _remedy(kind),
            )
        missing = [name for name in declared if name not in values]
        if missing:
            raise ArtifactRefusal(
                Rule.KEY_IS_A_TUPLE,
                f"this {kind} key states no {', '.join(missing)}",
                _remedy(kind),
            )
        fields = []
        for name in declared:
            value = values[name]
            if _looks_like_a_path(value):
                raise ArtifactRefusal(
                    Rule.KEY_IS_A_TUPLE,
                    f"`{kind}.{name}` holds {str(value)!r}, which is a path",
                    _remedy(kind) + "; write the value itself there, not the "
                    "path of a file holding it -- the file can change after the "
                    "key is written, and the key would then no longer say what "
                    "the entry holds",
                )
            if name in SWITCHES:
                ok, want = isinstance(value, bool), "write True or False there"
            elif name in LAYOUT:
                ok = type(value) is int and value >= 1
                want = "write a width of at least one there"
            else:
                ok = isinstance(value, (str, int, float)) and not isinstance(
                    value, bool
                )
                ok = ok and not (isinstance(value, str) and not value.strip())
                want = "write a non-empty string or a number there"
            if not ok:
                raise ArtifactRefusal(
                    Rule.KEY_IS_A_TUPLE,
                    f"`{kind}.{name}` holds {value!r}, which a key cannot compare by",
                    want,
                )
            fields.append((name, value))
        return cls(kind, tuple(fields))

    @property
    def values(self) -> dict[str, object]:
        """The key's fields as a mapping, in declared order."""
        return dict(self.fields)

    def canonical(self) -> str:
        """The key as one line of text, which is what is digested and compared."""
        return json.dumps(
            {"kind": self.kind.value, "key": self.values},
            sort_keys=True,
            separators=(",", ":"),
        )

    def token(self) -> str:
        """A short digest of the canonical key: the part of a directory name that
        is unambiguous."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:12]

    def dirname(self) -> str:
        """Where this key's entry lives, relative to its kind's directory.

        A readable label for a person reading `ls`, and a digest so that two
        keys can never land on one directory. The label is lossy on purpose;
        the entry restates its key, so the name is never asked to be the
        authority.
        """
        label = "-".join(
            _SLUG.sub("-", str(value)).strip("-") for _, value in self.fields
        )
        return f"{label[:64].strip('-')}-{self.token()}" if label else self.token()

    def __str__(self) -> str:
        stated = ", ".join(f"{name}={value!r}" for name, value in self.fields)
        return f"{self.kind}({stated})"


def refuse_not_a_key(given: object) -> None:
    """Decline something used where a key belongs, saying what a key is."""
    raise ArtifactRefusal(
        Rule.KEY_IS_A_TUPLE,
        f"{given!r} is not a key",
        "address an entry by Key.of(kind, ...); an API that accepts a path "
        "where a key belongs reproduces the price file that stated no width",
    )
