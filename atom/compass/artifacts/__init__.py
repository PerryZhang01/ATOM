# SPDX-License-Identifier: MIT
"""The artifact store: where an entry lives, what made it, and when it stops answering.

This package exists for eight incidents that share one shape -- the number was
fine and the question of *which artifact answered* was not -- turned into
refusals, plus the answer to a ninth: an artifact that is still read after the
thing it was measured against has moved.

* **A key is a tuple, never a path** (`keys`). The six artifacts and what keys
  each one; a `price_list` is `(model, parallel layout, machine-spec digest,
  source-root digest)`, because a bare or merged price file states no layout
  and silently prices nothing.
* **One naming function** (`naming`). Write and read sides call it, per-rank
  files carry every axis's coordinates, and the suffix is applied at width one
  too -- so a width-2 read cannot answer from a width-1 neighbour.
* **Provenance names every executed source root** (`provenance`). ATOM's by the
  tree `git archive` would ship, aiter's by `git describe --tags --always
  --dirty`, the call the GPU gate makes on aiter's checkout. **Only the recording
  half**: the version is recorded, and what a bump *means* is left open.
* **A handed-off entry is immutable** (`store`). **The physical form is a
  directory convention** -- no index, no manifest -- and publishing is a rename
  onto a name that must not exist, so an overwrite is impossible rather than
  discouraged.
* **Invalidation is a per-artifact matrix** (`matrix`, `fingerprints`). The
  design's table is a table in the code, so a reader can hold the two side by
  side.
  Each entry records the fingerprint of *its own dependency row* -- three rows
  for a `machine_spec` -- and a load recomputes and compares cell by cell, so a
  refusal names the cell and a change in a cell the row does not depend on
  loads clean.
* **Every gate's state is in the artifact** (`gates`). *"A dead gate is worse
  than no gate."* Loading checks the state the entry recorded, not the flag,
  which is the one thing the `PRICE_KERNELS` incident could not do.
* **Resolution names its answer** (`resolution`). A step's ledger says which
  artifacts answered it and which key missed, because `incomplete: N/2570`
  states a count and not which of two stages declined.

**Not decided here**: whether an aiter bump invalidates an artifact or only
warns (filed as #168) -- while that is open the default stands and a bump
refuses; and whether a dirty source root may publish (also #168).
"""

from .fingerprints import (
    Conditions,
    Fingerprint,
    Mismatch,
    OnMismatch,
    Reading,
    StaleArtifact,
    differences,
    fingerprint,
    verify,
)
from .gates import Gate, GateState
from .keys import KEY_FIELDS, Key, Kind, machine_spec_digest, machine_spec_key
from .matrix import MATRIX, ROWS_OF, Axis, Cell, Row, axes_of, rows_for
from .naming import AXES, RankCoords, Topology, member_name, read_back
from .provenance import (
    REQUIRED_ROOTS,
    REVISION_KINDS,
    Provenance,
    SourceRoot,
    git_described_root,
    git_tree_root,
    module_root,
    roots_for,
)
from .resolution import Answer, Miss, Resolution
from .rules import ArtifactRefusal, Rule
from .store import SCHEMA_VERSION, ArtifactStore, Entry

__all__ = [
    "AXES",
    "KEY_FIELDS",
    "MATRIX",
    "REQUIRED_ROOTS",
    "REVISION_KINDS",
    "ROWS_OF",
    "SCHEMA_VERSION",
    "Answer",
    "ArtifactRefusal",
    "ArtifactStore",
    "Axis",
    "Cell",
    "Conditions",
    "Entry",
    "Fingerprint",
    "Gate",
    "GateState",
    "Key",
    "Kind",
    "Mismatch",
    "Miss",
    "OnMismatch",
    "Provenance",
    "RankCoords",
    "Reading",
    "Resolution",
    "Row",
    "Rule",
    "SourceRoot",
    "StaleArtifact",
    "Topology",
    "axes_of",
    "differences",
    "fingerprint",
    "git_described_root",
    "git_tree_root",
    "machine_spec_digest",
    "machine_spec_key",
    "member_name",
    "module_root",
    "read_back",
    "roots_for",
    "rows_for",
    "verify",
]
