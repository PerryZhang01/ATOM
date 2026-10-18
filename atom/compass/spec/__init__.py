# SPDX-License-Identifier: MIT
"""The machine specification: what Compass is told about the machine it simulates.

Compute, memory, bandwidth and interconnect are configured, not read from a
device runtime. The spec describes the machine; ATOM's config describes the
deployment, and the closed schema refuses any key it does not declare. An
engine start's memory readings are not the spec's: `compass measure memory`
publishes them per configuration as a `memory_readings` entry.

A per-machine script (`scripts/compass/mi3*_spec.py`) authors a spec with the
verbs here: `tokenizer_sweep` and `tokenizer_fragment` probe the tokenizer,
`merge` combines fragments and refuses ones that describe two hosts, `validate`
reports every refusal of the merged document, and `explain` takes a field or
block back to its values and their origins. A run reads the result with
`MachineSpec`. `across_ranks` is `compass measure memory`'s spread check.

Nothing here imports a tensor library, a device runtime or the engine.
"""

from .explain import explain
from .machine import MachineSpec
from .merge import Fragment, merge
from .probes import tokenizer_fragment, tokenizer_sweep
from .ranks import across_ranks
from .rules import SpecRefusal
from .tokenizers import Backend
from .validate import validate

__all__ = [
    "Backend",
    "Fragment",
    "MachineSpec",
    "SpecRefusal",
    "across_ranks",
    "explain",
    "merge",
    "tokenizer_fragment",
    "tokenizer_sweep",
    "validate",
]
