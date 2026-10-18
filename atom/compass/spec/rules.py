# SPDX-License-Identifier: MIT
"""What a refusal says: the rule that declined, what tripped it, and the fix.

The schema is closed: a document key that is not a declared field is refused
wherever it sits, so a spec cannot carry a deployment knob even when no rule
names it. `DEPLOYMENT_OWNED` only improves the message for the knobs an author
is most likely to reach for, naming where ATOM configures each instead.
"""

import enum
from typing import NoReturn


class Rule(enum.Enum):
    """The rules a spec is read against, and the one a probe is held to."""

    SEPARATION = "the spec describes the machine, not the deployment"
    ONE_MACHINE = "one spec describes one host"
    RANK_AGREEMENT = "ranks of one group measure one machine"
    DERATE = "a spec-peak number carries a derate"
    PINNED_STACK = "the constants are pinned to a software stack"
    TOKENIZER_IDENTITY = "a tokenizer is probed or it is refused"
    SHAPE = "the document has the shape the schema declares"
    TOTALITY = "a spec resolves every field the document was required to state"
    ADDRESSING = "a field is asked for by the path of the field itself"
    VERSION = "the reader knows the schema version the document was written to"
    MEASURED = "a probe reports what it read, and refuses what it could not"
    BAND = "a probed number is published with the band its probe measured"


#: Knobs the engine already owns, mapped to where it owns them.
DEPLOYMENT_OWNED = {
    "block_size": "EngineArgs.block_size",
    "cudagraph_capture_sizes": "EngineArgs.cudagraph_capture_sizes",
    "cudagraph_mode": "EngineArgs.cudagraph_mode",
    "gpu_memory_utilization": "EngineArgs.gpu_memory_utilization",
    "kv_cache_dtype": "EngineArgs.kv_cache_dtype",
    "level": "EngineArgs.level",
    "max_model_len": "EngineArgs.max_model_len",
    "max_num_seqs": "EngineArgs.max_num_seqs",
    "tensor_parallel_size": "EngineArgs.tensor_parallel_size",
}


class SpecRefusal(Exception):
    """A declined answer that names its rule and what would satisfy it."""

    def __init__(self, rule: Rule, what: str, remedy: str) -> None:
        self.rule = rule
        self.what = what
        self.remedy = remedy
        super().__init__(f"{rule.value}: {what}. {remedy}")


class FingerprintMismatch(UserWarning):
    """The tokenizer now loaded is not the one whose rates were probed."""


def refuse_unknown_key(path: str) -> NoReturn:
    """Decline a key the schema does not declare, saying why it is not one."""
    owner = DEPLOYMENT_OWNED.get(path.rsplit(".", 1)[-1])
    if owner is not None:
        raise SpecRefusal(
            Rule.SEPARATION,
            f"`{path}` states how ATOM was launched, not what the machine is",
            f"remove it; {owner} carries it and Compass reads it from there",
        )
    raise SpecRefusal(
        Rule.SEPARATION,
        f"`{path}` is not a field of this schema",
        "the schema is closed: a property of the machine is added to the field "
        "table deliberately, and anything ATOM could configure stays in ATOM's "
        "config, where Compass reads it",
    )
