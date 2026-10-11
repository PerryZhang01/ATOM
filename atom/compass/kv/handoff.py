# SPDX-License-Identifier: MIT
"""What one deployment hands the next, and what a simulated peer may say.

A disaggregated prefill ends with a parameter blob on the producer's response.
The HTTP router does not interpret it -- it copies it onto the decode request
verbatim -- but it does refuse a prefill response that carries none, so the
blob is not optional decoration: without it there is no decode leg at all.

**The field set is not a choice.** It is re-derived from Mooncake's own
`request_finished`, which is the shape the router and the consumer were built
against, and the test beside this module parses that source rather than
trusting a copy of the list, so a field added there turns this red instead of
leaving a simulated deployment quietly one field short.

**Four of the fields name an endpoint, and a simulated deployment has none.**
The real backends fill them from the host they run on -- the machine's own
address and a port they actually bound -- which is the one answer a simulated
producer must not give: that blob names the machine running the simulation, so
anything that read it would open a connection to a real host that has no KV to
serve and no idea it was named. The answers here are the other way round --
well-formed, of the type the field carries, and impossible to mistake for a
listener:

| field | value | why it cannot be an endpoint |
|---|---|---|
| `remote_host` | `compass-simulated.invalid` | `.invalid` is reserved by RFC 2606 precisely so that it can never be delegated, so the name resolves nowhere, on any network, forever -- and it says in the name what it is |
| `remote_port` | `0` | the sockets API's own "no port": nothing listens on it, and a connection to it is refused at once rather than reaching whatever else is up on that host |
| `remote_handshake_port` | `0` | the same, for the same reason: there is no handshake because there are no two processes to hold one |
| `remote_engine_id` | `compass-simulated` | an identity, not an address. The real backends carry a constant here too -- the literal string `"None"` in both -- so nothing downstream derives a route from it |

The rest of the blob is the request's own truth and is taken from the sequence:
the block table it occupies, the parallel widths it was launched under, the
first sampled token the consumer resumes from, and the drafts and prefix-cache
hit that ride beside it.
"""

from __future__ import annotations

from typing import Any

#: A name reserved by RFC 2606 so that it can never resolve anywhere.
SIMULATED_HOST = "compass-simulated.invalid"

#: No socket was bound, so there is no port; this is what that is spelled as.
SIMULATED_PORT = 0

#: Who served the prefill. An identity, and deliberately not an address.
SIMULATED_ENGINE_ID = "compass-simulated"


def transfer_params(
    seq: Any, *, tp_size: int, dp_rank: int, pp_size: int, hash_block_size: int
) -> dict[str, Any]:
    """The blob the router relays for *seq*, as a simulated producer sees it.

    The widths and `hash_block_size` describe the deployment rather than the request,
    so they are passed in from the config the connector was built with, which
    has already put each through `whole_number`. They go out as given. They
    must be `int` because the router takes `dp_rank` only if it is a number,
    and otherwise **substitutes its own registry value for the prefilling
    worker**, dropping it outright only when it has none either way. So a
    rank of the wrong type is not lost loudly; it is replaced by a plausible
    one, with no error anywhere.
    """
    drafts = getattr(seq, "spec_token_ids", None)
    draft_token_ids = (
        [int(token) for token in drafts] if drafts is not None and len(drafts) else []
    )
    slot = getattr(seq, "state_slot", -1)
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": list(seq.block_table),
        "remote_swa_block_ids": [int(slot)] if slot is not None and slot >= 0 else [],
        "remote_engine_id": SIMULATED_ENGINE_ID,
        "remote_host": SIMULATED_HOST,
        "remote_port": SIMULATED_PORT,
        "remote_handshake_port": SIMULATED_PORT,
        "tp_size": tp_size,
        "dp_rank": dp_rank,
        "remote_pp_size": pp_size,
        "hash_block_size": hash_block_size,
        "transfer_id": seq.id,
        "first_token_id": seq.output_tokens[0] if seq.output_tokens else None,
        "draft_token_ids": draft_token_ids,
        "local_slot_index": slot,
        "prefix_cache_hit_tokens": getattr(seq, "prefix_cache_hit_tokens", 0),
    }


def whole_number(field: str, value: Any) -> int:
    """*value* as an `int`, or a `ValueError` naming *field* if it is not one.

    An `int` is taken as it is, and so is a float with no fractional part.
    Text is refused, "8" included. ATOM's own launch path does not deliver a
    width as text -- its CLI flags and `ATOM_DP_RANK` are parsed with `int`,
    and its `Config` raises on a string width -- so reading text would be a
    conversion that no launch needs.

    Refused: a fraction, because `int` truncates and 8.5 would go out as 8,
    naming a deployment that was never launched; a boolean, which would go
    out as a width of 1 or 0 -- a `bool`, or anything whose `dtype` has "bool"
    in its text, as numpy's and torch's booleans do, neither being a `bool`
    subclass; and anything else `int` does not take exactly. No value that is
    accepted changes on the way to the `int` returned, and that check is what
    refuses integer text: `int("8")` is 8, which is not equal to "8".

    Limits: a whole number whose dtype text has "bool", such as a 0-d array
    of a numpy union dtype with a field "is_bool", is refused as not one. An
    exception from the value, from its dtype, or from an object either of them
    returns, such as the result of `__ne__`, can escape as raised, not as this
    refusal, and an `AttributeError` from reading `dtype` counts as no dtype.
    """
    try:
        if isinstance(value, bool) or "bool" in str(getattr(value, "dtype", "")):
            raise TypeError(f"{value!r} is a bool")
        whole = int(value)
    except (TypeError, ValueError, OverflowError):
        whole = None
    if whole is None or whole != value:
        raise ValueError(
            f"{field} is {value!r}, which is not a whole number; a parallel "
            "width or rank is a count, so it is refused rather than converted"
        )
    return whole
