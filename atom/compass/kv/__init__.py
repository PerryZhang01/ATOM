# SPDX-License-Identifier: MIT
"""Simulated KV transfer: the engine's connector contract, on the LP clock.

Three pieces, split by what each one may depend on.

`transfer` is arithmetic -- a latency, a derated bandwidth and an exact byte
count -- and imports nothing from the engine itself. Importing it through this
file is not that bare import, though: this file loads `connector` as well, and
with it the engine's connector interface and its factory, and what those pull
in. No tensor library and no device runtime among them, so a price can still be
computed where no driver exists.

`handoff` is the parameter blob a producing deployment hands back for the
router to relay, whose field set belongs to the backend it stands in for and
whose endpoint fields belong to a peer that does not exist. It imports nothing
at all.

`connector` implements the engine's connector interface over that price and
that blob, and is what the connector factory builds under the name `compass`.
It times transfers on the LP clock of the runtime installed in the engine.
"""

from atom.compass.kv.connector import (
    TRANSFER_KEY,
    WRITE_REQ_KEY,
    SimulatedKVConnector,
    SimulatedKVConnectorScheduler,
    UnboundSeam,
)
from atom.compass.kv.transfer import Scope, TransferModel

__all__ = [
    "TRANSFER_KEY",
    "WRITE_REQ_KEY",
    "Scope",
    "SimulatedKVConnector",
    "SimulatedKVConnectorScheduler",
    "TransferModel",
    "UnboundSeam",
]
