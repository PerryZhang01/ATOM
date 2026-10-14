# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The memory model: the five device readings, produced without a device.

`ModelRunner.get_num_blocks` is five readings off a card and then arithmetic
over them. This package owns the five, so that an MI308X host can size the KV
pool of a card it does not have, and it deliberately owns none of the
arithmetic -- the budget formula, the 2% margin, the `min(budget, free)` clamp
and `plan_pools` are ATOM's and stay ATOM's.

Three properties are enforced here rather than described.

**A reading is its terms.** `Reading` derives its total on every read, defines
no `__int__`, and prints as a per-term table. The reason is an incident: a
summed non-KV check once read +13.8% while holding a 25% error in one of its
terms, and an object printable only as a total reproduces that by design.

**`free` is a clean box.** `total - peak_torch - non_torch`, never a reading, so
the neighbour contamination in `(total - free) - reserved` cannot enter and the
`min(budget, free)` clamp cannot bind.

**Nothing here imports a tensor library, a device runtime or the engine.** The
absence of a device read is therefore a property of the import graph rather than
a claim about the code paths, and it is checked as one.

The two graph-pool numbers are kept apart in `graph_pool`, by name and at the
call site, because they disagree by 4-19x and only one of them reserves.

**A prediction is held to its gate one term at a time.** `compare` takes a
breakdown a hardware run printed -- as input data; nothing here runs a model --
and returns a per-term table with no total in it. The aggregate exists only as
a `SummedCheck`, which cannot be built without the comparison it summarises,
because the instrument that reads one figure over a decomposition is the one
the per-term rule was written to reject.

**A block count keeps its readings.** ATOM's own arithmetic turns the five into
one integer, and an integer cannot say that most of what produced it was a
coefficient somebody wrote down. `budget.SizedKVPool` is the count carried
beside the readings it came out of; it computes nothing ATOM computes.
"""

from atom.compass.memory.budget import SUBTRACTED, SizedKVPool
from atom.compass.memory.compare import (
    DISCHARGES,
    NON_KV_TERM_GATE,
    Comparison,
    GraphPoolComparison,
    Predicted,
    Recorded,
    Shape,
    SummedCheck,
    TermComparison,
    TermRefusal,
    Verdict,
    compare,
    compare_graph_pool,
    footprint_terms,
)
from atom.compass.memory.graph_pool import (
    EAGER_SOURCE,
    PREDICTS,
    RESERVES,
    PiecewiseCapture,
    capture_token_shapes,
    piecewise_per_token_bytes,
    predicts,
    reserves,
)
from atom.compass.memory.readings import (
    DeviceReadings,
    MemoryRefusal,
    ModelTerms,
    device_readings,
)
from atom.compass.memory.terms import Basis, Reading, Term

__all__ = [
    "DISCHARGES",
    "EAGER_SOURCE",
    "NON_KV_TERM_GATE",
    "PREDICTS",
    "RESERVES",
    "SUBTRACTED",
    "Basis",
    "Comparison",
    "DeviceReadings",
    "GraphPoolComparison",
    "MemoryRefusal",
    "ModelTerms",
    "PiecewiseCapture",
    "Predicted",
    "Reading",
    "Recorded",
    "Shape",
    "SizedKVPool",
    "SummedCheck",
    "Term",
    "TermComparison",
    "TermRefusal",
    "Verdict",
    "capture_token_shapes",
    "compare",
    "compare_graph_pool",
    "device_readings",
    "footprint_terms",
    "piecewise_per_token_bytes",
    "predicts",
    "reserves",
]
