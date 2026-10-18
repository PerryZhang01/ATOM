# SPDX-License-Identifier: MIT
"""A later PP stage's launch runs `W_recv`, posts its receive, and runs the
rest of `W_launch` once the metadata is in: metadata that arrives after the
post ends the launch `W_launch - W_recv` after it, in the runtime and in the
gate."""

import math

import pytest
from test_dp_step_max import prefill
from test_pp_gate_holds import FREE, _step, _times

from atom.compass.runner.overrides import HostDeviceRecurrence
from atom.compass.runner.pp_send import CarrierModel

#: 1 s of prepare, a 3 s launch whose first 2 s run before the receive.
LAW = {"E": 0, "W_pre": 1, "W_launch": 3, "W_post": 0, "W_recv": 2}


@pytest.mark.parametrize("meta, exit", [(-math.inf, 4), (3, 4), (5, 6), (9, 10)])
def test_a_later_stages_launch_ends_its_rest_after_the_metadata(meta, exit):
    """Posted at 3 s; metadata in at the post or before leaves the launch at
    4 s, metadata in at 5 or 9 s ends it 1 s after."""
    law = {g: (v, 0, 0) for g, v in LAW.items()}
    rec = HostDeviceRecurrence({"prefill": law}, (), lambda: 0.0, (1, 2))
    batch = prefill(8)
    batch.compass_pp_host = (0.0, meta)
    assert rec.charge(batch, 0.0, 0.0) == pytest.approx(exit)


@pytest.mark.parametrize("pp", [2, 4])
def test_the_gate_ends_a_later_stages_launch_its_rest_after_the_metadata(pp):
    """`_step`'s 3 ms launch with a 2 ms `W_recv` posts at 3 ms; the
    metadata, 1 ms a part on gloo, arrives at 5 ms, and the launch ends at 6."""
    slow = CarrierModel(0, 0.0, 0.0, 0.0, 1e-3)
    [t] = _times([_step(0, 0, 0, recv=2)], (1, pp), [(0, 0)], gloo=slow, rccl=FREE)
    assert t[3] == pytest.approx(6)
