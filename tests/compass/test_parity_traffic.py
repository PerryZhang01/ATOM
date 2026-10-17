# SPDX-License-Identifier: MIT
"""The parity workload's request bodies."""

from atom.compass.parity.traffic import bodies


def test_long_gives_request_0_alone_that_many_output_tokens():
    assert [b["max_tokens"] for b in bodies("m", 3)] == [16, 32, 48]
    assert [b["max_tokens"] for b in bodies("m", 3, long=10000)] == [10000, 32, 48]
