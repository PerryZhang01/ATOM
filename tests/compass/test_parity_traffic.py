# SPDX-License-Identifier: MIT
"""The parity workload's request bodies."""

from atom.compass.parity.traffic import bodies


def test_long_gives_request_0_alone_that_many_output_tokens():
    assert [b["max_tokens"] for b in bodies("m", 3)] == [16, 32, 48]
    assert [b["max_tokens"] for b in bodies("m", 3, long=10000)] == [10000, 32, 48]


def test_families_share_the_prefix_only_within_a_family():
    prompts = [b["prompt"] for b in bodies("m", 4, prefix_words=3, families=2)]
    assert prompts[0].startswith("0 parity replays each 0 ")
    assert prompts[2].startswith("0 parity replays each 2 ")
    assert prompts[1].startswith("1 parity replays each 1 ")
    assert [b["prompt"] for b in bodies("m", 2, 3)][1].startswith(
        "parity replays each 1 "
    )
