# SPDX-License-Identifier: MIT
"""The parity workload's request bodies."""

from atom.compass.parity.traffic import WORDS, bodies


def test_long_gives_request_0_alone_that_many_output_tokens():
    assert [b["max_tokens"] for b in bodies("m", 3)] == [16, 32, 48]
    assert [b["max_tokens"] for b in bodies("m", 3, long=10000)] == [10000, 32, 48]


def test_families_share_the_prefix_only_within_a_family():
    shared = " ".join(WORDS[:3])
    prompts = [b["prompt"] for b in bodies("m", 4, prefix_words=3, families=2)]
    assert prompts[0].startswith(f"0 {shared} 0 ")
    assert prompts[2].startswith(f"0 {shared} 2 ")
    assert prompts[1].startswith(f"1 {shared} 1 ")
    assert [b["prompt"] for b in bodies("m", 2, 3)][1].startswith(f"{shared} 1 ")


def test_no_default_prompt_repeats_a_run_of_eight_words():
    """A prompt that repeats itself is continued by copying, and a speculative
    decoder on it accepts every draft."""
    for k, body in enumerate(bodies("m", 64)):
        words = body["prompt"].split()
        runs = [tuple(words[i : i + 8]) for i in range(len(words) - 7)]
        assert len(set(runs)) == len(runs), f"request {k} repeats itself"
