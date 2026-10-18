# SPDX-License-Identifier: MIT
"""--spec-decode-acceptance-rates: a per-position list reaches the forced path.

The list is parsed by `EngineArgs`, validated by `SpeculativeConfig`, and handed
unchanged to `RejectionSampler`, whose forced-acceptance kernel then accepts
draft position i with the given marginal probability.
"""

import argparse
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

import atom.config
from atom.model_engine.arg_utils import EngineArgs
from atom.model_ops.rejection_sampler import (
    RejectionSampler,
    _get_synthetic_cond_rates,
)

RATES = [0.92, 0.71, 0.48, 0.19]


def speculative_config(monkeypatch, *argv):
    monkeypatch.setattr(
        atom.config, "get_hf_config", lambda *a, **k: PretrainedConfig()
    )
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--method", "mtp", "--num-speculative-tokens", "4", *argv]
    )
    return EngineArgs.from_cli_args(args)._get_engine_kwargs()["speculative_config"]


def rates_flag(rates):
    return ["--spec-decode-acceptance-rates", ",".join(map(str, rates))]


def test_the_list_reaches_the_sampler_unchanged(monkeypatch):
    spec = speculative_config(monkeypatch, *rates_flag(RATES))
    assert spec.synthetic_acceptance_rates == RATES
    # A list is a curve of its own, not the min-variance schedule of its mean.
    assert spec.synthetic_acceptance_length is None
    sampler = RejectionSampler(spec.synthetic_acceptance_rates)
    assert sampler.synthetic_acceptance_rates == tuple(RATES)
    cond = _get_synthetic_cond_rates(
        sampler.synthetic_acceptance_rates, torch.device("cpu")
    )
    expected = [RATES[0]] + [b / a for a, b in pairwise(RATES)]
    assert cond.tolist() == pytest.approx(expected)


@pytest.mark.parametrize(
    "argv, message",
    [
        (rates_flag([1.2, 0.5, 0.1, 0.0]), r"must be in \[0, 1\]"),
        (rates_flag([0.9, 0.5, -0.1, -0.2]), r"must be in \[0, 1\]"),
        (rates_flag([0.5, 0.6, 0.1, 0.0]), "must be non-increasing"),
        (rates_flag([0.9, 0.5]), "needs 4 rates, one per speculative position"),
        (rates_flag([0.9, 0.5, 0.2, 0.1, 0.0]), "needs 4 rates"),
        (
            rates_flag(RATES) + ["--spec-decode-acceptance-length", "2.0"],
            "describe the same curve",
        ),
        (
            rates_flag(RATES) + ["--spec-decode-acceptance-rate", "0.5"],
            "describe the same curve",
        ),
    ],
)
def test_each_invalid_list_is_refused_by_name(monkeypatch, argv, message):
    with pytest.raises(ValueError, match="--spec-decode-acceptance-rates") as err:
        speculative_config(monkeypatch, *argv)
    assert err.match(message)


def test_the_scalars_still_resolve_to_the_min_variance_schedule(monkeypatch):
    spec = speculative_config(monkeypatch, "--spec-decode-acceptance-length", "2.5")
    assert spec.synthetic_acceptance_rates == [1.0, 0.5, 0.0, 0.0]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the forced-acceptance kernel is Triton"
)
def test_a_forced_run_accepts_each_position_at_its_rate(monkeypatch):
    spec = speculative_config(monkeypatch, *rates_flag(RATES))
    sampler = RejectionSampler(spec.synthetic_acceptance_rates)
    k, batch, steps, vocab = len(RATES), 512, 20, 8
    # Each request: one anchor row followed by its k drafts.
    rows = torch.arange(batch, device="cuda") * (k + 1)
    metadata = SimpleNamespace(
        input_ids=torch.zeros(batch * (k + 1), dtype=torch.int32, device="cuda"),
        target_logits_indices=(rows[:, None] + torch.arange(k, device="cuda"))
        .flatten()
        .contiguous(),
        num_spec_steps=k,
        cu_num_draft_tokens=torch.arange(
            k, k * batch + 1, k, dtype=torch.int32, device="cuda"
        ),
        bonus_logits_indices=rows + k,
    )
    logits = torch.zeros(batch * k, vocab, device="cuda")
    bonus = torch.ones(batch, 1, dtype=torch.int32, device="cuda")
    accepted = torch.cat(
        [sampler(metadata, logits, bonus)[1][0] for _ in range(steps)]
    ).cpu()
    measured = [(accepted > i).double().mean().item() for i in range(k)]
    # batch * steps draws: a rate's standard error is at most 0.005.
    assert measured == pytest.approx(RATES, abs=0.02), measured
