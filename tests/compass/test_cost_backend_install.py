# SPDX-License-Identifier: MIT
"""A runner prices its steps with the cost backend installed on it, or refuses."""

from types import SimpleNamespace

import pytest

from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.runner.overrides import (
    RunnerRefusal,
    _installed_backend,
    install_cost_backend,
)


def test_an_installed_backend_is_the_one_read_back():
    runner, backend = SimpleNamespace(), StepStubBackend()
    install_cost_backend(runner, backend)
    assert _installed_backend(runner) is backend


def test_what_is_not_a_cost_backend_is_refused_by_type():
    with pytest.raises(RunnerRefusal, match="dict is not one"):
        install_cost_backend(SimpleNamespace(), {"seconds": 1.0})


def test_a_runner_with_no_backend_refuses_and_names_what_would_supply_it():
    with pytest.raises(RunnerRefusal, match="install_cost_backend"):
        _installed_backend(SimpleNamespace())
