# SPDX-License-Identifier: MIT
"""Checks over the simulated path, run in CI.

The static checks read source files; `determinism` compares the step tables
of two runs. Each returns an exit code beside a report, and none of them runs
inside a simulated run. Import the module you need; this package re-exports
nothing.
"""
