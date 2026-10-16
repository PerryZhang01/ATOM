# SPDX-License-Identifier: MIT
"""Checks over the simulated path, run in CI.

The static checks read source files; `determinism` compares the step tables
of two runs, and `scenario` holds the runs of one deployment to the acceptance
checks. None of them runs inside a simulated run. Import the module you need;
this package re-exports nothing.
"""
