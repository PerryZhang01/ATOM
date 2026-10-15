<!-- SPDX-License-Identifier: MIT -->
# Compass tools

Run from a tree root with ATOM importable (`pip install -e .`, or
`PYTHONPATH=<tree root>`).

## The step grid's planner and coverage check

`atom/compass/backends/step_grid_plan.py` plans which grid points a step grid
bundle must measure, from the shapes of recorded steps, and checks that a
bundle covers a set of steps. Needs no device, and measures and fits nothing.

```
python -m atom.compass.backends.step_grid_plan plan DEFINITION RECORD...
python -m atom.compass.backends.step_grid_plan coverage BUNDLE RECORD...
```

**RECORD** is a step record directory (`atom/compass/parity/README.md`), real
or simulated; every line of each rank's `dp<r>.jsonl` is read, and `plan`
plans for the engine every rank's `dp<r>.engine.json` states. **DEFINITION**
is the grid definition and **BUNDLE** a step grid bundle's member file, the
mapping `atom/compass/run/step_grid.py` describes; both are JSON. Each command
prints JSON, which the module's `plan` and `coverage` say how to read;
`coverage` exits 1 unless every step is covered.

## `detect_loose_refusals.py`

Its docstring says how to run it and what it reports.
