# SPDX-License-Identifier: MIT
"""The step grid in a run: its bundle loaded from the artifact store by key, and
checked against the resolved ATOM config before the first step.

The run file's ``step_grid`` is the store's root and the bundle's key fields:

    {"store": <root directory>, "model": ..., "device": ..., "parallel": ...,
     "engine_config": ..., "grid": ..., "source_root": ...}

A bundle prices a TP group's whole step, so it is one member at any width,
``bundle.dp0of1.pp0of1.pcp0of1.tp0of1.json``:

    {"grid": <a StepGrid mapping>, "residual": <a Residual mapping>,
     "bounds": {"max_num_seqs": n, "max_num_batched_tokens": n,
                "max_model_len": n}}

Its publisher states the readings `conditions` takes. The engine-config reading
is `keyed`, as JSON: the ladder the runner replays (none under
``enforce_eager``), the KV block size and dtype, the quantization and the TP
width. Loading refuses, naming what differs:

- a run at data-parallel width above one, since every bundle is measured at one;
- a keyed condition, the grid's graph mode, or a timing scope other than the
  whole ``ModelRunner.forward``;
- the gates, or any reading, through `ArtifactStore.load`;
- a rung of the ladder between the grid's lowest and highest measured rungs
  that was not measured;
- a configured bound above the one the grid was planned for.

A cell past the KV capacity is incomplete, so it refuses at lookup and needs no
check here. Recipe v1 reads no chunk size, so none is keyed.
"""

import json
from collections.abc import Mapping

import torch

from atom.compass.artifacts import (
    ArtifactStore,
    Axis,
    Conditions,
    GateState,
    Key,
    Kind,
    Reading,
    Row,
    Topology,
    git_tree_root,
    module_root,
)
from atom.compass.backends.step_grid import StepGrid, StepGridBackend
from atom.compass.backends.step_grid_plan import BOUNDS, SCOPE
from atom.compass.backends.step_grid_residual import Residual
from atom.compass.runner.overrides import schedulable_ladder
from atom.compass.spec.fields import PINNED

#: The bundle's one member: a group's step has a single writer.
MEMBER = ("bundle", Topology().ranks()[0], "json")


def keyed(config) -> dict:
    """The conditions a bundle is measured under, as the resolved config holds them."""
    return {
        "ladder": [] if config.enforce_eager else schedulable_ladder(config),
        "kv_cache_block_size": config.kv_cache_block_size,
        "kv_cache_dtype": config.kv_cache_dtype,
        "quantization": config.quant_config.compute_hash(),
        "tensor_parallel_size": config.tensor_parallel_size,
    }


def conditions(config, machine) -> Conditions:
    """This run's readings: the stack and device the machine spec states,
    this process's torch and ATOM tree, the served config.json and `keyed`."""
    from atom.compass.run import _file_digest

    pins = (f"{c} {machine.value(f'device.software_pinned_to.{c}')}" for c in PINNED)
    return Conditions.of(
        software_stack=" / ".join(pins),
        torch=torch.__version__,
        atom_src=Reading.of_source_root(git_tree_root("atom", module_root("atom"))),
        model=_file_digest(config.model, "config.json"),
        device=f"{machine.value('device.name')} {machine.value('device.arch')}",
        engine_config=json.dumps(keyed(config), sort_keys=True),
    )


def load(spec: Mapping, config, machine) -> tuple[StepGridBackend, dict]:
    """The backend the bundle at the run file's key prices with, and the
    bundle's key and digest for the run summary; or a refusal naming what
    differs."""
    dp = config.parallel_config.data_parallel_size
    if dp != 1:
        raise ValueError(
            "a step grid bundle is measured at data-parallel width 1, and this "
            f"run has data_parallel_size {dp}"
        )
    fields = dict(spec)
    store = ArtifactStore(fields.pop("store"))
    key = Key.of(Kind.STEP_GRID_BUNDLE, **fields)
    entry = store.read(key)
    try:
        member = json.loads(entry.read_member(*MEMBER))
        grid, residual = StepGrid(member["grid"]), Residual(member["residual"])
        bounds = {name: member["bounds"][name] for name in BOUNDS}
        row = dict(entry.fingerprints[Row.STEP_GRID_BUNDLE].cells)
        recorded = json.loads(row[Axis.ENGINE_CONFIG].value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{key} does not load: {type(error).__name__}: {error}"
        ) from error
    ours = keyed(config)
    pairs = {
        "graph mode": (grid.graph_mode, "eager" if config.enforce_eager else "replay"),
        "timing scope": (grid.scope, SCOPE),
        **{name: (recorded.get(name), value) for name, value in ours.items()},
    }
    differ = [
        f"{name} {theirs!r} in the bundle and {here!r} here"
        for name, (theirs, here) in pairs.items()
        if theirs != here
    ]
    if differ:
        raise ValueError(
            f"{key} was measured under other conditions: " + "; ".join(differ)
        )
    store.load(key, conditions=conditions(config, machine), gates=GateState.of())
    if grid.graph_mode == "replay":
        measured = sorted(rung for rung, cells in grid.decode.items() if cells.points)
        unmeasured = [
            r
            for r in ours["ladder"]
            if measured and measured[0] < r < measured[-1] and r not in measured
        ]
        if unmeasured:
            raise ValueError(
                f"{key} measured the decode rungs {measured}, and not the "
                f"ladder's rungs {unmeasured} between them"
            )
    over = [
        f"{name} {getattr(config, name)} is above the {bounds[name]}"
        for name in BOUNDS
        if getattr(config, name) > bounds[name]
    ]
    if over:
        raise ValueError(
            f"this run's {'; '.join(over)} the grid of {key} was planned for"
        )
    return StepGridBackend(grid, residual, bundle=str(key)), {
        "key": key.values,
        "digest": entry.digest,
    }
