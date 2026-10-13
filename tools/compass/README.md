<!-- SPDX-License-Identifier: MIT -->
# Compass tools

Run from a tree root with ATOM importable (`pip install -e .`, or
`PYTHONPATH=<tree root>`).

## `step_fit_plan.py`: step fit's grid, planned and checked

Plans which grid points step fit's bundle must measure, from the shapes of
recorded steps, and checks that a published bundle covers a set of steps.
Needs no device, and measures and fits nothing.

```
python tools/compass/step_fit_plan.py plan DEFINITION RECORD...
python tools/compass/step_fit_plan.py coverage RUN CONFIG RECORD...
```

**RECORD** is a step record directory: set `ATOM_COMPASS_PARITY_RECORD` on a
real run, or on a simulated run of the target configuration
(`atom/compass/parity/README.md`). Only shapes are read, never timestamps.

**DEFINITION** is the grid definition, a JSON object; every key is required:

| Key | Holds |
|---|---|
| `prefill` | `{"B": [...], "T": [...], "c": [...]}`: the prefill nodes, `T` the step's token count and `c` the cached context |
| `decode` | `{"c": [...]}`: the decode context nodes; the B nodes come from the ladder |
| `margin` | cells added outward on every axis, and rungs either side on the ladder |
| `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, `capture_sizes` | the target configuration's |
| `kv_cache_block_size`, `num_kvcache_blocks` | read from one real startup of the target on the target device |

Every number is a non-negative integer and each node list ascends.

**`plan`** first places the nodes the target's bounds give, dropping any
definition node past one. Prefill's top nodes are `max_num_seqs` on B,
`max_num_batched_tokens` on T and `max_model_len - max_num_batched_tokens` on
c. Decode runs on the target's ladder (the capture sizes the scheduler can
produce): each rung has B nodes at the previous rung + 1, at the rung, and at
each B between them that a recorded step on the rung reaches past the rung's
KV limit; its c nodes add each B node's KV limit,
`block size · ⌊num_kvcache_blocks / B⌋ - 1`, and `max_model_len - 1`.

It then reads each step where step fit's backend reads it: prefill on
`(B, T, c)`, split by whether the step produces output, and decode on its
rung. It plans every corner of each step's cell, widened by `margin` nodes on
every axis and to `margin` rungs either side, each such rung at all its B
nodes; and, on any rung between two planned rungs, its B nodes at its c nodes
spanning those planned. A point that is no batch the target runs is not
planned: B above `max_num_seqs`, T above `max_num_batched_tokens`, T not a
multiple of B or below it (N_Q = T/B is a whole count of at least one),
c + N_Q above `max_model_len`, or B·⌈(c + N_Q) / block size⌉ above the block
count, assuming no prefix sharing.

It prints JSON: `graph_mode`, `scope`, `bounds`, and `prefill` and `decode` in
a bundle grid's layout (`atom/compass/backends/step_fit.py`) with each point as
its coordinates, `[B, T, c]` or `[rung, B, c]`; `counts` per phase and rung;
`infeasible`, the points not planned and why; and `unplaced`, the steps outside
the grid with the backend's reason. To publish the grid, give each point its
measurement (`{"at": ..., "time": ..., "statistic": ..., "repeats": ...}`) and
a `unit`, and put `bounds` beside the grid in the bundle
(`atom/compass/run/step_fit.py`).

**`coverage`** loads the bundle with `run/step_fit.py::load`. **RUN** is the run
file the graded run will use, for its `step_fit` key and its `machine`.
**CONFIG** is that run's resolved ATOM config as `load` reads it, a JSON
object with the `Config` fields `model`, `tensor_parallel_size`,
`enforce_eager`, `capture_sizes`, `max_num_seqs`, `max_num_batched_tokens`,
`max_model_len`, `kv_cache_block_size` and `kv_cache_dtype`, plus
`data_parallel_size` and `quantization`, the quantization config's
`compute_hash()`. A step is covered when every corner of its cell was measured
and the residual answers it, inside its hull.

It prints JSON: `steps`, `covered` and `share`; `missing`, each grid's
unmeasured corners, keyed `prefill output`, `prefill no_output` or
`decode rung <R>`; `refused`, each step the grid cannot place or the residual
does not answer, with the reasons; and `prefill_above_top_c`, how many prefill
steps are cached past prefill's top c node, which refuse. Those are a prefix
hit near `max_model_len`, or a long prompt's last chunk when `max_model_len` is
not a multiple of the token budget; check records of a run with prefix caching
to see the first kind. It exits 1 unless every step is covered.

## `detect_loose_refusals.py`

Its docstring says how to run it and what it reports.
