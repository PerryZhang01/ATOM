<!-- SPDX-License-Identifier: MIT -->
# The step record and its compare

Record a run by setting `ATOM_COMPASS_PARITY_RECORD` to a directory before
starting it, real or simulated; `__init__.py` says what each line holds.
Beside each `dp<r>.jsonl`, the rank's first step writes `dp<r>.engine.json`,
the engine its runner resolved: `keyed` (`atom/compass/run/step_grid.py`),
which holds the ladder the runner replays; the scheduler's bounds
`max_num_seqs`, `max_num_batched_tokens` and `max_model_len`; and
`num_kvcache_blocks`, the KV block count the runner's `allocate_kv_cache` set.
A simulated run also writes `dp<r>.cost.jsonl`, a line per step saying what
its charge was made of (`_group_step_seconds` in
`atom/compass/runner/overrides.py` says what each line holds).
Compare two records with

```
python -m atom.compass.parity REAL_DIR SIMULATED_DIR
```

It prints `compare`'s report as JSON: per DP rank, the first step whose
decision differs (null where none does), the steps compared and how many are
identical, and each request's DP rank in both.

`python -m atom.compass.parity.recurrence REAL_DIR` gates the host/device
recurrence on a real record with device windows (`recurrence.py` says how).

`scripts/compass/parity_one_node.sh` runs the whole test on one node: a real
DP-attention run of `traffic.py`'s workload (DP2, or DeepSeek-V4-Pro at DP8;
its header says how to pick), then a simulated one that replays the real
run's arrivals and step times (`replay.py`), then the compare.
