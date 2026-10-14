<!-- SPDX-License-Identifier: MIT -->
# The step record and its compare

Record a run by setting `ATOM_COMPASS_PARITY_RECORD` to a directory before
starting it, real or simulated; `__init__.py` says what each line holds.
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
DP2 run of `traffic.py`'s workload, then a simulated one that replays the real
run's arrivals and step times (`replay.py`), then the compare.
