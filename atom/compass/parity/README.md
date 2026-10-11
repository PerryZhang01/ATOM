<!-- SPDX-License-Identifier: MIT -->
# The step record and its compare

Record a run by setting `ATOM_COMPASS_PARITY_RECORD` to a directory before
starting it, real or simulated; `__init__.py` says what each line holds.
Compare two records with

```
python -m atom.compass.parity REAL_DIR SIMULATED_DIR
```

It prints `compare`'s report as JSON: per DP rank, the first step whose
decision differs (null where none does), and each request's DP rank in both.
