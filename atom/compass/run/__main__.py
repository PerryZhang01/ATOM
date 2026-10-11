# SPDX-License-Identifier: MIT
"""The standalone Clock Authority of a simulated run, started before its engines."""

import argparse
import os

from atom.compass import run

parser = argparse.ArgumentParser(prog="python -m atom.compass.run", description=__doc__)
parser.add_argument("--compass-run", required=True, help="the run file")
parser.add_argument(
    "--compass-clock-endpoint", required=True, help="where it listens, tcp://host:port"
)
args = parser.parse_args()
os.environ[run.ENV] = args.compass_run
run.authority(args.compass_clock_endpoint)
