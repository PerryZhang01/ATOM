# SPDX-License-Identifier: MIT
"""Print, per DP rank, the first step where a real and a simulated record part."""

import argparse
import json

from atom.compass.parity import compare

parser = argparse.ArgumentParser(
    prog="python -m atom.compass.parity", description=__doc__
)
parser.add_argument("real", help="the real run's record directory")
parser.add_argument("simulated", help="the simulated run's record directory")
args = parser.parse_args()
print(json.dumps(compare(args.real, args.simulated), indent=1))
