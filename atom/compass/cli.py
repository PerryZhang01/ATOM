# SPDX-License-Identifier: MIT
"""The `compass` console script: it parses a command's words and dispatches.

Each command in `COMMANDS` runs its module's `main` with the arguments after
its words; a module not yet built refuses as not built. Each command in
`REFUSED`, the op-level half and the deprecated ones, refuses with its reason.
A refusal a command raises (`ValueError`, `OSError`, `ArtifactRefusal`,
`GpuRefusal`, `SpecRefusal`) exits with its message rather than a traceback.
"""

import importlib
import sys

from atom.compass.artifacts import ArtifactRefusal
from atom.compass.drive.gpu import GpuRefusal
from atom.compass.spec import SpecRefusal

#: Each command, to its module and what it does.
COMMANDS = {
    "record": ("atom.compass.drive.record", "a real run's step record [GPU]"),
    "trace steps": (
        "atom.compass.drive.trace_steps",
        "the steps a simulated run schedules, published as a shape_population",
    ),
    "trace ops": (
        "atom.compass.capture",
        "steps traced with no device, an op_graph per shape",
    ),
    "measure steps": (
        "atom.compass.bench.measure",
        "batches replayed or planned, measured on the bench [GPU]",
    ),
    "measure ops": (
        "atom.compass.bench.ops",
        "each op_graph leaf timed alone on one device, cold [GPU]",
    ),
    "measure memory": (
        "atom.compass.drive.measure_memory",
        "an engine start's memory readings [GPU]",
    ),
    "fit steps": ("atom.compass.fit.steps", "a step-level backend's fit"),
    "fit host": ("atom.compass.fit.host", "the host laws, fitted on real runs"),
    "run": ("atom.compass.drive.run", "a simulated serve"),
}
OPS = "the op-level half of the toolchain is not built yet"
MACHINE = (
    "machine specs are maintained by developers, one per supported machine; "
    "pass one with --machine"
)
#: Each command that refuses, to why and where its work is.
REFUSED = {
    "fit ops": OPS,
    "trace": "there is no bare trace: use compass trace steps or compass trace ops",
    "plan": (
        "deprecated: compass measure steps --grid plans the step grid's points, "
        "and compass run --coverage checks a run's coverage before it starts"
    ),
    "validate": "deprecated: grading is python -m atom.compass.validate",
    "explain": "deprecated: " + MACHINE,
    "probe": "deprecated: " + MACHINE,
    "spec": "deprecated: " + MACHINE,
    "measure collectives": "deprecated: collectives are priced by compass measure ops",
    "discover": "deprecated: renamed compass trace steps",
}
USAGE = "usage: compass <command> [<args>]; compass <command> --help for its own\n\n"
USAGE += "\n".join(f"  {c:16} {doc}" for c, (_, doc) in COMMANDS.items())
USAGE += "\n" + "\n".join(
    f"  {c:16} (ops half)" for c, why in REFUSED.items() if why == OPS
)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] in (["-h"], ["--help"]):
        print(USAGE)
        return
    words = (" ".join(argv[:2]), " ".join(argv[:1]))
    name = next((w for w in words if w in COMMANDS or w in REFUSED), None)
    if name is None:
        raise SystemExit(f"compass: no command {words[0]!r}\n{USAGE}")
    if name in REFUSED:
        raise SystemExit(f"compass {name}: {REFUSED[name]}")
    module = COMMANDS[name][0]
    try:
        command = importlib.import_module(module)
    except ModuleNotFoundError as missing:
        if module != missing.name and not module.startswith(f"{missing.name}."):
            raise
        raise SystemExit(f"compass {name}: not built yet") from missing
    try:
        command.main(argv[len(name.split()) :])
    except (ValueError, OSError, ArtifactRefusal, GpuRefusal, SpecRefusal) as refused:
        raise SystemExit(f"compass {name}: {refused}") from refused
