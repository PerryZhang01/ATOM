# SPDX-License-Identifier: MIT
"""`compass fit steps`: a step-level backend fitted on labelled samples and published.

    compass fit steps --backend step-linear [--form FORM] [--seed S] RECORD...
        --client-run RECORD EXPORT... --machine M --store ROOT --out DIR

Each backend in `BACKENDS` states the kind it publishes, the label sources it
takes and its fit, which returns the key fields the configuration does not
give, the params and the report. `step-linear` takes real runs only and fits
`step_linear_fit.FORMS`' `--form` at the records' DP width, holding out each
run on the others when there are two or more at width 1, as that module's
command does; `--seed` fixes its shuffles. The law states the configuration's
`machine_spec`, as that command's does, and the key holds it. The report's
provenance names each run with every trace its steps served. A step stub's
coefficients are stated, not fitted, so `step-stub` refuses.
"""

import json

from atom.compass.artifacts import Key, Kind
from atom.compass.backends import step_linear_fit
from atom.compass.fit import publish

PROG = "compass fit steps"
STUB = "a step stub's coefficients are stated, not fitted; a run takes their file"


def _step_linear(samples, config, args) -> tuple[dict, dict, dict]:
    options = {"form": args.form, "dp": len({s.rank for s in samples})}
    law, report = step_linear_fit.fit(samples, options, args.seed)
    if options["dp"] == 1 and len({s.provenance["run"] for s in samples}) > 1:
        step_linear_fit.mark_held_out(law, report, samples, options, args.seed)
    report["provenance"] = publish.runs(samples)
    law["machine_spec"] = config["machine_spec"]
    return {"form": args.form, "machine_spec": law["machine_spec"]}, law, report


#: Each backend's name, to the kind it publishes, its label sources and its fit.
BACKENDS = {"step-linear": (Kind.STEP_LINEAR_LAW, ("record",), _step_linear)}


def main(argv: list[str]) -> None:
    parser = publish.parser(PROG, __doc__.split("\n\n")[0])
    parser.add_argument("--backend", required=True, help=", ".join(BACKENDS))
    parser.add_argument("--form", choices=step_linear_fit.FORMS, default="base")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.backend == "step-stub":
        raise ValueError(STUB)
    if args.backend not in BACKENDS:
        raise ValueError(
            f"no backend {args.backend} is fitted; {', '.join(BACKENDS)} is"
        )
    kind, sources, fit = BACKENDS[args.backend]
    samples, config = publish.inputs(args, sources)
    fields, params, report = fit(samples, config, args)
    key = Key.of(
        kind,
        **{f: config[f] for f in ("model", "parallel", "source_root")},
        engine_config=json.dumps(config["engine_config"], sort_keys=True),
        **fields,
    )
    publish.publish(args, " ".join([PROG, *argv]), key, config, params, report)
