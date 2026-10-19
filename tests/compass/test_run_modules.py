# SPDX-License-Identifier: MIT
"""The published Qwen3.8-27B's meta build, walked at TP1 and TP2, classifies
every module, each at its rank's shapes.

The build is ATOM's own model class on the meta device, as the simulated
runner builds it, at a peerless group of the width; it owns its process
because the driverless declarations are process-global.
"""

import collections
import json
import os
import pathlib
import subprocess
import sys

import pytest

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")

BUILD = """
import json, pathlib, sys, torch
tp, root = int(sys.argv[1]), pathlib.Path(sys.argv[2])
from atom.compass.capture import driverless
driverless.install(tp, str(root))
from atom.config import Config, set_current_atom_config
config = Config(model=str(root / "model"), tensor_parallel_size=tp, load_dummy="empty")
set_current_atom_config(config)
from atom.model_engine.model_runner import support_model_arch_dict
from atom.utils import resolve_obj_by_qualname
from atom.compass.runner.overrides import FactoriesOnMeta
from atom.compass.run.modules import walk
model = support_model_arch_dict[config.hf_config.architectures[0]]
torch.set_default_dtype(config.torch_dtype)
with FactoriesOnMeta():
    tree = resolve_obj_by_qualname(model)(config)
found = [[m.kind, list(m.dims), m.fmt, m.ranks, m.fused] for m in walk(tree)]
print(json.dumps(found))
"""


def _walked(tp, tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    done = subprocess.run(
        [sys.executable, "-c", BUILD, str(tp), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return collections.Counter(
        (kind, tuple(dims), fmt, ranks, fused)
        for kind, dims, fmt, ranks, fused in json.loads(done.stdout.splitlines()[-1])
    )


@pytest.mark.parametrize("tp", [1, 2])
def test_the_27b_classifies_every_module_at_its_ranks_shapes(tp, tmp_path):
    text = json.loads(CONFIG.read_text())["text_config"]
    full = text["layer_types"].count("full_attention")
    linear = text["layer_types"].count("linear_attention")
    hidden, inter, layers = 5120, 17408, text["num_hidden_layers"]
    found = _walked(tp, tmp_path)

    assert not [m for m in found if m[0] == "unclassified"]
    assert found[("attention", (24 // tp, 4 // tp, 256), "bf16", 1, False)] == full
    assert found[("rotary", (64,), "bf16", 1, False)] == full
    gdn = ("linear_attention", (48 // tp, 16 // tp, 128, 128), "bf16", 1, False)
    assert found[gdn] == linear
    assert found[("conv", (10240 // tp,), "bf16", 1, False)] == linear
    assert found[("act", (inter // tp,), "bf16", 1, False)] == layers
    # The two row-parallel GEMMs of every layer, the only ones whose output is
    # the hidden width, reduce over the group; an attention's and a linear
    # attention's output projection share a shape.
    rows = {
        d: (n, r)
        for (k, d, _, r, _), n in found.items()
        if k == "gemm" and d[0] == hidden
    }
    assert rows == {
        (hidden, inter // tp): (layers, tp),
        (hidden, 24 * 256 // tp): (full + linear, tp),
    }
    assert found[("embedding", (hidden,), "bf16", tp, False)] == 1
    assert found[("lm_head", (248320 // tp, hidden), "bf16", tp, False)] == 1
    # Two norms a layer, the final norm, a q and k norm per attention, and
    # one gated norm per linear attention; none fuses the all-reduce.
    norms = sum(n for (k, _, _, r, f), n in found.items() if k == "norm" and r == 1)
    assert norms == 2 * layers + 1 + 2 * full + linear
