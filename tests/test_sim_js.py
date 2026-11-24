import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mc.models import _centre
from mc.risk import prob_end_beyond, prob_touch, var_cvar

JS = Path(__file__).resolve().parents[1] / "src" / "mc" / "sim.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _node(expr: str):
    script = f"const m = require({json.dumps(str(JS))}); console.log(JSON.stringify({expr}));"
    # through stdin: Linux caps a single command-line argument at 128 KB
    out = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_risk_numbers_match_python():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1000, 5000)
    js = _node(f"m.varCvar({json.dumps(pnl.tolist())}, 0.95)")
    assert js == pytest.approx(var_cvar(pnl, 0.95), rel=1e-9)
    paths = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (400, 60)), axis=1))
    paths = np.hstack([np.full((400, 1), 100.0), paths])
    js = _node(f"(() => {{ const p = {json.dumps(paths.tolist())};"
               " return [m.probTouch(p, 80), m.probEnd(p, 80), m.probTouch(p, 120)]; })()")
    assert js == pytest.approx([prob_touch(paths, 80), prob_end_beyond(paths, 80),
                                prob_touch(paths, 120)])


def test_centre_matches_python():
    hist = np.random.default_rng(1).normal(0.001, 0.02, 500)
    assert _node(f"m.centre({json.dumps(hist.tolist())})") == pytest.approx(_centre(hist).tolist())


@pytest.mark.parametrize("model", ["gbm", "t", "jump"])
def test_parametric_models_have_zero_drift_and_the_right_volatility(model):
    p = {"sd": 0.02, "df": 4.0, "jumpOk": True, "lam": 0.02, "mJ": -0.01, "sJ": 0.05,
         "sdD": 0.017}
    js = _node(f"(() => {{ const r = m.makeRandom(3), s = [];"
               " for (let i = 0; i < 200; i++)"
               f" s.push(...m.steps('{model}', {json.dumps(p)}, [], 250, r));"
               " const mean = s.reduce((a, b) => a + b, 0) / s.length;"
               " const sd = Math.sqrt(s.reduce((a, b) => a + (b - mean) ** 2, 0)"
               " / s.length);"
               " const e = s.reduce((a, b) => a + Math.exp(b), 0) / s.length;"
               " return [mean, sd, e]; })()")
    mean, sd, e = js
    if model == "jump":
        expect_sd = np.sqrt(0.017**2 + 0.02 * (0.01**2 + 0.05**2))
        assert sd == pytest.approx(expect_sd, rel=0.05)
    else:
        assert sd == pytest.approx(0.02, rel=0.03)
        assert e == pytest.approx(1.0, abs=5e-4)  # zero drift: a day's growth averages 1


def test_bootstraps_replay_history_and_blocks_stay_in_order():
    hist = np.random.default_rng(2).normal(0, 0.02, 300)
    c = _centre(hist)
    js = _node(f"(() => {{ const r = m.makeRandom(5), c = m.centre({json.dumps(hist.tolist())});"
               " return [Array.from(m.steps('bootstrap', {}, c, 100, r)),"
               " Array.from(m.steps('block', {}, c, 40, r))]; })()")
    boot, block = np.array(js[0]), np.array(js[1])
    assert all(np.isclose(c, v).any() for v in boot)
    first = int(np.argmin(np.abs(c - block[0])))
    assert block[:20] == pytest.approx(c[(first + np.arange(20)) % len(c)])
