"""Quick checks (under a minute):  python -m pytest tests/ -q   from the repository root.

The first three are the self-checks of stage9_sequential_qpe.py; the last runs paper/check_numbers.py, which asserts
the numbers quoted in the manuscript against results/ and verifies the code/result manifest."""
import importlib.util, pathlib, subprocess, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def s9():
    import os
    os.chdir(ROOT)                                # the stage scripts load each other by relative path
    spec = importlib.util.spec_from_file_location("s9", ROOT / "stage9_sequential_qpe.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_record_identity(s9): s9.check_reduction()
def test_posterior_identity_rule(s9): s9.check_identity()
def test_two_level_learning(s9): s9.demo()


def test_manuscript_numbers_and_manifest():
    r = subprocess.run([sys.executable, "check_numbers.py"], cwd=ROOT / "paper", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "manifest verified" in r.stdout


def test_power_grid_problems_and_budgets():
    """Stage 10: DC power-flow self-check, and no sequential run exceeds its time budget in the shipped results."""
    import glob, json, os
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("s10", ROOT / "stage10_power_grid.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    g.demo()
    over = lambda runs, budget: max(runs) > budget * (1 + 1e-9)
    files = [f for f in glob.glob("results/stage10_*.json") if not f.endswith("manifest.json")]; assert len(files) == 24
    for f in files:
        d = json.load(open(f))
        if "scale" in f: assert not any(over(r["runs"]["T"], d["mult"] * d["T_unit"]) for k, r in d.items() if k.startswith("sequential"))
        elif "qoi" in f: assert not any(over(r["sequential"]["runs"]["T"], r["T_tot"]) for k, r in d.items() if k != "problem")
        elif "compare" in f:
            for res in d["6-bit clock"].values(): assert not any(over(r["runs"]["T"], r["T_target"]) for k, r in res.items() if k.startswith("sequential"))
            assert not any(over(r["sequential"]["runs"]["T"], r["T_ref"]) for k, r in d.items() if k.endswith("-bit clock") and k != "6-bit clock")
