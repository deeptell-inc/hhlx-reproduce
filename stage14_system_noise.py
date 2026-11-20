"""Stage 14: the sequential estimator of stage 9 when the system register is not ideal.

A copy loses its eigencomponent at rate 1/T_sys and continues in a component drawn afresh, during identification and
during refinement.  Two models of the draw:
    redraw       : with the weights w of the input (the register is prepared again in |b>; the model of stage 9 `lifetime`)
    depolarising : uniformly over all N eigenvectors of A (the register is left maximally mixed)
Systems, each with the protocol and the seeds of its stored runs, without dephasing, ten runs:
    poisson8 : stage 9 `lifetime` (Poisson N = 8, budget 2e6, target 1e-3)
    case9    : the sequential row of the 9-bus system in the power-flow table (stage 10 `compare`)
    lorenz   : the finest-resolution row of the Lorenz readout at ten budget units (stage 11 `scale`)
so the runs with T_sys = inf repeat stored runs, and on poisson8 the redraw model repeats stage 9; `merge` asserts both.
Recorded per run: solution infidelity, compliance error, total-variation distance of the learned weights from the true
ones (each learned value assigned to the nearest eigenvalue), learned values far from every eigenvalue, copies, the
longest copy, the numbers of identification and refinement measurements, and the learned values with their weights and
label counts (sorted by value).  A jump is applied at the start of a measurement, whose whole evolution then runs with
the new eigenvalue; the identification count includes the copies of the last batch that the budget cuts off.  Nothing of stages 2-13 is modified; stage 9
gained the arguments `w_jump` and `count`, with unchanged defaults.  Run from the repository root:
    python stage14_system_noise.py                            # self-check
    python stage14_system_noise.py run case9 depolarising 1e5 # ten runs of one setting -> a part file
    python stage14_system_noise.py run lorenz redraw 3e4 4 5  # repetition 4 only (one OS process per repetition on lorenz)
    python stage14_system_noise.py merge | table | manifest   # sh run_stage14_all.sh runs all of it
"""
import os, sys, glob, json, importlib.util, numpy as np


def _load(name, path):
    m = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location(name, path)); spec.loader.exec_module(m); return m


S10 = _load("s10g", "stage10_power_grid.py")     # the grid problems, with its own instance of stage 9
S11 = _load("s11", "stage11_time_series.py")      # the ridge systems (it redirects its own instances of stages 9 and 10)
SYSTEMS, MODELS, REPS = ("poisson8", "case9", "lorenz"), ("redraw", "depolarising"), 10
T_SYS = (3e6, 1e6, 3e5, 1e5, 3e4, 1e4)            # finite lifetimes; inf is run once per system (the two models coincide)
KEYS = ("1-F", "eC", "wTV", "m", "false", "copies", "unlabelled", "T_copy_max", "n_id", "n_ref")
LISTS = ("mu", "weights", "counts")                # per run: learned values (7 decimals), their weights and label counts


def setup(system):
    """Stage-9 instance, problem, identification resolution, time budget, first seed and the stored reference runs."""
    if system == "poisson8":
        S9 = S10.S9; a, b = S9.S2.problems()["P3 Poisson N=8"]; lams = np.linalg.eigh(a)[0]
        return S9, a, b, S9.sig_id_for(lams, np.inf), 2e6, 1700, None
    if system == "case9":
        S9 = S10.S9; a, b = S10.problem(system)[:2]; lams = np.linalg.eigh(a)[0]; S9.set_noise(np.inf)
        ref = json.load(open(f"results/stage10_compare.{system}.json"))["6-bit clock"]["T2=inf"]; name = next(n for n in ref if n.startswith("sequential (budget x1.0, delta=0.001"))
        return S9, a, b, S9.sig_id_for(lams, np.inf, S9.FLOOR6), ref[name]["T_target"], 700, ref[name]["runs"]["1-F"]
    S9 = S11.S9; a, b = S11.problem(system)[:2]; lams = np.linalg.eigh(a)[0]; sid = S9.sig_id_for(lams, 2000.0)
    ref = json.load(open(f"results/stage11_scale.{system}.x10.json"))
    return S9, a, b, sid, 10 * ref["T_unit"], 700, ref[f"sequential, sig_id={sid:.5f}"]["runs"]["1-F"]


def run(system, model, T_sys, k0=0, k1=REPS):
    """Repetitions k0..k1-1 of one system, jump model and lifetime."""
    S9, a, b, sid, budget, seed, _ = setup(system); S9.set_noise(np.inf)
    lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; C0 = S9.truth(a, b, b)[0]
    w_jump = np.full(len(lams), 1.0 / len(lams)) if model == "depolarising" else None; r = {k: [] for k in KEYS + LISTS}
    for k in range(k0, k1):
        n = [0]; mu, wh, st = S9.learn(lams, w, np.inf, budget, np.random.default_rng(seed + k), sid, 1e-3, T_sys=T_sys, w_jump=w_jump, count=n)
        near = np.argmin(np.abs(lams[None, :] - np.asarray(mu)[:, None]), axis=1) if len(mu) else np.empty(0, int)
        val = (1 - S9.S2.fidelity(a, b, mu), abs(np.sum(wh / mu) - C0) / C0, 0.5 * float(np.abs(np.bincount(near, weights=wh, minlength=len(lams)) - w).sum()), len(mu),
               int(sum(np.min(np.abs(lams - c)) > 4 * sid for c in mu)), st["copies"], st["unlabelled"], st["T_copy_max"], n[0], st["n_ref_meas"])
        for key, x in zip(KEYS, val): r[key].append(float(x))
        o = np.argsort(mu); r["mu"].append([round(float(x), 7) for x in np.asarray(mu)[o]]); r["weights"].append([round(float(x), 8) for x in np.asarray(wh)[o]]); r["counts"].append([int(x) for x in np.asarray(st["counts"])[o]])
    return {"system": system, "model": model, "T_sys": "inf" if not np.isfinite(T_sys) else float(T_sys), "k0": k0, "sig_id": sid, "budget": budget, "N": len(lams), "runs": r}


def _part(system, model, T_sys, k0): return f"results/stage14_part.{system}.{model}.{T_sys:g}.{k0}.json"


def merge():
    """Joins the part files into results/stage14_system.json and checks them against the stored runs they repeat."""
    out = {}
    for system in SYSTEMS:
        S9, a, b, sid, budget, seed, ref = setup(system); o = out[system] = {"sig_id": sid, "budget": budget, "N": len(a), "weights": sorted(np.abs(np.linalg.eigh(a)[1].conj().T @ b)**2)}
        for model, T in [("redraw", np.inf)] + [(m, t) for m in MODELS for t in T_SYS]:
            parts = sorted((json.load(open(f)) for f in glob.glob(_part(system, model, T, 0)[:-6] + "*.json")), key=lambda p: p["k0"])
            r = {k: [x for p in parts for x in p["runs"][k]] for k in KEYS + LISTS}; assert len(r["1-F"]) == REPS and all(p["sig_id"] == sid and p["budget"] == budget for p in parts), (system, model, T)
            row = {k: float(np.median(r[k])) for k in KEYS} | {"1-F_max": float(np.max(r["1-F"])), "runs": r}
            if np.isfinite(T): o.setdefault(model, {})[f"T_sys={T:g}"] = row
            else: o["T_sys=inf"] = row
        if ref is not None: assert np.allclose(o["T_sys=inf"]["runs"]["1-F"], ref, rtol=0, atol=1e-15), system        # the stored runs of stages 10 and 11
    L = json.load(open("results/stage9_lifetime.json"))                # stage 9: medians of the same runs on Poisson 8
    for key, row in [("T_sys=inf", out["poisson8"]["T_sys=inf"])] + [(f"T_sys={T}", out["poisson8"]["redraw"][f"T_sys={T:g}"]) for T in T_SYS if f"T_sys={T}" in L]:
        assert all(abs(row[k] - L[key][k]) < 1e-15 for k in ("1-F", "eC", "m", "1-F_max")), key
    json.dump(out, open("results/stage14_system.json", "w"), indent=1)
    for f in glob.glob("results/stage14_part.*.json"): os.remove(f)
    print("stage 14:", sum(1 + len(MODELS) * len(T_SYS) for _ in SYSTEMS), "settings; the runs without system noise repeat the stored runs, the redraw model on Poisson 8 repeats stage 9")


def table():
    d = json.load(open("results/stage14_system.json")); g = lambda x: f"{x:.2g}"
    print("Medians of ten runs (largest infidelity in brackets). wTV: total-variation distance of the learned weights; id/copy: identification measurements per copy.\n")
    print("| system | model | T_sys | 1-F | compliance error | wTV | components | far values | copies | longest copy | id/copy | refinements |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s, o in d.items():
        for model, key, r in [("-", "inf", o["T_sys=inf"])] + [(m, k[6:], r) for m in MODELS for k, r in o[m].items()]:
            print(f"| {s} | {model} | {key} | {g(r['1-F'])} [{g(r['1-F_max'])}] | {g(r['eC'])} | {g(r['wTV'])} | {r['m']:g} | {r['false']:g} | {r['copies']:.0f} | {g(r['T_copy_max'])} | {r['n_id'] / r['copies']:.1f} | {r['n_ref']:.0f} |")


def run_manifest():
    """SHA-256 of this script, of the scripts whose functions it runs and of the result file (checked by paper/check_numbers.py)."""
    import hashlib, time
    ent = lambda p: {"sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "mtime": os.path.getmtime(p)}
    json.dump({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": {p: ent(p) for p in ("stage14_system_noise.py", "stage9_sequential_qpe.py", "stage10_power_grid.py", "stage11_time_series.py")},
               "results": {"results/stage14_system.json": ent("results/stage14_system.json")}}, open("results/stage14_manifest.json", "w"), indent=1); print("stage-14 manifest written")


def demo():                                       # self-check on a short run of the 9-bus system
    S9, a, b, sid, budget, seed, ref = setup("case9"); S9.set_noise(np.inf); lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; uni = np.full(len(lams), 1 / len(lams))
    go = lambda **kw: S9.learn(lams, w, np.inf, budget / 100, np.random.default_rng(5), sid, 1e-3, **kw)
    n = [0]; plain, same = go(), go(w_jump=uni, count=n)               # without jumps the new arguments change nothing, and the count is that of the measurements
    assert np.array_equal(plain[0], same[0]) and np.array_equal(plain[1], same[1]) and n[0] > plain[2]["copies"]
    red, dep = go(T_sys=300.0), go(T_sys=300.0, w_jump=uni)            # a lifetime far below a copy: the two models differ, and uniform jumps pull the weights towards 1/N
    tv = lambda r: 0.5 * np.abs(np.bincount(np.argmin(np.abs(lams[None, :] - r[0][:, None]), axis=1), weights=r[1], minlength=len(lams)) - w).sum()
    assert not np.array_equal(red[1], dep[1]) and tv(dep) > tv(red) and tv(dep) > 0.2 > tv(plain), (tv(plain), tv(red), tv(dep))
    assert abs(setup("lorenz")[3] - 3e-4) < 1e-12 and setup("poisson8")[4] == 2e6
    print("stage 14 self-check passed")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: demo()
    elif a[0] == "run":
        T = float(a[3]); k = (int(a[4]), int(a[5])) if len(a) > 4 else (0, REPS); assert a[1] in SYSTEMS and a[2] in MODELS and (np.isfinite(T) or a[2] == "redraw")
        p = run(a[1], a[2], T, *k); json.dump(p, open(_part(a[1], a[2], T, k[0]), "w")); print({kk: v for kk, v in p.items() if kk != "runs"}, "1-F", [float(f"{x:.3g}") for x in p["runs"]["1-F"]], "wTV", [float(f"{x:.2g}") for x in p["runs"]["wTV"]], flush=True)
    elif a[0] == "merge": merge()
    elif a[0] == "table": table()
    elif a[0] == "manifest": run_manifest()
    else: raise SystemExit(__doc__)
