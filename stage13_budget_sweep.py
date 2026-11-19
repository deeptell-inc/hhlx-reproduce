"""Stage 13: learning error and forecast error of the readout problems of stage 11 against the budget.

Estimators (Projective HHL, Recycled sequential HHL at its two identification resolutions): the time budget is
`mult` x the median time of 10^4 projective shots without dephasing (T_unit of stage 11), mult in MULTS, without
dephasing and with T2 = 300, ten runs with the seeds of stage 11 (so mult = 10 repeats its stored runs; `merge` asserts it).
Projective shots follow stage 11: 10^4 x mult without dephasing, with dephasing the largest number that keeps every run
within the budget.  Baselines of stage 12 (variational solver, quantum reservoir, circuit learning, DQI): circuit
executions in EXEC, ten runs.  Nothing of stages 2-12 is modified.  Run from the repository root:
    python stage13_budget_sweep.py                               # self-check
    python stage13_budget_sweep.py point lorenz 3 300 projective # one point; methods: projective, sequential-coarse, sequential-fine; T2: inf or 300
    python stage13_budget_sweep.py point lorenz 30 inf sequential-fine 4 5   # repetition 4 only (one OS process per repetition at the large budgets)
    python stage13_budget_sweep.py base lorenz qcl 1000000       # one baseline (vqls, qrc, qcl, dqi), one number of executions (all of EXEC when omitted)
    python stage13_budget_sweep.py merge                         # joins the part files into results/stage13_sweep.json
    python stage13_budget_sweep.py table                         # markdown table of results/stage13_sweep.json; sh run_stage13_all.sh runs all of it
    python stage13_budget_sweep.py manifest                      # hashes of code and result; the figure is drawn by paper/make_tables.py
"""
import os, sys, glob, json, importlib.util, numpy as np
S12 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s12", "stage12_nisq_baselines.py")); spec.loader.exec_module(S12)
S11 = S12.S11; S10, S9 = S11.S10, S11.S9
MULTS, EXEC, REPS = (0.1, 0.3, 1, 3, 10, 30), (10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7), 10
METHODS, BASE = ("projective", "sequential-coarse", "sequential-fine"), ("vqls", "qrc", "qcl", "dqi")


def point(case, mult, T2, method, k0=0, k1=REPS):
    """Repetitions k0..k1-1 of one estimator at one budget, with the forecast error of every run (the projective shots
    are fitted to all repetitions, so that estimator is run in one piece)."""
    a, b, v, sc, lams, U, w, C0, L0, n_gap, err = S10._setup(case); S9.set_noise(T2); deph = np.isfinite(T2); reps = k1 - k0
    budget = mult * json.load(open(f"results/stage11_scale.{case}.x10.json"))["T_unit"]
    if method == "projective":
        assert (k0, k1) == (0, REPS)
        if deph:                                  # as run_dephased of stage 11
            n_c = S9.n_c_for(lams, T2); t1 = next(r for k, r in json.load(open(f"results/stage11_compare.{case}.json"))["6-bit clock"][S11._key(T2)].items() if k.startswith("projective ("))["T"]
            shots = int(0.97e4 * budget / t1); r = S10._proj(lams, w, T2, shots, n_c, err, reps)
            while max(r["T"]) > budget: shots = int(0.97 * shots); r = S10._proj(lams, w, T2, shots, n_c, err, reps)
        else: n_c, shots = 6, int(round(1e4 * mult)); r = S10._proj(lams, w, T2, shots, n_c, err, reps)    # as run_scale of stage 10
        r["shots"] = [shots] * reps; extra = {"clock bits": n_c}
    else:
        sids = sorted({S9.sig_id_for(lams, T2, S9.FLOOR6), S9.sig_id_for(lams, T2 if deph else 2000.0)}, reverse=True); sid = sids[0 if method == "sequential-coarse" else -1]
        r = S10._seq(lams, w, T2, budget, sid, err, reps, seed=700 + k0); extra = {"sig_id": sid}; assert max(r["T"]) <= budget * (1 + 1e-9)
    r["NMSE"] = [S11.forecast_nmse(case, m) for m in r["mu"]]; r["dev"] = [S11.forecast_dev(case, m) for m in r["mu"]]
    return {"series": case, "noise": S11._key(T2), "method": method, "mult": mult, "budget": budget, "k0": k0} | extra | {"runs": r}


def base(case, method, E):
    """Ten runs of one stage-12 baseline within E circuit executions (None: not runnable)."""
    rngs = [np.random.default_rng(1200 + k) for k in range(REPS)]; p = S11.system(case)
    if method == "vqls":
        runs = [S12.vqls(case, g, E) for g in rngs]
        if runs[0] is None: return None
        W = [p["scale"] * x["x"][:S11.N].real for x in runs]; ref = S11.readout(case, np.linalg.eigvalsh(p["a"]))
        r = {"1-F": [x["1-F"] for x in runs], "NMSE": [S11.nmse(case, x) for x in W], "dev": [float(np.mean((p["Zt"] @ (x - ref))**2) / np.var(p["yt"])) for x in W]}
    elif method == "qrc":
        F = S12.qrc_features(case); s = E // ((S11.NTR + S11.NTE) * S12.NV)
        if s < 1: return None
        runs = [S12._fit(case, S12.est(F, s, g)) for g in rngs]; r = {q: [x[q] for x in runs] for q in ("NMSE", "dev")} | {"shots per circuit": [int(s)] * REPS}
    elif method == "qcl": runs = [S12.qcl(case, g, E) for g in rngs]; r = {q: [x[q] for x in runs] for q in ("NMSE", "dev")}
    else:                                         # DQI: the best (most constraints satisfied) of E samples, as run_dqi of stage 12, for both sizes of the constraint sets
        r = {}; m = S12.P_DQI - 1
        for size in S12.R_DQI:
            S, ERR, _ = S12.dqi_instance(case, size); amp, _ = S12.dqi_weights(m, (S12.N_DQI - 1) // 2, S12.P_DQI, size); cnt = np.bincount(S, minlength=m + 1)
            pr = cnt * amp**2 / S12.P_DQI**S12.N_DQI; order = np.argsort(S, kind="stable"); start = np.concatenate([[0], np.cumsum(cnt)]); r[f"NMSE, sets of {size}"] = []
            for g in rngs: top = g.choice(m + 1, E, p=pr).max(); r[f"NMSE, sets of {size}"].append(float(ERR[order[start[top] + g.integers(0, cnt[top])]]))
    return {q: float(np.median(x)) for q, x in r.items()} | ({"reproduced": int(np.sum(np.array(r["dev"]) < 1e-3))} if "dev" in r else {}) | {"runs": r}


def merge():
    files = sorted(glob.glob("results/stage13_part.*.json")); parts = [json.load(open(f)) for f in files]; out = {"points": [], "baselines": {}}; grp = {}
    for p in parts:
        if "base" in p: out["baselines"].setdefault(p["series"], {}).setdefault(p["base"], {}).update(p["rows"])
        else: grp.setdefault((p["series"], p["noise"], p["method"], p["mult"]), []).append(p)
    for ps in grp.values():                       # the repetitions of a point in order, then the medians
        ps.sort(key=lambda p: p["k0"]); r = {q: sum((p["runs"][q] for p in ps), []) for q in ps[0]["runs"]}; assert len(r["dev"]) == REPS, (ps[0]["series"], ps[0]["method"], ps[0]["mult"], len(r["dev"]))
        out["points"].append({k: x for k, x in ps[0].items() if k not in ("runs", "k0")} | {"reproduced": int(np.sum(np.array(r["dev"]) < 1e-3))} | S10._med(r))
    want = {(c, k, m, x) for c in S11.SERIES for k in ("T2=inf", "T2=300.0") for m in METHODS for x in MULTS}; got = {(p["series"], p["noise"], p["method"], p["mult"]) for p in out["points"]}
    assert got == want, sorted(want ^ got)
    assert all(set(out["baselines"][c][b]) == {str(E) for E in EXEC} for c in S11.SERIES for b in BASE), "baseline rows missing"
    for p in out["points"]:                       # mult = 10 repeats the stored runs of stage 11
        if p["mult"] == 10:
            ref = json.load(open(f"results/stage11_scale.{p['series']}.x10{S11._tag(float(p['noise'][3:]))}.json"))
            key = next(k for k in ref if k.startswith("projective")) if p["method"] == "projective" else f"sequential, sig_id={p['sig_id']:.5f}"
            assert np.allclose(p["runs"]["1-F"], ref[key]["runs"]["1-F"]), (p["series"], p["noise"], p["method"])
    out["points"].sort(key=lambda p: (p["series"], p["noise"], p["method"], p["mult"])); json.dump(out, open("results/stage13_sweep.json", "w"), indent=1)
    for f in files: os.remove(f)
    print("stage 13:", len(out["points"]), "points and", sum(len(v) for v in out["baselines"].values()), "baseline sweeps; the points at ten times the budget equal the runs of stage 11")


def table():
    d = json.load(open("results/stage13_sweep.json")); g = lambda x: "-" if x is None else f"{x:.2g}"
    print("| series | noise | method | budget (x 10^4 projective shots) | shots | time | 1-F | difference from the ridge forecast | forecast NMSE | runs reproducing the forecast |\n|---|---|---|---|---|---|---|---|---|---|")
    for p in d["points"]: print(f"| {p['series']} | {p['noise']} | {p['method']} | {p['mult']:g} | {g(p.get('shots'))} | {g(p['T'])} | {g(p['1-F'])} | {g(p['dev'])} | {g(p['NMSE'])} | {p['reproduced']} of {len(p['runs']['dev'])} |")
    print("\n| series | baseline | executions | 1-F | difference from the ridge forecast | forecast NMSE | runs reproducing the forecast |\n|---|---|---|---|---|---|---|")
    for c, bs in d["baselines"].items():
        for b in BASE:
            for E in EXEC:
                r = bs[b][str(E)]
                if r is None: print(f"| {c} | {b} | {E:.0e} | not runnable | | | |"); continue
                for q in ([k for k in r if k.startswith("NMSE,")] or ["NMSE"]): print(f"| {c} | {b}{q[4:]} | {E:.0e} | {g(r.get('1-F'))} | {g(r.get('dev'))} | {g(r[q])} | {r.get('reproduced', '-')} |")


def run_manifest():
    """SHA-256 of this script, of the two scripts it imports and of the result file, with modification times (checked by
    paper/check_numbers.py; the budgets come from stage-11 results, which the stage-11 manifest covers)."""
    import hashlib, time
    ent = lambda p: {"sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "mtime": os.path.getmtime(p)}
    json.dump({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": {p: ent(p) for p in ("stage13_budget_sweep.py", "stage12_nisq_baselines.py", "stage11_time_series.py")},
               "results": {"results/stage13_sweep.json": ent("results/stage13_sweep.json")}}, open("results/stage13_manifest.json", "w"), indent=1); print("stage-13 manifest written")


def demo():                                       # self-check: a short sequential run stays within its budget and the forecast errors are those of its eigenvalues
    p = point("lorenz", 0.02, np.inf, "sequential-coarse", 3, 5); q = base("lorenz", "qrc", 10**5)
    assert max(p["runs"]["T"]) <= p["budget"] and len(p["runs"]["dev"]) == 2 and p["k0"] == 3 and np.isclose(p["runs"]["NMSE"][0], S11.forecast_nmse("lorenz", p["runs"]["mu"][0]))
    assert base("lorenz", "qrc", 10**4) is None and q["runs"]["shots per circuit"][0] == 8 and 0 < q["NMSE"] < 1
    print("stage 13 self-check passed")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: demo()
    elif a[0] == "point":
        p = point(a[1], float(a[2]), float(a[3]), a[4], *map(int, a[5:])); print({k: x for k, x in p.items() if k != "runs"}, "1-F", [float(f"{x:.3g}") for x in p["runs"]["1-F"]], flush=True)
        json.dump(p, open(f"results/stage13_part.{a[1]}.{a[3]}.{a[4]}.x{float(a[2]):g}.k{p['k0']:02d}.json", "w"))
    elif a[0] == "base":
        rows = {str(E): base(a[1], a[2], E) for E in ([int(a[3])] if len(a) > 3 else EXEC)}
        for E, r in rows.items(): print(a[1], a[2], E, None if r is None else {k: (float(f"{x:.3g}") if isinstance(x, float) else x) for k, x in r.items() if k != "runs"}, flush=True)
        json.dump({"series": a[1], "base": a[2], "rows": rows}, open(f"results/stage13_part.{a[1]}.{a[2]}.{a[3] if len(a) > 3 else 'all'}.json", "w"))
    elif a[0] == "merge": merge()
    elif a[0] == "table": table()
    elif a[0] == "manifest": run_manifest()
