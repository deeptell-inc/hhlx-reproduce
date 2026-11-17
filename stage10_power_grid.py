"""Stage 10: the estimators of stage 9 on DC power-flow problems built from standard test systems.

Linear system: B' theta = P (reduced bus susceptance matrix, net real-power injections, slack bus removed), as in
MATPOWER's DC power flow (b = 1/(x tap), bus shunt conductance counted as load).  A = B'/lam_max, |b> = P/|P|.
Observables: fidelity of the normalised solution; the DC quadratic form C = P^T theta = sum_l f_l^2/b_l [x baseMVA],
which approximates the series reactive consumption of the lines for lossless branches, unit voltages, small angles
and nominal taps (series_q compares the two); the flow on one line, a linear functional <v|theta> [MW].  The line is
the one with the largest true flow, chosen from the classical solution.
Data: data/grid_cases.json (MATPOWER test cases; provenance in its "source" field).  Run from the repository root:
    python stage10_power_grid.py                 # self-check and a table of the spectral measures
    python stage10_power_grid.py compare case14  # one OS process per case; writes results/stage10_compare.<case>.json
    python stage10_power_grid.py qoi case14      # writes results/stage10_qoi.<case>.json
    python stage10_power_grid.py scale case30 4  # error against time; writes results/stage10_scale.<case>.x<mult>.json
    python stage10_power_grid.py methods|observables|backend|poly <case>, table, manifest
"""
import sys, os, glob, json, importlib.util, numpy as np
S9 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s9", "stage9_sequential_qpe.py")); spec.loader.exec_module(S9)
CASES = json.load(open("data/grid_cases.json"))["cases"]


def dc_power_flow(case):
    """Reduced B', injections P [pu], branch-flow matrix Bf (columns of the non-slack buses), base MVA."""
    d = CASES[case]; ids = [r[0] for r in d["bus"]]; ix = {b: k for k, b in enumerate(ids)}; n = len(ids)
    br = [r for r in d["branch"] if r[6] > 0]; assert all(r[5] == 0 for r in br), "phase shifters not handled"
    Bf = np.zeros((len(br), n))
    for l, (f, t, r_, x, tap, _, _) in enumerate(br):
        b = 1 / (x * (tap or 1.0)); Bf[l, ix[f]], Bf[l, ix[t]] = b, -b
    inc = np.sign(Bf); B = inc.T @ Bf
    P = -np.array([r[2] + r[3] for r in d["bus"]])
    for g, pg, on in d["gen"]: P[ix[g]] += pg * (on > 0)
    keep = [k for k, r in enumerate(d["bus"]) if r[1] != 3]; assert len(keep) == n - 1
    return B[np.ix_(keep, keep)], P[keep] / d["baseMVA"], Bf[:, keep], d["baseMVA"], [(r[0], r[1]) for r in br]


def series_q(case):
    """Series reactive consumption [MVAr] of the lossless branch model with unit bus voltages at the DC angles:
    (angle-dependent part sum 2(1-cos dtheta)/(x tap), tap term sum (1/tap-1)^2/x, number of non-unit taps).
    base MVA x P^T theta is the small-angle form of the first part; the tap term does not depend on the angles."""
    d = CASES[case]; B, P, Bf, base, lines = dc_power_flow(case); ids = [r[0] for r in d["bus"]]; ix = {b: k for k, b in enumerate(ids)}
    th = np.zeros(len(ids)); th[[k for k, r in enumerate(d["bus"]) if r[1] != 3]] = np.linalg.solve(B, P); qa = qt = 0.0; n = 0
    for f, t, r_, x, tap, _, on in d["branch"]:
        if on > 0: tau = tap or 1.0; qa += 2 * (1 - np.cos(th[ix[f]] - th[ix[t]])) / (x * tau); qt += (1 / tau - 1)**2 / x; n += tau != 1.0
    return base * qa, base * qt, int(n)


def problem(case):
    """(a, b, v, scales): unit-norm b and v, a with largest eigenvalue 1; scales convert back to MVAr and MW."""
    B, P, Bf, base, lines = dc_power_flow(case); lam_max = np.linalg.eigvalsh(B).max(); theta = np.linalg.solve(B, P)
    l = int(np.argmax(np.abs(Bf @ theta))); v = Bf[l] + 0j; nv, nP = np.linalg.norm(v), np.linalg.norm(P)
    return B / lam_max, P / nP + 0j, v / nv, {"C_MVAr": base * nP**2 / lam_max, "L_MW": base * nv * nP / lam_max, "line": lines[l], "n": len(P)}


def describe():
    for case in CASES:
        a, b, v, sc = problem(case); lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; C, E, L = S9.truth(a, b, v)
        ws = np.sort(w)[::-1]; wC = np.sort(w / lams)[::-1] / np.sum(w / lams)
        print(f"{case}: n={sc['n']} kappa={1/lams.min():.0f} min gap={np.diff(lams).min():.2g} eff. components={1/np.sum(w**2):.1f} "
              f"components for 99% of weight={int(np.searchsorted(np.cumsum(ws), 0.99)) + 1}, of C={int(np.searchsorted(np.cumsum(wC), 0.99)) + 1}, weights<1e-3: {int((w < 1e-3).sum())} | "
              f"C={C * sc['C_MVAr']:.2f} MVAr (series Q of the unit-voltage model: {series_q(case)[0]:.2f} + tap term {series_q(case)[1]:.2f}), line {sc['line']} flow={abs(L) * sc['L_MW']:.1f} MW")


def demo():                                       # self-check: power balance and Kirchhoff on every case
    for case in CASES:
        B, P, Bf, base, _ = dc_power_flow(case); theta = np.linalg.solve(B, P); f = Bf @ theta
        assert np.allclose(np.sign(Bf).T @ f, P, atol=1e-9)                       # flows leaving each non-slack bus = injection
        assert np.isclose(P @ theta, np.sum(f**2 / np.abs(Bf).max(1)))            # C = sum f^2 / b
        a, b, v, sc = problem(case); C, _, L = S9.truth(a, b, v)
        assert np.isclose(C * sc["C_MVAr"], base * P @ theta) and np.isclose(abs(L) * sc["L_MW"], base * np.abs(f).max())
    print("stage 10 self-check passed")

def _setup(case):
    a, b, v, sc = problem(case); lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; C0, _, L0 = S9.truth(a, b, v)
    n_gap = min(int(np.ceil(np.log2(2.5 / max(np.diff(lams).min(), 1e-3)))), 10)   # clock that resolves the smallest gap, at most 10 bits
    err = lambda mu, wh: (1 - S9.S2.fidelity(a, b, mu), abs(np.sum(wh / mu) - C0) / C0)
    return a, b, v, sc, lams, U, w, C0, L0, n_gap, err


def _mu(mu): return [round(float(x), 7) for x in np.sort(np.asarray(mu, float).ravel())]   # learned eigenvalues of one run, as stored


def _med(r):                                      # medians over the finite runs; "mu" (learned eigenvalues per run) is stored, not summarised
    return {k: float(np.nanmedian(x)) for k, x in r.items() if k != "mu"} | {"runs": {k: (x if k == "mu" else [float(y) for y in x]) for k, x in r.items()}}


def _proj(lams, w, T2, shots, n_c, err, reps, seed=900):
    """Projective estimator with an n_c-bit clock (stage 9 caps the clock at 6 bits; the cap is lifted here)."""
    keep, S9.n_c_for, bc2 = S9.n_c_for, (lambda l, T: n_c), S9.S2.bc2_learn; r = {"1-F": [], "eC": [], "T": [], "m": [], "dropped": [], "mu": []}
    def clean(*a, **kw):                          # a label that mixes several eigenvalues can empty its grid posterior (NaN): drop that label
        mu, x, counts = bc2(*a, **kw); ok = np.isfinite(mu); r["dropped"].append(int((~ok).sum())); return mu[ok], x, counts
    S9.S2.bc2_learn = clean
    for k in range(reps):
        mu, wh, T = S9.projective(lams, w, T2, shots, np.random.default_rng(seed + k)); f, c = err(mu, wh)
        r["1-F"].append(f); r["eC"].append(c); r["T"].append(T); r["m"].append(len(mu)); r["mu"].append(_mu(mu))
    S9.n_c_for, S9.S2.bc2_learn = keep, bc2; return r


def _seq(lams, w, T2, budget, sid, err, reps, delta=1e-3, seed=700):
    r = {"1-F": [], "eC": [], "T": [], "m": [], "copies": [], "unlabelled": [], "mu": []}
    for k in range(reps):
        mu, wh, st = S9.learn(lams, w, T2, budget, np.random.default_rng(seed + k), sid, delta); f, c = err(mu, wh)
        for key, val in zip(r, (f, c, st["T_tot"], len(mu), st["copies"], st["unlabelled"], _mu(mu))): r[key].append(val)
    return r


def run_compare(case):
    """Protocol of stage 9 (6-bit clock, 10^4 projective shots, sequential at the same time, with and without
    dephasing), then the same comparison without dephasing with a clock that resolves the smallest gap."""
    a, b, v, sc, lams, U, w, C0, L0, n_gap, err = _setup(case)
    proj, learned = S9.projective, []
    def recording(lams_, w_, T2, shots, rng, delta=None):
        run = proj(lams_, w_, T2, shots, rng, delta=delta)
        if delta is None: learned.append(_mu(run[0]))                  # the ten plain runs per noise setting, in the order of the repetitions
        return run
    S9.projective = recording; S9.S2.problems = lambda: {case: (a, b)}; _, res = S9._compare_one(case); S9.projective = proj
    for i, key in enumerate(res):                                       # T2=inf first, then T2=300: ten plain projective runs each
        row = next(v for k, v in res[key].items() if k.startswith("projective (")); n = len(row["runs"]["1-F"]); row["runs"]["mu"] = learned[i * n:(i + 1) * n]
    assert len(learned) == sum(len(next(v for k, v in res[key].items() if k.startswith("projective ("))["runs"]["1-F"]) for key in res)
    out = {"problem": sc | {"kappa": 1 / lams.min(), "min_gap": float(np.diff(lams).min())}, "6-bit clock": res}
    if n_gap > 6:
        S9.set_noise(np.inf); p = _proj(lams, w, np.inf, 10_000, n_gap, err, 10); T_ref = float(np.median(p["T"])); sid = S9.sig_id_for(lams, 2000.0)
        out[f"{n_gap}-bit clock"] = {"T_ref": T_ref, "sig_id": sid, "projective": _med(p), "sequential": _med(_seq(lams, w, np.inf, T_ref, sid, err, 10))}
        print(case, f"{n_gap}-bit clock", {m: {k: float(f"{x:.3g}") for k, x in out[f'{n_gap}-bit clock'][m].items() if k != "runs"} for m in ("projective", "sequential")}, flush=True)
    json.dump(out, open(f"results/stage10_compare.{case}.json", "w"), indent=1)


def run_qoi(case, reps=6):
    """Line flow (linear functional) and the DC quadratic form C = P^T theta at equal time, as _qoi_one of stage 9.
    The line is the one with the largest true flow (chosen in problem() from the classical solution).  base MVA x C
    approximates the series reactive consumption of the lines for lossless branches, unit voltages, small angles and
    nominal taps; the result keys keep the name C_MVAr."""
    S7 = importlib.util.module_from_spec(sp := importlib.util.spec_from_file_location("s7", "stage7_observable_first.py")); sp.loader.exec_module(S7)
    a, b, v, sc, lams, U, w, C0, L0, n_gap, err = _setup(case); out = {"problem": sc | {"C_MVAr": C0 * sc["C_MVAr"], "flow_MW": abs(L0) * sc["L_MW"], "C_scale": sc["C_MVAr"], "L_scale": sc["L_MW"]}}
    for T2 in (np.inf, 300.0):
        S7.S3.set_noise(T2); n6 = S9.n_c_for(lams, T2)
        for n_c in sorted({n6, n_gap} if not np.isfinite(T2) else {n6}):
            sid = S9.sig_id_for(lams, T2, S9.FLOOR6) if n_c == n6 else S9.sig_id_for(lams, 2000.0)
            for shots in (10_000, 160_000) if n_c <= 6 else (10_000,):
                p = {"eC": [], "eL": [], "T": []}; q = {"eC": [], "eL": [], "T": [], "copies": []}
                for k in range(reps):
                    S7.S2.T_ACC[0] = 0.0; C, _, L, _, _ = S7.spectral_qois(lams, U, b, v, n_c, T2, shots, np.random.default_rng(2000 + k))
                    p["eC"].append(abs(C - C0) / C0); p["eL"].append(abs(L - L0) / abs(L0)); p["T"].append(S7.S2.T_ACC[0])
                T_ref = float(np.median(p["T"]))
                for k in range(reps):
                    C, _, L, mu, wb, st = S9.qois(lams, U, b, v, T2, T_ref, np.random.default_rng(2500 + k), sid, "auto")
                    q["eC"].append(abs(C - C0) / C0); q["eL"].append(abs(L - L0) / abs(L0)); q["T"].append(st["T_used"]); q["copies"].append(st["copies"])
                key = f"T2={'inf' if not np.isfinite(T2) else T2}, shots={shots}, n_c={n_c}"; out[key] = {"T_tot": T_ref, "sig_id": sid, "projective": _med(p), "sequential": _med(q)}
                print(case, key, {m: {k: float(f"{x:.3g}") for k, x in out[key][m].items() if k != "runs"} for m in ("projective", "sequential")}, flush=True)
    json.dump(out, open(f"results/stage10_qoi.{case}.json", "w"), indent=1)


def run_scale(case, mult, reps=5):
    """Error against time without dephasing: budgets `mult` x the time of 10^4 shots with a 6-bit clock."""
    a, b, v, sc, lams, U, w, C0, L0, n_gap, err = _setup(case); S9.set_noise(np.inf)
    T1 = float(np.median(_proj(lams, w, np.inf, 10_000, 6, err, reps)["T"])); out = {"T_unit": T1, "mult": mult}
    out["projective, 6-bit clock"] = _med(_proj(lams, w, np.inf, 10_000 * mult, 6, err, reps))
    if n_gap > 6 and mult <= 4: out[f"projective, {n_gap}-bit clock"] = _med(_proj(lams, w, np.inf, 10_000 * mult, n_gap, err, reps))   # ponytail: the label sampler holds shots x 2^n_c x n_c floats, so longer clocks stop at 4x10^4 shots
    for sid in sorted({S9.sig_id_for(lams, np.inf, S9.FLOOR6), S9.sig_id_for(lams, 2000.0)}, reverse=True):
        out[f"sequential, sig_id={sid:.5f}"] = _med(_seq(lams, w, np.inf, mult * T1, sid, err, reps))
    for k, x in out.items():
        if isinstance(x, dict): print(case, f"x{mult}", k, {kk: float(f"{y:.3g}") for kk, y in x.items() if kk != "runs"}, flush=True)
    json.dump(out, open(f"results/stage10_scale.{case}.x{mult}.json", "w"), indent=1)


def _load(name, path):
    m = importlib.util.module_from_spec(sp := importlib.util.spec_from_file_location(name, path)); sp.loader.exec_module(m); return m


def _noise(T2, *mods):
    S9.set_noise(T2)
    for m in mods: m.S3.set_noise(T2)


def run_methods(case, reps=100, k0=0, k1=None):
    """Learning stages of the manuscript at 10^4 shots (protocol of its Table I): single-ancilla mixture estimators
    (Bayesian clock of the pilot, QMEGS-type, MM-QCELS), the QPE histogram of the hybrid solver, and the noiseless
    discretisation of standard HHL.  The total evolution time of every run is recorded.  Repetition k uses only the
    seeds 100+k and 900+k, so the repetitions k0..k1-1 can run in a process of their own (a part file, see merge_methods)."""
    a, b, v, sc, lams, U, w, C0, L0, n_gap, err = _setup(case); fid = lambda mu: 1 - S9.S2.fidelity(a, b, np.asarray(mu)) if len(mu) else 1.0
    S4 = _load("s4", "stage4_mmqcels_clockopt.py"); S4.S3.P = S4.P; S4.S3.S2 = S4.S2      # one noise object for stages 2-4
    P = S4.P; meas, T = P.measure, [0.0]
    def timed(lams_, w_, t, theta, shots, *x, **kw): T[0] += float(np.sum(np.abs(t))) * shots; return meas(lams_, w_, t, theta, shots, *x, **kw)
    P.measure = timed; raw = {"failures": {}}
    for T2 in (np.inf, 300.0):
        _noise(T2, S4); key = f"T2={'inf' if not np.isfinite(T2) else T2}"; R = raw[key] = {}
        def put(name, mu, t):                     # a run without a finite estimate is a failure: 1-F is NaN, the run is listed in "failures" and left out of the median
            R.setdefault(name, {"1-F": [], "T": [], "mu": []}); f = np.nan if mu is None or not np.all(np.isfinite(mu)) else fid(mu)
            if not np.isfinite(f): raw["failures"].setdefault(f"{key}, {name}", []).append([k, "non-finite estimate" if mu is not None else "fit did not converge"])
            R[name]["1-F"].append(f); R[name]["T"].append(t if np.isfinite(f) else np.nan); R[name]["mu"].append(_mu(mu) if np.isfinite(f) else [])
        def run(name, fn):
            T[0] = 0.0
            try: mu = fn()
            except Exception: mu = None
            put(name, mu, T[0])
        for k in range(k0, reps if k1 is None else k1):
            P.rng = np.random.default_rng(100 + k)
            def clock(): mu0, we, s1, ser = P.stage1a(lams, w, 1.0, len(lams)); return P.stage1b(lams, w, mu0, we, ser, shots_total=10_000 - s1)[0]
            run("Bayesian clock (mixture)", clock)
            rng = np.random.default_rng(900 + k)
            run("QMEGS-type (mixture)", lambda: S4.S3.qmegs_learn(lams, w, T2, 10_000, rng)[0])
            run("MM-QCELS (mixture)", lambda: S4.mm_qcels(lams, w, T2, 10_000, rng))
            for n_p in (6, 8, 10):
                if np.isfinite(T2) and 2**n_p * S9.T0 > 4 * T2: continue            # a register far longer than the coherence time is not run
                put(f"QPE histogram, {n_p} bits", S4.S2.qpe_learn(lams, w, n_p, T2, 10_000, rng)[0], 10_000 * (2**n_p - 1) * S9.T0)
    if k1 is None: _methods_write(case, [raw])
    else: json.dump(raw, open(f"results/stage10_methods.{case}.part{k0:03d}.json", "w"))


def _methods_write(case, parts):
    a, b, v, sc, lams, U, w, C0, L0, n_gap, err = _setup(case); keys = [k for k in parts[0] if k != "failures"]
    out = {"shots": 10_000, "reps": sum(len(p[keys[0]]["QMEGS-type (mixture)"]["1-F"]) for p in parts), "failures": {}}
    for p in parts:
        for m, f in p["failures"].items(): out["failures"].setdefault(m, []).extend(f)
    for key in keys:
        R = {m: {q: sum((p[key][m][q] for p in parts), []) for q in ("1-F", "T", "mu")} for m in parts[0][key]}
        out[key] = {m: _med(r) for m, r in R.items()} | {f"standard HHL, {n}-bit clock, noiseless": {"1-F": 1 - S9.S2.fidelity(a, b, np.maximum(np.round(lams * 2**n), 1) / 2**n)} for n in (6, 8, 10)}
        print(case, key, {m: (float(f"{r['1-F']:.3g}"), float(f"{r.get('T', 0):.3g}")) for m, r in out[key].items()}, flush=True)
    print(case, "failures", {m: len(f) for m, f in out["failures"].items()}, flush=True)
    json.dump(out, open(f"results/stage10_methods.{case}.json", "w"), indent=1)


def merge_methods(case, reps=100):
    """Joins the part files of run_methods in order of repetition and removes them."""
    files = sorted(glob.glob(f"results/stage10_methods.{case}.part*.json")); parts = [json.load(open(f)) for f in files]
    n = sum(len(p["T2=inf"]["QMEGS-type (mixture)"]["1-F"]) for p in parts); assert n == reps, f"{case}: {n} of {reps} repetitions present"
    _methods_write(case, parts)
    for f in files: os.remove(f)


def run_observables(case, reps=4):
    """Model-free baselines of the manuscript's observable table on the line flow and on C: the randomised Fourier
    functional and quantum Krylov subspace diagonalisation (best of the manuscript's parameter grid)."""
    S7b = _load("s7b", "stage7b_qks_baseline.py"); S7 = _load("s7", "stage7_observable_first.py")
    a, b, v, sc, lams, U, w, C0, L0, n_gap, err = _setup(case); N = len(lams); out = {}
    for T2 in (np.inf, 300.0):
        _noise(T2, S7, S7b)
        for shots in (10_000, 160_000):
            eF, tF = [], []
            for k in range(reps):
                Lf, tmax = S7.fourier_functional(lams, U, b, v, T2, shots, np.random.default_rng(2000 + k)); eF.append(abs(Lf - L0) / abs(L0)); tF.append(tmax)
            best = None
            for K, Delta, thr in ((N, 2.0, 1e-2), (N + 4, 2.0, 1e-2), (N + 4, 4.0, 1e-2), (N + 4, 4.0, 1e-3), (2 * N, 3.0, 1e-3)):
                eC, eL = [], []
                for k in range(reps):
                    C, _, L, dim = S7b.qks(lams, U, b, v, K, Delta, shots, np.random.default_rng(3000 + k), thr=thr); eC.append(abs(C - C0) / C0); eL.append(abs(L - L0) / abs(L0))
                cand = {"K": K, "Delta": Delta, "thr": thr, "eC": float(np.median(eC)), "eL": float(np.median(eL)), "max_time": (K - 1) * Delta}
                if best is None or cand["eC"] + cand["eL"] < best["eC"] + best["eL"]: best = cand
            key = f"T2={'inf' if not np.isfinite(T2) else T2}, shots={shots}"; out[key] = {"fourier": {"eL": float(np.median(eF)), "max_time": float(np.median(tF))}, "qks": best}
            print(case, key, out[key], flush=True)
    json.dump(out, open(f"results/stage10_observables.{case}.json", "w"), indent=1)


def run_backend(case, reps=10):
    """Solver with a real clock register (joint fidelity of stage 9): standard HHL without learning for several clock
    sizes, learned angles from the projective estimator, and the sequential spectrum with uniform, sine and DPSS clocks."""
    from scipy.signal.windows import dpss
    a, b, v, sc, lams, U, w, C0, L0, n_gap, err = _setup(case); out = {}; bf = lambda *x, **kw: S9.backend_fidelity(*x, **kw)[0]
    for T2 in (np.inf, 300.0):
        S9.set_noise(T2); n0 = S9.n_c_for(lams, T2); nmc = 256 if np.isfinite(T2) else 1; R = {}; put = lambda name, f: R.setdefault(name, []).append(f)
        for k in range(reps):
            rng = np.random.default_rng(1500 + k)
            for n in sorted({n0, 8, 10} if not np.isfinite(T2) else {n0, n0 + 1}):
                ly = 2 * np.pi * np.arange(2**n) / (2**n * S9.T0); put(f"standard HHL, {n}-bit clock", bf(a, b, n, T2, np.where(ly >= lams.min(), ly, np.nan), rng, nmc))
            mu_p, w_p, T_p = S9.projective(lams, w, T2, 10_000, rng); put(f"projective + {n0}-bit clock", bf(a, b, n0, T2, S9.lam_table(mu_p, 2**n0, S9.T0, 1), rng, nmc))
            mu, wh, _ = S9.learn(lams, w, T2, T_p, rng, S9.sig_id_for(lams, T2, S9.FLOOR6))
            for n in (n0, n0 + 1):
                M = 2**n; sine = np.sqrt(2 / M) * np.sin(np.pi * (np.arange(M) + 0.5) / M); dp = dpss(M, 2.0) / np.linalg.norm(dpss(M, 2.0))
                for wn, win, half in (("uniform", None, 1), ("sine", sine, 2), ("DPSS NW=2", dp, 2)):
                    put(f"sequential + {n}-bit {wn} clock", bf(a, b, n, T2, S9.lam_table(mu, M, S9.T0, half), rng, nmc, window=win))
        key = f"T2={'inf' if not np.isfinite(T2) else T2}"; out[key] = {m: {"F": float(np.median(x)), "runs": [float(y) for y in x]} for m, x in R.items()}
        print(case, key, {m: round(x["F"], 5) for m, x in out[key].items()}, flush=True)
    json.dump(out, open(f"results/stage10_backend.{case}.json", "w"), indent=1)


def run_poly(case, target=0.9999, d_max=199):
    """Smallest odd degree with F >= target under |p| <= 1 (stage 6): node-aware polynomial on learned nodes, the
    interval polynomial that only knows lambda_min, and its min-norm spectral correction on the learned nodes (which
    leaves |p| <= 1; tau is its subnormalisation).  Nodes: exact, projective as in stage 6, sequential with delta=1e-4."""
    S6 = _load("s6", "stage6_spectral_correction.py"); a, b, v, sc, lams, U, w, C0, L0, n_gap, err = _setup(case); S9.set_noise(np.inf)
    mu_p = np.sort(S9.S2.bc2_learn(lams, w, n_gap, np.inf, 10_000, np.random.default_rng(1300))[0]); mu_p = mu_p[np.isfinite(mu_p)]
    T_p = S9.projective(lams, w, np.inf, 10_000, np.random.default_rng(900))[2]; out = {"kappa": 1 / lams.min(), "target": target, "d_max": d_max}
    seq = S9.learn(lams, w, np.inf, T_p, np.random.default_rng(700), S9.sig_id_for(lams, 2000.0), 1e-4); near = lambda mu: np.array([w[np.argmin(np.abs(lams - x))] for x in mu])
    for label, (mu, wm) in {"exact spectrum": (lams, w), f"projective, {n_gap}-bit clock": (mu_p, near(mu_p)), "sequential, delta=1e-4": (seq[0], seq[1])}.items():
        keep = wm > 0; mu, wm = mu[keep], wm[keep]; om = np.sqrt(wm) / mu**2; res = {"nodes": len(mu), "node-aware": None, "interval": None, "interval + spectral correction": None}
        for d in range(3, d_max + 1, 2):
            n = (d + 1) // 2
            if res["node-aware"] is None and S6.fidelity_poly(a, b, S6.poly_from_odd(S6.lp_odd(n, nodes=mu, weights=om))) >= target: res["node-aware"] = d
            if res["interval"] is None or res["interval + spectral correction"] is None:
                c0 = S6.lp_odd(n, interval=mu.min())
                if res["interval"] is None and S6.fidelity_poly(a, b, S6.poly_from_odd(c0)) >= target: res["interval"] = d
                if res["interval + spectral correction"] is None:
                    try:
                        pc = S6.poly_from_odd(S6.spectral_correction(c0, mu, mu.min()))
                        if S6.fidelity_poly(a, b, pc) >= target: res["interval + spectral correction"] = d; res["tau of the correction"] = S6.tau(pc, lams)
                    except np.linalg.LinAlgError: pass
            if all(res[k] is not None for k in ("node-aware", "interval", "interval + spectral correction")): break
        out[label] = res; print(case, label, res, flush=True)
    json.dump(out, open(f"results/stage10_poly.{case}.json", "w"), indent=1)


def run_manifest():
    """SHA-256 of this script, of the grid data and of every stage-10 result file, with modification times (checked by
    paper/check_numbers.py together with the stage-9 manifest, which covers the other imported stage scripts; the Krylov
    baseline stage7b_qks_baseline.py is recorded here)."""
    import hashlib, glob, os, time
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); ent = lambda p: {"sha256": h(p), "mtime": os.path.getmtime(p)}
    res = sorted(p for p in glob.glob("results/stage10_*.json") if not p.endswith("manifest.json"))
    json.dump({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": {p: ent(p) for p in ("stage10_power_grid.py", "data/grid_cases.json", "stage7b_qks_baseline.py")}, "results": {p: ent(p) for p in res}},
              open("results/stage10_manifest.json", "w"), indent=1); print("stage-10 manifest:", len(res), "result files")


def table():
    """Markdown tables of the stage-10 results (medians; fidelity errors as 1-F, others relative)."""
    import glob
    g = lambda x: f"{x:.2g}"
    print("| system | setting | time | projective 1-F | sequential 1-F | sequential 1-F, finest sig_id | projective C err | sequential C err | sequential C err, finest sig_id |\n|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob("results/stage10_compare.*.json"), key=lambda f: int(f.split("case")[1].split(".")[0])):
        d = json.load(open(f)); case = f.split(".")[1]
        for key, r in d["6-bit clock"].items():
            pk = next(k for k in r if k.startswith("projective (")); sk = next(k for k in r if k.startswith("sequential (budget x1.0, delta=0.001"))
            fk = [k for k in r if k.startswith("sequential (budget x1.0, delta=0.001")][-1]; fine = (lambda q: g(r[fk][q]) if fk != sk else "same")
            print(f"| {case} | {pk[12:-1]}, {key} | {g(r[pk]['T'])} | {g(r[pk]['1-F'])} | {g(r[sk]['1-F'])} | {fine('1-F')} | {g(r[pk]['eC'])} | {g(r[sk]['eC'])} | {fine('eC')} |")
        for key in (k for k in d if k.endswith("-bit clock") and k != "6-bit clock"):
            r = d[key]; print(f"| {case} | {key}, T2=inf | {g(r['T_ref'])} | {g(r['projective']['1-F'])} | - | {g(r['sequential']['1-F'])} | {g(r['projective']['eC'])} | - | {g(r['sequential']['eC'])} |")
    print("\n| system | noise | method | time | 1-F |\n|---|---|---|---|---|")
    for f in sorted(glob.glob("results/stage10_methods.*.json"), key=lambda f: int(f.split("case")[1].split(".")[0])):
        d = json.load(open(f)); case = f.split(".")[1]; cmp_ = json.load(open(f"results/stage10_compare.{case}.json"))["6-bit clock"]
        for key in ("T2=inf", "T2=300.0"):
            for m, r in d[key].items(): print(f"| {case} | {key} | {m} | {g(r['T']) if 'T' in r else '-'} | {g(r['1-F'])} |")
            r = cmp_[key]; pk = next(k for k in r if k.startswith("projective (")); sks = [k for k in r if k.startswith("sequential (budget x1.0, delta=0.001")]
            print(f"| {case} | {key} | projective ({pk[12:17]}) | {g(r[pk]['T'])} | {g(r[pk]['1-F'])} |")
            for sk in sks: print(f"| {case} | {key} | sequential ({sk.split(', ')[-1][:-1]}) | {g(r[sk]['T'])} | {g(r[sk]['1-F'])} |")
    print("\n| system | setting | Fourier flow err | Krylov flow err | Krylov C err | projective flow err | sequential flow err | projective C err | sequential C err |\n|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob("results/stage10_observables.*.json"), key=lambda f: int(f.split("case")[1].split(".")[0])):
        d = json.load(open(f)); case = f.split(".")[1]; q = json.load(open(f"results/stage10_qoi.{case}.json"))
        for key, r in d.items():
            qk = next(k for k in q if k.startswith(key + ", n_c=")); x = q[qk]
            print(f"| {case} | {key} | {g(r['fourier']['eL'])} | {g(r['qks']['eL'])} | {g(r['qks']['eC'])} | {g(x['projective']['eL'])} | {g(x['sequential']['eL'])} | {g(x['projective']['eC'])} | {g(x['sequential']['eC'])} |")
    print("\n| system | noise | solver with a real clock | joint fidelity |\n|---|---|---|---|")
    for f in sorted(glob.glob("results/stage10_backend.*.json"), key=lambda f: int(f.split("case")[1].split(".")[0])):
        d = json.load(open(f)); case = f.split(".")[1]
        for key, R in d.items():
            for m, r in R.items(): print(f"| {case} | {key} | {m} | {r['F']:.4f} |")
    print("\n| system | kappa | nodes from | nodes | node-aware degree | interval degree | corrected interval degree (tau) |\n|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob("results/stage10_poly.*.json"), key=lambda f: int(f.split("case")[1].split(".")[0])):
        d = json.load(open(f)); case = f.split(".")[1]; dd = lambda x: f"> {d['d_max']}" if x is None else str(x)
        for key, r in d.items():
            if isinstance(r, dict): print(f"| {case} | {d['kappa']:.0f} | {key} | {r['nodes']} | {dd(r['node-aware'])} | {dd(r['interval'])} | {dd(r['interval + spectral correction'])}" + (f" ({g(r['tau of the correction'])})" if "tau of the correction" in r else "") + " |")
    print("\n| system | setting | projective flow err [MW] | sequential flow err [MW] | projective C err [MVAr] | sequential C err [MVAr] |\n|---|---|---|---|---|---|")
    for f in sorted(glob.glob("results/stage10_qoi.*.json"), key=lambda f: int(f.split("case")[1].split(".")[0])):
        d = json.load(open(f)); pr = d["problem"]; case = f.split(".")[1]
        for key, r in d.items():
            if key != "problem": print(f"| {case} ({pr['flow_MW']:.1f} MW, {pr['C_MVAr']:.1f} MVAr) | {key} | {g(r['projective']['eL'] * pr['flow_MW'])} | {g(r['sequential']['eL'] * pr['flow_MW'])} | {g(r['projective']['eC'] * pr['C_MVAr'])} | {g(r['sequential']['eC'] * pr['C_MVAr'])} |")
    print("\n| system | budget | estimator | time | components | 1-F | C err |\n|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob("results/stage10_scale.*.json"), key=lambda f: (int(f.split("case")[1].split(".")[0]), int(f.split(".x")[1].split(".")[0]))):
        d = json.load(open(f)); case = f.split(".")[1]
        for key, r in d.items():
            if isinstance(r, dict): print(f"| {case} | x{d['mult']} | {key} | {g(r['T'])} | {r['m']:.0f} | {g(r['1-F'])} | {g(r['eC'])} |")


if __name__ == "__main__":
    if len(sys.argv) == 1: demo(); describe()
    elif sys.argv[1] == "compare": run_compare(sys.argv[2])
    elif sys.argv[1] == "qoi": run_qoi(sys.argv[2])
    elif sys.argv[1] == "scale": run_scale(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "methods": run_methods(sys.argv[2], *map(int, sys.argv[3:]))   # methods <case> [reps [k0 k1]]
    elif sys.argv[1] == "methods-merge": merge_methods(sys.argv[2], *map(int, sys.argv[3:]))
    elif sys.argv[1] in ("observables", "backend", "poly"): globals()["run_" + sys.argv[1]](sys.argv[2])
    elif sys.argv[1] == "table": table()
    elif sys.argv[1] == "manifest": run_manifest()
