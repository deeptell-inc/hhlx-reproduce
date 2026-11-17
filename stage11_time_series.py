"""Stage 11: the estimators of stages 2-9 on the readout training of a reservoir computer for nonlinear time series.

Predictor: echo state network with N tanh nodes (fixed random couplings), linear readout trained by ridge regression
on standardised node states.  Linear system: (G + alpha 1) W = g with G = Z^T Z / n, g = Z^T (y - mean y) / n;
A = (G + alpha 1)/(lam_max(G) + alpha), |b> = g/|g|, and alpha = lam_max(G)/(KAPPA - 1) bounds the condition number of A by KAPPA.
Series (targets): NARMA2 driven by uniform noise (the output at the same step), Mackey-Glass with delay 17 (six
samples ahead) and the x coordinate of the Lorenz system (five integration steps ahead).  A learned spectrum is
turned into forecasts as the hybrid solver does it: every eigencomponent of |b> is divided by the learned eigenvalue
nearest to its own (pilot_bayes_clock.hhl_from_spectrum, here without normalising), so the forecast error measures
the learning stage with an ideal back end.  Nothing of stages 2-10 is modified; stage 10 supplies the protocols and
its result files are written under the prefix stage11_.  Run from the repository root:
    python stage11_time_series.py                    # self-check and a table of the problems
    python stage11_time_series.py compare lorenz     # one OS process per series; writes results/stage11_compare.<series>.json
    python stage11_time_series.py methods lorenz 100 0 10, methods-merge lorenz 100     # baselines, as in stage 10
    python stage11_time_series.py scale lorenz 10    # no dephasing, ten times the shots (projective) and the time (sequential), ten runs
    python stage11_time_series.py baselines lorenz 10 inf 0 1, baselines-merge lorenz 10 inf   # the baselines within the same time budget (after `scale` and `methods-merge`)
    python stage11_time_series.py dephased lorenz 10 300, baselines lorenz 10 300 0 1, ...    # the same budget with dephasing (after `compare`)
    python stage11_time_series.py forecast           # forecast errors of every stored run; writes results/stage11_forecast.json
    python stage11_time_series.py manifest           # hashes of code and results; sh run_stage11_all.sh runs all of it
"""
import os, sys, glob, json, types, builtins, importlib.util, functools, numpy as np
S10 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s10", "stage10_power_grid.py")); spec.loader.exec_module(S10)
S9 = S10.S9
N, WASH, NTR, NTE, KAPPA = 16, 200, 2000, 1000, 100.0     # nodes, discarded steps, training steps, test steps, bound on the condition number of A
RHO, S_IN = 0.9, 1.0                                      # spectral radius of the couplings, input scale


def narma2(n):
    rng = np.random.default_rng(1); u = rng.uniform(0, 0.5, n + 1); y = np.zeros(n + 1)
    for t in range(1, n): y[t + 1] = 0.4 * y[t] + 0.4 * y[t] * y[t - 1] + 0.6 * u[t]**3 + 0.1
    return u[1:n + 1], y[1:n + 1], y[:n]                  # input, target (it depends on the input up to the step before), previous target


def _rk4(f, s, dt): k1 = f(s); k2 = f(s + dt * k1 / 2); k3 = f(s + dt * k2 / 2); k4 = f(s + dt * k3); return s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def mackey_glass(n, ahead=6, tau=17.0, dt=0.1, sub=10):
    h = int(round(tau / dt)); x = np.full(h + (n + ahead) * sub + 1, 1.2); x[:h] += 0.1 * np.sin(np.arange(h))
    for t in range(h, len(x) - 1):                        # the delayed value is held over the step
        xd = x[t - h]; x[t + 1] = _rk4(lambda s: 0.2 * xd / (1 + xd**10) - 0.1 * s, x[t], dt)
    s = x[h::sub]; return s[:n], s[ahead:n + ahead], s[:n]


def lorenz(n, ahead=5, dt=0.02):
    f = lambda s: np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]]); s = np.ones(3); x = []
    for t in range(1000 + n + ahead): s = _rk4(f, s, dt); x.append(s[0] / 20)
    x = np.array(x[1000:]); return x[:n], x[ahead:n + ahead], x[:n]


SERIES = {"narma2": narma2, "mackey-glass": mackey_glass, "lorenz": lorenz}


def ridge(X, y, last):
    """Standardised training and test features, targets, ridge system and its scales (stage 12 reuses it for other features)."""
    tr, te = slice(WASH, WASH + NTR), slice(WASH + NTR, None)
    m, s, ym = X[tr].mean(0), X[tr].std(0), y[tr].mean(); Z, Zt = (X[tr] - m) / s, (X[te] - m) / s
    G = Z.T @ Z / NTR; g = Z.T @ (y[tr] - ym) / NTR; lam = np.linalg.eigvalsh(G).max(); alpha = lam / (KAPPA - 1)
    return {"Zt": Zt, "yt": y[te], "ym": ym, "G": G, "g": g, "alpha": alpha, "scale": np.linalg.norm(g) / (lam + alpha),
            "a": (G + alpha * np.eye(X.shape[1])) / (lam + alpha), "b": g / np.linalg.norm(g), "persistence": float(np.mean((last[te] - y[te])**2) / np.var(y[te]))}


@functools.lru_cache(None)
def system(case):
    """The ridge system of the echo state network for one series."""
    u, y, last = SERIES[case](WASH + NTR + NTE); rng = np.random.default_rng(2)
    W = rng.normal(size=(N, N)); W *= RHO / np.abs(np.linalg.eigvals(W)).max(); w_in = rng.uniform(-S_IN, S_IN, N); bias = rng.uniform(-0.5, 0.5, N)
    X = np.zeros((len(u), N)); x = np.zeros(N); u0 = u - u[:WASH + NTR].mean()
    for t in range(len(u)): x = np.tanh(W @ x + w_in * u0[t] + bias); X[t] = x
    return ridge(X, y, last)


def problem(case):
    """(a, b, v, scales) in the form of stage 10; v is the state of the first test step (one forecast is a linear functional)."""
    p = system(case); v = p["Zt"][0] + 0j
    return p["a"], p["b"] + 0j, v / np.linalg.norm(v), {"n": N, "alpha": p["alpha"], "W_scale": p["scale"]}


def nmse(case, W): p = system(case); return float(np.mean((p["Zt"] @ W + p["ym"] - p["yt"])**2) / np.var(p["yt"]))


def readout(case, mu):
    """Readout weights built from learned eigenvalues `mu` (zero when nothing was learned: the forecast is the training mean)."""
    p = system(case); lams, U = np.linalg.eigh(p["a"]); mu = np.asarray(mu, float); mu = mu[np.isfinite(mu) & (mu > 0)]
    return p["scale"] * U @ ((U.T @ p["b"]) / mu[np.argmin(np.abs(mu[None, :] - lams[:, None]), axis=1)]) if len(mu) else np.zeros(N)


def forecast_nmse(case, mu): return nmse(case, readout(case, mu))     # normalised mean squared error on the test steps


def forecast_dev(case, mu):
    """Mean squared difference between these forecasts and those of the exact ridge readout, in units of the target variance."""
    p = system(case); return float(np.mean((p["Zt"] @ (readout(case, mu) - readout(case, np.linalg.eigvalsh(p["a"]))))**2) / np.var(p["yt"]))


def describe():
    for case in SERIES:
        p = system(case); lams, U = np.linalg.eigh(p["a"]); w = (U.T @ p["b"])**2; x = w / lams**2; x /= x.sum(); best = nmse(case, np.linalg.lstsq(p["G"], p["g"], rcond=None)[0])
        print(f"{case}: N={N} kappa={1 / lams.min():.0f}, eigenvalues below the 6-bit bin (0.0195): {int((lams < 0.0195).sum())} holding {w[lams < 0.0195].sum():.2g} of |b> and {x[lams < 0.0195].sum():.2g} of |x>; "
              f"forecast NMSE: ridge at this kappa {forecast_nmse(case, lams):.3g}, without regularisation {best:.3g}, repeating the last known value {p['persistence']:.3g}")
        print("   eigenvalues", np.array2string(lams, precision=4, max_line_width=250), "\n   weights    ", np.array2string(w, precision=2, max_line_width=250))


def demo():                                       # self-check: the exact spectrum reproduces the ridge readout and its forecasts
    for case in SERIES:
        p = system(case); a, b, v, sc = problem(case); lams, U = np.linalg.eigh(a); W = np.linalg.solve(p["G"] + p["alpha"] * np.eye(N), p["g"])
        assert np.isclose(lams.max(), 1) and 0.9 * KAPPA < 1 / lams.min() <= KAPPA + 1e-9 and np.allclose(sc["W_scale"] * np.linalg.solve(a, b).real, W)
        assert np.isclose(forecast_nmse(case, lams), nmse(case, W)) and np.isclose(S9.S2.fidelity(a, b, lams), 1)
        assert forecast_nmse(case, lams) < min(p["persistence"], 0.5) and np.isclose(forecast_nmse(case, []), nmse(case, np.zeros(N))) and forecast_dev(case, lams) < 1e-20
    print("stage 11 self-check passed")


S10.problem = problem                              # stage 10 runs its protocols on these problems and writes under the prefix stage11_
SIG_MIN = 3e-4                                     # finest identification resolution (0.6/2000, as stage 10 uses with its long clocks):
_sig = S9.sig_id_for                               # the eigenvalues at the regularisation floor agree to 1e-7 and are not to be separated
S9.sig_id_for = lambda lams, T2, floor=0.0: _sig(lams, T2, max(floor, SIG_MIN))
S10.open = lambda path, *a, **kw: builtins.open(path.replace("stage10_", "stage11_"), *a, **kw)
S10.glob = types.SimpleNamespace(glob=lambda pat: glob.glob(pat.replace("stage10_", "stage11_")))


def _tag(T2): return "" if not np.isfinite(T2) else f".T2_{T2:g}"      # file-name suffix and row key of a noise setting
def _key(T2): return f"T2={'inf' if not np.isfinite(T2) else T2}"


def qmegs(P, lams, w, T2, shots, rng):
    """qmegs_learn of stage 3 with its spectrum evaluated in blocks of the frequency grid: the same estimate with bounded
    memory (the original holds 12001 x shots/2 complex numbers).  run_baselines asserts the agreement."""
    T = min(T2, 2000.0) if np.isfinite(T2) else 2000.0; N = shots // 2; tn = np.abs(rng.normal(0, T, N)); con = P.CONTRAST(tn)
    re = np.array([2 * P.measure(lams, w, t, 0.0, 1) - 1 for t in tn]) / con; im = np.array([2 * P.measure(lams, w, t, -np.pi / 2, 1) - 1 for t in tn]) / con
    Z = re + 1j * im; x = np.linspace(0, 1.2, 12001); G = np.concatenate([np.real(np.exp(1j * np.outer(xs, tn)) @ np.conj(Z)) for xs in np.array_split(x, 400)]) / N
    peaks = []
    for i in np.argsort(-G):
        if all(abs(x[i] - q) > 0.02 for q in peaks): peaks.append(x[i])
        if len(peaks) == len(lams): break
    return np.array(sorted(peaks))


def run_dephased(case, mult, T2, reps=10):
    """`scale` with dephasing, at the same budget (mult x the time of 10^4 projective shots without dephasing): the
    projective estimator gets the shots that keep every run within the budget, the sequential one the budget itself."""
    a, b, v, sc, lams, U, w, C0, L0, n_gap, err = S10._setup(case); S9.set_noise(T2); n_c = S9.n_c_for(lams, T2)
    budget = mult * json.load(open(f"results/stage11_scale.{case}.x{mult}.json"))["T_unit"]
    t1 = next(r for k, r in json.load(open(f"results/stage11_compare.{case}.json"))["6-bit clock"][_key(T2)].items() if k.startswith("projective ("))["T"]
    shots = int(0.97e4 * budget / t1); p = S10._proj(lams, w, T2, shots, n_c, err, reps)
    while max(p["T"]) > budget: shots = int(0.97 * shots); p = S10._proj(lams, w, T2, shots, n_c, err, reps)
    out = {"budget": budget, "mult": mult, "noise": _key(T2), f"projective, {n_c}-bit clock": S10._med(p) | {"shots": shots}}
    for sid in sorted({S9.sig_id_for(lams, T2, S9.FLOOR6), S9.sig_id_for(lams, T2)}, reverse=True):
        out[f"sequential, sig_id={sid:.5f}"] = S10._med(S10._seq(lams, w, T2, budget, sid, err, reps))
    assert all(max(r["runs"]["T"]) <= budget * (1 + 1e-9) for r in out.values() if isinstance(r, dict))
    for k, x in out.items():
        if isinstance(x, dict): print(case, f"x{mult}", _key(T2), k, {kk: float(f"{y:.3g}") for kk, y in x.items() if kk != "runs"}, flush=True)
    json.dump(out, open(f"results/stage11_scale.{case}.x{mult}{_tag(T2)}.json", "w"), indent=1)


def run_baselines(case, mult, T2=np.inf, k0=0, k1=10):
    """Baselines of `methods` within the time budget of `scale` (mult x the time of 10^4 projective shots without dephasing):
    every baseline gets the largest number of shots found for which its total evolution time stays within the budget.
    Repetition k uses only the seeds 100+k and 900+k; the repetitions k0..k1-1 go to a part file (merge_baselines)."""
    a, b, v, sc, lams, U, w, C0, L0, n_gap, err = S10._setup(case); budget = mult * json.load(open(f"results/stage11_scale.{case}.x{mult}.json"))["T_unit"]
    T1 = json.load(open(f"results/stage11_methods.{case}.json"))[_key(T2)]                                   # times at 10^4 shots
    S4 = S10._load("s4", "stage4_mmqcels_clockopt.py"); S4.S3.P = S4.P; S4.S3.S2 = S4.S2; P = S4.P; meas, T = P.measure, [0.0]   # one noise object for stages 2-4
    def timed(lams_, w_, t, theta, shots, *x, **kw): T[0] += float(np.sum(np.abs(t))) * shots; return meas(lams_, w_, t, theta, shots, *x, **kw)
    P.measure = timed; S10._noise(T2, S4); est = []
    for f in (lambda r: qmegs(P, lams, w, T2, 2000, r), lambda r: S4.S3.qmegs_learn(lams, w, T2, 2000, r)[0]): P.rng = np.random.default_rng(6); est.append(f(np.random.default_rng(5)))
    assert np.allclose(*est); rng = np.random.default_rng(99); P.rng = np.random.default_rng(99)
    def clock(s): mu0, we, s1, ser = P.stage1a(lams, w, 1.0, len(lams)); return P.stage1b(lams, w, mu0, we, ser, shots_total=s - s1)[0]
    fns = {"Bayesian clock (mixture)": clock, "QMEGS-type (mixture)": lambda s: qmegs(P, lams, w, T2, s, rng), "MM-QCELS (mixture)": lambda s: S4.mm_qcels(lams, w, T2, s, rng)}
    def go(name, s):                              # one run: learned eigenvalues (None when the fit fails) and its evolution time
        T[0] = 0.0
        try: mu = fns[name](int(s))
        except Exception: mu = None
        return mu, T[0]
    shots = {name: 0.97e4 * budget / T1[name]["T"] for name in fns}; fitted = set(); R = {}      # time proportional to shots, as a first guess
    def put(name, mu, t, s):                      # a run without a finite estimate has 1-F = NaN and no eigenvalues, and is left out of the medians
        r = R.setdefault(name, {"1-F": [], "T": [], "shots": [], "mu": []}); ok = mu is not None and len(mu) and np.all(np.isfinite(mu))
        for key, val in zip(r, (1 - S9.S2.fidelity(a, b, np.asarray(mu)) if ok else np.nan, t if ok else np.nan, int(s), S10._mu(mu) if ok else [])): r[key].append(val)
    for k in range(k0, k1):
        P.rng = np.random.default_rng(100 + k); rng = np.random.default_rng(900 + k)
        for name in fns:
            mu, t = go(name, shots[name])
            if not t <= budget and name not in fitted:      # not proportional (Bayesian clock without dephasing: its times double per round): bisection in log(shots), then the run again
                P.rng = np.random.default_rng(99); lo, hi = 1e4, shots[name]; fitted.add(name)
                for _ in range(8): s = np.sqrt(lo * hi); lo, hi = (s, hi) if go(name, s)[1] <= budget else (lo, s)
                shots[name] = lo; P.rng = np.random.default_rng(100 + k); mu, t = go(name, lo)
            while not t <= budget: shots[name] *= 0.9; mu, t = go(name, shots[name])
            put(name, mu, t, shots[name])
        for n_p in (6, 8, 10):
            if np.isfinite(T2) and 2**n_p * S9.T0 > 4 * T2: continue                # as in stage 10: a register far longer than the coherence time is not run
            s = int(budget // ((2**n_p - 1) * S9.T0)); put(f"QPE histogram, {n_p} bits", S4.S2.qpe_learn(lams, w, n_p, T2, s, rng)[0], s * (2**n_p - 1) * S9.T0, s)
    assert all(np.nanmax(r["T"]) <= budget for r in R.values() if np.isfinite(r["T"]).any())
    json.dump({"budget": budget, "runs": R}, open(f"results/stage11_baselines.{case}.x{mult}{_tag(T2)}.part{k0:02d}.json", "w"))


def merge_baselines(case, mult, T2=np.inf, reps=10):
    """Joins the part files of run_baselines in order of repetition and removes them."""
    files = sorted(glob.glob(f"results/stage11_baselines.{case}.x{mult}{_tag(T2)}.part*.json")); parts = [json.load(open(f)) for f in files]
    R = {name: {q: sum((p["runs"][name][q] for p in parts), []) for q in ("1-F", "T", "shots", "mu")} for name in parts[0]["runs"]}
    assert all(len(r["1-F"]) == reps for r in R.values()), f"{case}: {[len(r['1-F']) for r in R.values()]} of {reps} repetitions present"
    out = {"budget": parts[0]["budget"], "mult": mult, "noise": _key(T2)} | {name: S10._med(r) for name, r in R.items()}
    for name, r in R.items(): print(case, f"x{mult}", _key(T2), name, {q: float(f"{out[name][q]:.3g}") for q in ("1-F", "T", "shots")}, "failures", int(np.isnan(r["1-F"]).sum()), flush=True)
    json.dump(out, open(f"results/stage11_baselines.{case}.x{mult}{_tag(T2)}.json", "w"), indent=1)
    for f in files: os.remove(f)


def run_forecast():
    """Forecast error of every stored learning run of `compare` and `methods` (median over the runs with a finite estimate)."""
    out = {}; g = lambda x: f"{x:.2g}"
    print("| series | noise | learning stage | runs | time | 1-F | forecast NMSE | difference from the ridge forecast | runs with NMSE > 1 |\n|---|---|---|---|---|---|---|---|---|")
    for case in SERIES:
        p = system(case); a = p["a"]; lams = np.linalg.eigvalsh(a); rows = []; out[case] = {"ridge at this kappa": forecast_nmse(case, lams), "persistence": p["persistence"], "mean of the training targets": forecast_nmse(case, [])}
        cmp_ = json.load(open(f"results/stage11_compare.{case}.json")); meth = json.load(open(f"results/stage11_methods.{case}.json")) if glob.glob(f"results/stage11_methods.{case}.json") else {}
        for key, res in cmp_["6-bit clock"].items():
            for name, r in meth.get(key, {}).items():
                if "runs" in r: rows.append((key, name, [m for m in r["runs"]["mu"] if m], r["T"], r["1-F"]))
                else: n = int(name.split(", ")[1].split("-")[0]); rows.append((key, name, [list(np.maximum(np.round(lams * 2**n), 1) / 2**n)], None, r["1-F"]))
            for name, r in res.items():
                mus = r["runs"].get("mu") or [rec["mu"] for rec in r.get("records", [])]
                if mus: rows.append((key, name, mus, r["T"], r["1-F"]))
        for key in (k for k in cmp_ if k.endswith("-bit clock") and k != "6-bit clock"):
            for m in ("projective", "sequential"): rows.append(("T2=inf", f"{m}, {key}", cmp_[key][m]["runs"]["mu"], cmp_[key][m]["T"], cmp_[key][m]["1-F"]))
        for fn in sorted(glob.glob(f"results/stage11_scale.{case}.x*.json") + glob.glob(f"results/stage11_baselines.{case}.x*.json")):   # no dephasing; time budget multiplied
            sc = json.load(open(fn)); rows += [(sc.get("noise", "T2=inf"), f"{name}, x{sc['mult']} time", [m for m in r["runs"]["mu"] if m], r["T"], r["1-F"]) for name, r in sc.items() if isinstance(r, dict)]
        for key, name, mus, T, f in rows:
            e = [forecast_nmse(case, m) for m in mus]; dv = [forecast_dev(case, m) for m in mus]; bad = int(np.sum(np.array(e) > 1))
            out[case][f"{key}, {name}"] = {"runs": len(e), "T": T, "1-F": f, "NMSE": float(np.median(e)), "dev": float(np.median(dv)), "NMSE>1": bad, "NMSE_runs": e, "dev_runs": dv}
            print(f"| {case} | {key} | {name} | {len(e)} | {g(T) if T else '-'} | {g(f)} | {g(np.median(e))} | {g(np.median(dv))} | {bad} |")
    json.dump(out, open("results/stage11_forecast.json", "w"), indent=1)


def run_manifest():
    """SHA-256 of this script and of every stage-11 result file, with modification times (checked by paper/check_numbers.py;
    the manifests of stages 9 and 10 cover the imported scripts)."""
    import hashlib, time
    ent = lambda p: {"sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "mtime": os.path.getmtime(p)}
    res = sorted(p for p in glob.glob("results/stage11_*.json") if not p.endswith("manifest.json") and ".part" not in p)
    json.dump({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": {"stage11_time_series.py": ent("stage11_time_series.py")}, "results": {p: ent(p) for p in res}},
              open("results/stage11_manifest.json", "w"), indent=1); print("stage-11 manifest:", len(res), "result files")


if __name__ == "__main__":
    if len(sys.argv) == 1: demo(); describe()
    elif sys.argv[1] == "compare": S10.run_compare(sys.argv[2])
    elif sys.argv[1] == "methods": S10.run_methods(sys.argv[2], *map(int, sys.argv[3:]))
    elif sys.argv[1] == "methods-merge": S10.merge_methods(sys.argv[2], *map(int, sys.argv[3:]))
    elif sys.argv[1] == "scale": S10.run_scale(sys.argv[2], int(sys.argv[3]), 10)
    elif sys.argv[1] == "dephased": run_dephased(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), *map(int, sys.argv[5:]))
    elif sys.argv[1] == "baselines": run_baselines(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), *map(int, sys.argv[5:]))     # baselines <series> <mult> <T2|inf> [k0 k1]
    elif sys.argv[1] == "baselines-merge": merge_baselines(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
    elif sys.argv[1] == "forecast": run_forecast()
    elif sys.argv[1] == "manifest": run_manifest()
