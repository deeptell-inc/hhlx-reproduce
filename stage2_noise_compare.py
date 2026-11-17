"""Stage 2: coherence-limited comparison of eigenvalue-learning front ends for hybrid HHL.

Noise model (calibrated, shared by all methods): contrast c(t) = (1-p_d) exp(-t/T2) on every
controlled-evolution of duration t (dephasing of the control qubit), p_d = 0.02 depolarising.

Methods (all feed the same idealised method-A HHL back end: exact tagging, learned angles):
  BC  : Bayesian-clock (pilot stage 1a + 1b), evolution time capped at T2.
  QPE : hybrid-HHL learning stage [184]: n_p-bit QPE sampled S times, eigenvalues read off the
        histogram peaks (resolution lam_max/2^n_p; coherent time 2^n_p t0, so n_p is limited by T2).
  BC2 : coarse-QPE-conditioned Bayesian clock: each shot runs a short n_c-bit QPE whose clock
        measurement projects the register onto ONE eigencomponent (prob w_j, information per shot
        ~ w_j instead of w_j^2), then a single-ancilla Hadamard test at adaptive time t on the
        collapsed state updates a 1-D grid posterior for that component (Wiebe-Granade rule
        t = 1/sigma, theta = -mu t - pi/2).  Coherent time per shot = 2^n_c t0 + t, both <= T2.
  REF : noiseless standard-HHL discretisation with n_p clock bits (no shots, no noise) -- reference.
Problems: P5 random 4x4 (seed 7), P3 1-D Poisson N=4 and N=8, P4 symmetric Toeplitz N=8.
Output: results/stage2_results.json, results/stage2_fidelity.png
"""
import json, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("pilot", "pilot_bayes_clock.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

PD, SHOTS = 0.02, 10_000
T_ACC = [0.0]                                   # accumulated evolution time of the labelled estimator (read by stage 9)


def problems():
    out = {}
    P.rng = np.random.default_rng(7); a, b = P.random_problem(); out["P5 random 4x4"] = (a, b)
    for n in (4, 8):
        a = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1); a = a / np.linalg.eigvalsh(a).max()
        b = np.random.default_rng(1).normal(size=n) + 0j; out[f"P3 Poisson N={n}"] = (a, b / np.linalg.norm(b))
    r = np.random.default_rng(2); c = r.normal(size=8); c[0] = 8.0        # diagonally dominant Toeplitz
    a = np.array([[c[abs(i - j)] for j in range(8)] for i in range(8)]); a = a / np.linalg.eigvalsh(a).max()
    b = r.normal(size=8) + 0j; out["P4 Toeplitz N=8"] = (a, b / np.linalg.norm(b))
    return out


def fidelity(a, b, lam_learned):
    x = np.linalg.solve(a, b); x /= np.linalg.norm(x)
    return float(abs(np.vdot(x, P.hhl_from_spectrum(a, b, lam_learned)))**2)


def qpe_learn(lams, w, n_p, T2, shots, rng):
    """Sample the n_p-bit QPE outcome of a mixture input under control-qubit dephasing."""
    M, t0 = 2**n_p, 2 * np.pi / 1.25                         # lam in (0,1] -> phase in (0, 0.8*2pi)
    k = np.arange(M); bits = ((k[:, None] >> np.arange(n_p)) & 1)            # (M, n_p)
    dur = 2.0**np.arange(n_p) * t0                             # duration of controlled-U^(2^q)
    j = rng.choice(len(lams), size=shots, p=w)
    xi = rng.normal(0, np.sqrt(2 * dur / T2), size=(shots, n_p)) if np.isfinite(T2) else np.zeros((shots, n_p))
    phase = k[None, :] * (lams[j] * t0)[:, None] + (bits[None, :, :] * xi[:, None, :]).sum(-1)   # (shots, M)
    amp = np.exp(1j * phase) / np.sqrt(M)                      # (shots, M) clock amplitudes before inverse QFT
    y_amp = np.fft.fft(amp, axis=1) / np.sqrt(M)               # inverse-QFT (sign convention irrelevant for |.|)
    prob = np.abs(y_amp)**2; prob /= prob.sum(1, keepdims=True)
    prob = (1 - PD) * prob + PD / M
    y = np.array([rng.choice(M, p=pr) for pr in prob])
    counts = np.bincount(y, minlength=M)
    peaks = []                                                 # greedy peak picking, m = number of eigenvalues
    order = np.argsort(-counts)
    for yy in order:
        if counts[yy] < 3: break
        if all(min(abs(yy - q), M - abs(yy - q)) > 2 for q in peaks): peaks.append(yy)
        if len(peaks) == len(lams): break
    lam = []
    for yy in peaks:                                           # centroid over +-1 bins
        nb = [(yy + d) % M for d in (-1, 0, 1)]; cnt = counts[nb]
        lam.append(float(np.dot([yy - 1, yy, yy + 1], cnt) / cnt.sum() * 2 * np.pi / (M * t0)))
    return np.array(lam), 2**n_p * t0


def qpe_sample_labels(lams, w, n_c, T2, shots, rng):
    """Per-shot coarse QPE: returns true component j (register collapse) and the clock outcome y."""
    M, t0 = 2**n_c, 2 * np.pi / 1.25
    k = np.arange(M); bits = ((k[:, None] >> np.arange(n_c)) & 1); dur = 2.0**np.arange(n_c) * t0
    j = rng.choice(len(lams), size=shots, p=w)
    xi = rng.normal(0, np.sqrt(2 * dur / T2), size=(shots, n_c)) if np.isfinite(T2) else np.zeros((shots, n_c))
    phase = k[None, :] * (lams[j] * t0)[:, None] + (bits[None, :, :] * xi[:, None, :]).sum(-1)
    prob = np.abs(np.fft.fft(np.exp(1j * phase) / np.sqrt(M), axis=1) / np.sqrt(M))**2
    prob = (1 - PD) * prob / prob.sum(1, keepdims=True) + PD / M
    y = np.array([rng.choice(M, p=pr) for pr in prob])
    T_ACC[0] += shots * (M - 1) * t0
    return j, y, M, t0


def bc2_learn(lams, w, n_c, T2, shots, rng, n_grid=6001, eps=0.05, true_contrast=None, t_cap=None, return_counts=False, delta=None):
    """true_contrast: physics (defaults to the calibrated P.CONTRAST used by the estimator)."""
    true_contrast = true_contrast or P.CONTRAST
    j_true, y, M, t0 = qpe_sample_labels(lams, w, n_c, T2, shots, rng)
    counts = np.bincount(y, minlength=M)
    peaks = []                                                 # coarse peaks = component labels
    for yy in np.argsort(-counts):
        if counts[yy] < 3: break
        if all(min(abs(yy - q), M - abs(yy - q)) > 1 for q in peaks): peaks.append(yy)
        if len(peaks) == len(lams): break
    peaks = np.array(peaks); m = len(peaks)
    lam0 = np.array([np.dot([p - 1, p, p + 1], counts[[(p - 1) % M, p, (p + 1) % M]]) / counts[[(p - 1) % M, p, (p + 1) % M]].sum() for p in peaks]) * 2 * np.pi / (M * t0)
    sig0 = 2 * np.pi / (M * t0)                               # coarse resolution = prior width
    grids = [np.linspace(0, 1.2, n_grid) for _ in range(m)]   # per-component grids (re-centred as they narrow)
    post = [np.exp(-0.5 * ((grids[c] - lam0[c]) / sig0)**2) for c in range(m)]
    post = [p / p.sum() for p in post]
    mu, sig = lam0.copy(), np.full(m, sig0)
    if t_cap is None: t_cap = T2 if np.isfinite(T2) else 2000.0   # refinement-time cap: T2, or 2000 (= mixture baselines' window) without dephasing
    for jt, yy in zip(j_true, y):                              # same shots: refine the collapsed component
        d = np.abs(((yy - peaks + M / 2) % M) - M / 2); c = int(np.argmin(d))   # label from the clock outcome
        if d[c] > 1: continue
        if delta is not None and sig[c] <= delta * mu[c]: continue   # stage-9 ablation: stop refining at the target (label-only shot)
        t = min(0.6 / max(sig[c], 1e-12), t_cap)              # conservative Wiebe-Granade rule (phase-slip safety)
        theta = -mu[c] * t - np.pi / 2; T_ACC[0] += t
        con_true, con = true_contrast(t), P.CONTRAST(t)
        p_true = 0.5 + 0.5 * con_true * np.cos(lams[jt] * t + theta)       # register is |u_{j_true}>
        dd = rng.uniform() < p_true
        g = grids[c]
        like = 0.5 + 0.5 * con * np.cos(g * t + theta); like = like if dd else 1 - like
        like = (1 - eps) * like + eps / 2
        new = post[c] * like
        if new.sum() <= 1e-300: continue
        post[c] = new / new.sum(); mu[c] = post[c] @ g; sig[c] = np.sqrt(post[c] @ (g - mu[c])**2)
        if sig[c] < 10 * (g[1] - g[0]):                        # re-grid around the posterior (Gaussian carry-over)
            grids[c] = np.linspace(mu[c] - 25 * sig[c], mu[c] + 25 * sig[c], 4001)
            post[c] = np.exp(-0.5 * ((grids[c] - mu[c]) / sig[c])**2); post[c] /= post[c].sum()
    if return_counts: return mu, M * t0 + min(t_cap, 1.0 / max(sig.min(), 1e-9)), counts
    return mu, M * t0 + min(t_cap, 1.0 / max(sig.min(), 1e-9))


def run():
    res = {}
    for name, (a, b) in problems().items():
        lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2
        res[name] = {"lams": lams.tolist(), "w": w.tolist(), "rows": []}
        for T2 in (np.inf, 3000.0, 1000.0, 300.0, 100.0):
            row = {"T2": T2 if np.isfinite(T2) else "inf"}
            P.CONTRAST = (lambda t, T2=T2: (1 - PD) * np.exp(-np.asarray(t) / T2)) if np.isfinite(T2) else (lambda t: (1 - PD) + 0 * np.asarray(t))
            P.T_CAP = T2 if np.isfinite(T2) else np.inf
            F_bc, T_bc = [], []
            for rep in range(3):
                P.rng = np.random.default_rng(100 + rep)
                mu0, we, s1, ser = P.stage1a(lams, w, 1.0, len(lams))
                mu, sig, s2, tr = P.stage1b(lams, w, mu0, we, ser, shots_total=SHOTS - s1)
                F_bc.append(fidelity(a, b, mu)); T_bc.append(min(P.T_CAP, 2400.0))
            row["BC"] = {"F_median": float(np.median(F_bc)), "F_min": float(np.min(F_bc)), "ancilla": 1,
                         "max_time": float(np.median(T_bc)), "shots": SHOTS}
            n_gap = int(np.ceil(np.log2(2 * 1.25 / max(np.diff(lams).min(), 1e-3))))   # bits to resolve all gaps
            n_coh = int(np.floor(np.log2(T2 / (2 * np.pi / 1.25)))) if np.isfinite(T2) else n_gap   # coarse QPE time <= T2 (contrast >= 1/e)
            n_c = max(2, min(n_gap, n_coh, 6))                  # <= 6 bits (coherent time = QPE6); unresolved pairs -> one cluster (Prop. 8)
            if T2 == res[name]["rows"][0]["T2"] if res[name]["rows"] else True:
                print(f"  [{name}] min gap={np.diff(lams).min():.4f}  n_gap={n_gap}")
            row["n_c"] = n_c
            F_b2, T_b2 = [], []
            for rep in range(3):
                lam2, tmax2 = bc2_learn(lams, w, n_c, T2, SHOTS, np.random.default_rng(300 + rep))
                F_b2.append(fidelity(a, b, lam2)); T_b2.append(tmax2)
            row["BC2"] = {"F_median": float(np.median(F_b2)), "F_min": float(np.min(F_b2)), "ancilla": n_c + 1,
                          "max_time": float(np.median(T_b2)), "shots": SHOTS}
            for n_p in (4, 6, 8):
                F_q = []
                for rep in range(3):
                    lam_q, tmax = qpe_learn(lams, w, n_p, T2, SHOTS, np.random.default_rng(200 + rep))
                    F_q.append(fidelity(a, b, lam_q) if len(lam_q) else 0.0)
                row[f"QPE n_p={n_p}"] = {"F_median": float(np.median(F_q)), "F_min": float(np.min(F_q)),
                                         "ancilla": n_p, "max_time": float(tmax), "shots": SHOTS}
            row["REF n_p=6"] = {"F_median": fidelity(a, b, np.maximum(np.round(lams * 64), 1) / 64), "ancilla": 6, "max_time": 64 * 2 * np.pi / 1.25}
            res[name]["rows"].append(row)
            print(name, row["T2"], f"n_c={n_c}", {k: round(v["F_median"], 5) for k, v in row.items() if k not in ("T2", "n_c")})
    json.dump(res, open("results/stage2_results.json", "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.6), sharey=True)
        for ax, (name, r) in zip(np.atleast_1d(axes), res.items()):
            T2s = [1e4 if row["T2"] == "inf" else row["T2"] for row in r["rows"]]
            for key in ("BC", "BC2", "QPE n_p=4", "QPE n_p=6", "QPE n_p=8"):
                ax.plot(T2s, [1 - row[key]["F_median"] for row in r["rows"]], "o-", label=key)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("T2 (units of 1/lam_max); 1e4 = no dephasing")
            ax.set_title(name, fontsize=9); ax.grid(alpha=.3)
        np.atleast_1d(axes)[0].set_ylabel("1 - fidelity (median of 3)"); np.atleast_1d(axes)[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig("results/stage2_fidelity.png", dpi=130)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    run()
