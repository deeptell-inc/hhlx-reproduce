"""Stage 4.
(i)  MM-QCELS-style baseline (Ding & Lin 2023): multi-level time grids tau_l = tau_0 2^l, N points per
     level, Z(t) from Hadamard tests, multi-modal least squares for {lambda_j, r_j} warm-started from the
     previous level (level 0 from a global grid search).  Mixture statistics, single ancilla.
(ii) Clock-scaling optimisation for the method-A back end: choose t0 so that the LEARNED eigenvalues sit
     on clock bins (hybrid HHL++ idea), minimising sum_j w_j dist(mu_j t0 M/2pi, Z)^2.
(iii) 10 repetitions, median and IQR.
(iv) Refinement time cap t_max = kappa_t * T2 sweep for BC2 (true T2 = 300, calibrated).
Output: results/stage4_results.json
"""
import json, importlib.util, numpy as np
from scipy.optimize import least_squares
S2 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s2", "stage2_noise_compare.py")); spec.loader.exec_module(S2)
S3 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s3", "stage3_backend_and_baselines.py")); spec.loader.exec_module(S3)
S3.P = S2.P; S3.S2 = S2                       # one shared noise hook (set_noise must act on the P that mm_qcels/bc2_learn/measure use)
P, PD, SHOTS = S2.P, S2.PD, S2.SHOTS
REPS = 100                                    # repetitions per table cell (ten do not pin the medians of the mixture estimators)


def mm_qcels(lams, w, T2, shots, rng, N=12, levels=None):
    """Returns learned eigenvalues (sorted)."""
    m = len(lams)
    tau0 = 2 * np.pi / 1.25 / N                                   # level-0 grid spans one period of lam_max
    T_cap = T2 if np.isfinite(T2) else 4000.0
    if levels is None:
        levels = int(np.floor(np.log2(T_cap / ((N - 1) * tau0)))) + 1
    levels = max(1, levels)
    spp = max(2, shots // (2 * N * levels))                      # shots per (point, quadrature)
    data_t, data_Z = [], []
    lam = None
    for l in range(levels):
        tau = tau0 * 2**l
        ts = np.arange(N) * tau
        con = P.CONTRAST(ts)
        Z = np.array([(2 * P.measure(lams, w, t, 0.0, spp) / spp - 1) + 1j * (2 * P.measure(lams, w, t, -np.pi / 2, spp) / spp - 1) for t in ts]) / con
        data_t.append(ts); data_Z.append(Z)
        tt, ZZ = np.concatenate(data_t), np.concatenate(data_Z)
        cw = P.CONTRAST(tt)
        if lam is None:                                           # level 0: global coordinate grid search
            lam = np.full(m, 0.6); r = np.full(m, 1.0 / m)
            cand = np.linspace(0.02, 1.2, 300)
            for _ in range(3):
                for c in range(m):
                    best, bx = np.inf, lam[c]
                    for v in cand:
                        lam[c] = v
                        V = np.exp(1j * np.outer(tt, lam)); rr = np.clip(np.real(np.linalg.lstsq(V, ZZ, rcond=None)[0]), 1e-3, None)
                        res = np.sum(cw**2 * np.abs(V @ rr - ZZ)**2)
                        if res < best: best, bx = res, v
                    lam[c] = bx
            V = np.exp(1j * np.outer(tt, lam)); r = np.clip(np.real(np.linalg.lstsq(V, ZZ, rcond=None)[0]), 1e-3, None)
        def resid(x):
            model = np.exp(1j * np.outer(tt, x[:m])) @ x[m:]
            return np.concatenate([cw * (model - ZZ).real, cw * (model - ZZ).imag])
        lo = np.concatenate([np.zeros(m), np.full(m, 1e-3)]); hi = np.concatenate([np.full(m, 1.5), np.ones(m)])
        fit = least_squares(resid, np.clip(np.concatenate([lam, r]), lo + 1e-6, hi - 1e-6), bounds=(lo, hi), method="trf")
        lam, r = fit.x[:m], fit.x[m:]
    return np.sort(lam)


def optimise_t0(mu, w_est, M):
    """Place learned eigenvalues on clock bins: minimise sum_j w_j * dist(mu_j t0 M / 2pi, Z)^2 over t0."""
    best, bt = np.inf, 2 * np.pi / 1.25
    for t0 in np.linspace(0.4, 1.0, 1201) * (2 * np.pi / 1.25):
        if mu.max() * t0 >= 2 * np.pi * 0.98: continue
        pos = mu * t0 * M / (2 * np.pi)
        cost = np.sum(w_est * (pos - np.round(pos))**2)
        if cost < best: best, bt = cost, t0
    return bt


def lam_table(mu, M, t0):
    lam_y = np.full(M, np.nan)
    for m_ in mu:
        yy = int(round(m_ * M * t0 / (2 * np.pi))) % M
        for d in (-1, 0, 1): lam_y[(yy + d) % M] = m_
    return lam_y


def q(xs):
    xs = np.asarray(xs, float)
    return {"median": float(np.median(xs)), "q25": float(np.percentile(xs, 25)), "q75": float(np.percentile(xs, 75)), "runs": [float(x) for x in xs]}


def run():
    out = {}
    probs = S2.problems()
    cap_sweep(out, probs)
    for name, (a, b) in probs.items():
        lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2
        n_gap = int(np.ceil(np.log2(2 * 1.25 / max(np.diff(lams).min(), 1e-3))))
        rows = []
        for T2 in (np.inf, 300.0):
            S3.set_noise(T2)
            n_c = max(2, min(n_gap, int(np.floor(np.log2(T2 / (2 * np.pi / 1.25)))) if np.isfinite(T2) else n_gap, 6))   # <= 6 bits (resource-matched)
            M = 2**n_c
            R = {k: [] for k in ("MM-QCELS", "QMEGS", "QPE6", "BC2 ideal", "BC2+clock", "BC2+clock+t0opt", "no-learn clock", "no-learn clock+t0opt(true)")}
            learned = {"QPE6": [], "BC2 ideal": []}
            for rep in range(REPS):
                rng = np.random.default_rng(900 + rep)
                R["MM-QCELS"].append(S2.fidelity(a, b, mm_qcels(lams, w, T2, SHOTS, rng)))
                R["QMEGS"].append(S2.fidelity(a, b, S3.qmegs_learn(lams, w, T2, SHOTS, rng)[0]))
                lq, _ = S2.qpe_learn(lams, w, 6, T2, SHOTS, rng); R["QPE6"].append(S2.fidelity(a, b, lq) if len(lq) else 0.0)
                learned["QPE6"].append([float(x) for x in np.sort(lq)])
                mu, _ = S2.bc2_learn(lams, w, n_c, T2, SHOTS, rng, t_cap=(min(2 * T2, 2000.0) if np.isfinite(T2) else 2000.0))   # refinement time <= 2000 (= mixture window)
                R["BC2 ideal"].append(S2.fidelity(a, b, mu))
                learned["BC2 ideal"].append([float(x) for x in np.sort(mu)])
                S3.T0 = 2 * np.pi / 1.25
                R["BC2+clock"].append(S3.coherent_backend_fidelity(a, b, n_c, T2, lam_table(mu, M, S3.T0), rng, n_mc=32))
                R["no-learn clock"].append(S3.coherent_backend_fidelity(a, b, n_c, T2, S3.lam_table_standard(n_c, lams.min()), rng, n_mc=32))
                w_mu = np.array([w[np.argmin(np.abs(lams - m_))] for m_ in mu])       # weights of learned components (from coarse counts in practice)
                t0 = optimise_t0(mu, w_mu, M); S3.T0 = t0
                R["BC2+clock+t0opt"].append(S3.coherent_backend_fidelity(a, b, n_c, T2, lam_table(mu, M, t0), rng, n_mc=32))
                t0s = optimise_t0(lams, w, M); S3.T0 = t0s                                 # oracle: true spectrum on bins, no learning
                ly = 2 * np.pi * np.arange(M) / (M * t0s); R["no-learn clock+t0opt(true)"].append(S3.coherent_backend_fidelity(a, b, n_c, T2, np.where(ly >= lams.min() * 0.9, ly, np.nan), rng, n_mc=32))
                S3.T0 = 2 * np.pi / 1.25
            row = {"T2": "inf" if not np.isfinite(T2) else T2, "n_c": n_c, **{k: q(v) for k, v in R.items()}, "learned eigenvalues": learned}
            rows.append(row)
            print(name, row["T2"], f"n_c={n_c}", {k: f"{v['median']:.4f} [{v['q25']:.4f},{v['q75']:.4f}]" for k, v in row.items() if k not in ("T2", "n_c", "learned eigenvalues")})
        out[name] = rows
    json.dump(out, open("results/stage4_results.json", "w"), indent=1)


def cap_sweep(out, probs):
    # (iv) time-cap rule sweep, P3 Poisson N=8, true T2 = 300, calibrated
    a, b = probs["P3 Poisson N=8"]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2
    T2 = 300.0; S3.set_noise(T2); cap = {}
    for kt in (0.5, 1.0, 1.5, 2.0, 3.0):
        Fs = []
        for rep in range(REPS):
            mu, _ = S2.bc2_learn(lams, w, 5, T2, SHOTS, np.random.default_rng(1000 + rep), t_cap=kt * T2)   # coarse-QPE noise at the true T2; only the cap varies
            Fs.append(S2.fidelity(a, b, mu))
        cap[f"kappa_t={kt}"] = q(Fs); print("cap", kt, cap[f"kappa_t={kt}"])
    out["time-cap sweep (P3 N=8, T2=300)"] = cap


if __name__ == "__main__":
    run()
