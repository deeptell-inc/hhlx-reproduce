"""Stage 3 (repositioned paper: choosing the learning stage of hybrid HHL).
(A) QMEGS-like mixture-statistics baseline (Ding, Li, Lin, Ni 2024): Hadamard test at Gaussian-random
    times t_n ~ N(0, T^2), Gaussian-filtered spectral function G(x) = mean_n Re[Z_n e^{i x t_n}], peaks.
(B) Coherent method-A back end with a real n_c-bit clock: QPE leakage (sinc tails, dephasing) and the
    imperfect uncomputation that follows a y-dependent rotation are included exactly for the ideal
    circuit (Monte Carlo over dephasing phases).  Rotation angle for clock value y comes from the
    learned spectrum (BC2 peak -> refined eigenvalue) or, for unlearned bins, from 2*pi*y/(M t0).
(C) BC2 robustness to a mis-calibrated T2 (assumed T2 = 0.7x and 1.4x the true one).
(D) BC2 shot scaling of the eigenvalue error for the weakest component.
Output: results/stage3_results.json
"""
import json, importlib.util, numpy as np
S2 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s2", "stage2_noise_compare.py")); spec.loader.exec_module(S2)
P = S2.P; PD = S2.PD; SHOTS = S2.SHOTS
T0 = 2 * np.pi / 1.25


def set_noise(T2):
    P.CONTRAST = (lambda t, T2=T2: (1 - PD) * np.exp(-np.asarray(t) / T2)) if np.isfinite(T2) else (lambda t: (1 - PD) + 0 * np.asarray(t))
    P.T_CAP = T2 if np.isfinite(T2) else np.inf


def qmegs_learn(lams, w, T2, shots, rng, m=None):
    """QMEGS-like: N = shots/2 Hadamard-test pairs at t_n ~ N(0, T^2), T = min(T2, 2000); contrast-corrected."""
    m = m or len(lams)
    T = min(T2, 2000.0) if np.isfinite(T2) else 2000.0
    N = shots // 2
    tn = np.abs(rng.normal(0, T, N))
    con = P.CONTRAST(tn)
    re = np.array([2 * P.measure(lams, w, t, 0.0, 1) - 1 for t in tn]) / con
    im = np.array([2 * P.measure(lams, w, t, -np.pi / 2, 1) - 1 for t in tn]) / con
    Z = re + 1j * im
    x = np.linspace(0, 1.2, 12001)
    G = np.real(np.exp(1j * np.outer(x, tn)) @ np.conj(Z)) / N       # mean_n Re[Z_n^* e^{i x t_n}] peaks at lambda_j
    peaks = []
    for i in np.argsort(-G):
        if all(abs(x[i] - q) > 0.02 for q in peaks): peaks.append(x[i])
        if len(peaks) == m: break
    return np.array(sorted(peaks)), T


def qpe_amplitudes(lam, n_c, T2, rng, n_mc, window=None):
    """Clock amplitudes a_y for one eigencomponent under control-qubit dephasing; (n_mc, M).
    window: unit-norm clock state (default: uniform superposition)."""
    M = 2**n_c; k = np.arange(M); bits = ((k[:, None] >> np.arange(n_c)) & 1); dur = 2.0**np.arange(n_c) * T0
    xi = rng.normal(0, np.sqrt(2 * dur / T2), size=(n_mc, n_c)) if np.isfinite(T2) else np.zeros((n_mc, n_c))
    phase = k[None, :] * (lam * T0) + (bits[None, :, :] * xi[:, None, :]).sum(-1)
    return np.fft.fft(np.exp(1j * phase) * (1 / np.sqrt(M) if window is None else window), axis=1) / np.sqrt(M)


def coherent_backend_fidelity(a, b, n_c, T2, lam_of_y, rng, n_mc=64, window=None):
    """lam_of_y[y] = eigenvalue used for the rotation at clock value y, or NaN = no rotation (that bin is
    filtered away by the ancilla postselection, as in HHL's filter functions / hybrid HHL)."""
    """Exact ideal-circuit fidelity of method A with a real clock: rotation c_y = C/lam_of_y(y), then
    inverse QPE; the component of sum_y a_y c_y |y> along the QPE output vector returns to |0>, the rest
    leaks.  F = |<x,0|x_hat>|^2 / ||x_hat||^2, averaged over dephasing samples."""
    lams, U = np.linalg.eigh(a); beta = U.conj().T @ b
    x = np.linalg.solve(a, b); x /= np.linalg.norm(x); xb = U.conj().T @ x          # x in eigenbasis
    M = 2**n_c; cy = np.where(np.isnan(lam_of_y), 0.0, 1.0 / np.where(np.isnan(lam_of_y), 1.0, lam_of_y))   # C = 1
    Fs = []
    for s in range(n_mc):
        num, nrm = 0.0 + 0j, 0.0
        for j, lam in enumerate(lams):
            ay = qpe_amplitudes(lam, n_c, T2, rng, 1, window)[0]
            cbar = np.sum(np.abs(ay)**2 * cy)                                        # projection onto QPE output
            num += np.conj(xb[j]) * beta[j] * cbar
            nrm += np.abs(beta[j])**2 * np.sum(np.abs(ay)**2 * cy**2)
        Fs.append(abs(num)**2 / nrm)
    return float(np.mean(Fs))


def lam_table_from_bc2(lams, w, n_c, T2, shots, rng):
    """Run BC2 and build the clock-value -> rotation table: learned peaks refine their bins (+-1),
    every other bin gets NO rotation (filtered)."""
    mu, _ = S2.bc2_learn(lams, w, n_c, T2, shots, rng)
    M = 2**n_c; lam_y = np.full(M, np.nan)
    for m_ in mu:
        yy = int(round(m_ * M * T0 / (2 * np.pi))) % M
        for d in (-1, 0, 1): lam_y[(yy + d) % M] = m_
    return lam_y, mu


def lam_table_standard(n_c, lam_min):
    """Textbook HHL clock: rotate 1/lam_y on every bin with lam_y >= lam_min (filter below), lam_min known."""
    M = 2**n_c; lam_y = 2 * np.pi * np.arange(M) / (M * T0)
    return np.where(lam_y >= lam_min, lam_y, np.nan)


def run():
    out = {}
    probs = S2.problems()
    # (A) QMEGS baseline + (B) coherent back end, on all problems, T2 in {inf, 300}
    for name, (a, b) in probs.items():
        lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2
        n_gap = int(np.ceil(np.log2(2 * 1.25 / max(np.diff(lams).min(), 1e-3))))
        rows = []
        for T2 in (np.inf, 300.0):
            set_noise(T2)
            n_c = max(2, min(n_gap, int(np.floor(np.log2(T2 / T0))) if np.isfinite(T2) else n_gap))
            F_q, F_b2, F_be, F_be_std, F_be6 = [], [], [], [], []
            for rep in range(3):
                rng = np.random.default_rng(500 + rep)
                lq, _ = qmegs_learn(lams, w, T2, SHOTS, rng); F_q.append(S2.fidelity(a, b, lq))
                lam_y, mu = lam_table_from_bc2(lams, w, n_c, T2, SHOTS, rng)
                F_b2.append(S2.fidelity(a, b, mu))                                   # idealised tagging (stage 2)
                F_be.append(coherent_backend_fidelity(a, b, n_c, T2, lam_y, rng))    # real clock, learned angles
                F_be_std.append(coherent_backend_fidelity(a, b, n_c, T2, lam_table_standard(n_c, lams.min()), rng))  # same clock, no learning
                F_be6.append(coherent_backend_fidelity(a, b, 6, T2, lam_table_standard(6, lams.min()), rng))          # standard HHL n_p=6 clock
            rows.append({"T2": "inf" if not np.isfinite(T2) else T2, "n_c": n_c,
                         "QMEGS(mixture)": float(np.median(F_q)), "BC2 idealised": float(np.median(F_b2)),
                         "BC2 + real n_c clock": float(np.median(F_be)), f"n_c clock, no learning": float(np.median(F_be_std)),
                         "standard HHL n_p=6 clock": float(np.median(F_be6))})
            print(name, rows[-1])
        out[name] = rows
    # (C) calibration robustness on P3 Poisson N=8, true T2 = 300
    a, b = probs["P3 Poisson N=8"]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2
    cal = {}
    T2_true = 300.0
    true_con = lambda t: (1 - PD) * np.exp(-np.asarray(t) / T2_true)
    for fac in (0.7, 1.0, 1.4):
        T2_assumed = T2_true * fac
        P.CONTRAST = lambda t, T2=T2_assumed: (1 - PD) * np.exp(-np.asarray(t) / T2)    # what the estimator assumes
        P.T_CAP = T2_assumed
        Fs = [S2.fidelity(a, b, S2.bc2_learn(lams, w, 5, T2_assumed, SHOTS, np.random.default_rng(700 + r), true_contrast=true_con)[0]) for r in range(3)]
        cal[f"assumed/true T2 = {fac}"] = float(np.median(Fs))
        print("calibration", fac, cal[f"assumed/true T2 = {fac}"])
    out["calibration (P3 N=8, true T2=300)"] = cal
    # (D) shot scaling, noiseless, P3 Poisson N=8: rms error of the weakest and strongest component
    set_noise(np.inf)
    sc = {}
    for S in (2000, 5000, 10000, 20000, 40000):
        errs = []
        for r in range(4):
            mu, _ = S2.bc2_learn(lams, w, 5, np.inf, S, np.random.default_rng(800 + r))
            near = np.array([np.argmin(np.abs(mu - l)) for l in lams]); errs.append(np.abs(mu[near] - lams))
        rms = np.sqrt(np.mean(np.array(errs)**2, axis=0))
        sc[S] = {f"w={w[j]:.3f}": float(rms[j]) for j in range(len(lams))}
        print("shots", S, {k: f"{v:.1e}" for k, v in sc[S].items()})
    out["shot scaling (P3 N=8, noiseless, n_c=5)"] = sc
    json.dump(out, open("results/stage3_results.json", "w"), indent=1)


if __name__ == "__main__":
    run()
