"""Stage 7b: time-evolution quantum Krylov linear solving (QKS) as the strongest model-free baseline.
Same primitive as the spectral method: Hadamard tests giving g(t) = <b|e^{-iAt}|b>.
Krylov basis psi_k = e^{-iA t_k}|b>, t_k = k*Delta, k = 0..K-1.
  S_kl = <psi_k|psi_l> = g(t_l - t_k),   H_kl = <psi_k|A|psi_l> = h(t_l - t_k),  h(t) = <b|A e^{-iAt}|b>
  (h measured by a Hadamard test with one controlled application of A, same binomial shot noise, ||A|| <= 1)
Solve H c = r with r_k = <psi_k|b> = conj(g(t_k)) in the S-regularised subspace (eigenvalue threshold on S).
QoIs: C = <b|x> = sum_l c_l g(t_l);  E = (c^+ H c)/(c^+ S c);  L = <v|x> = sum_l c_l <v|e^{-iAt_l}|b> (polarisation).
Shots: g measured at 2K-1 lags x (central, +eps, -eps) x 2 quadratures, and for L on four input states.
"""
import json, importlib.util, numpy as np
S2 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s2", "stage2_noise_compare.py")); spec.loader.exec_module(S2)
S3 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s3", "stage3_backend_and_baselines.py")); spec.loader.exec_module(S3)
S3.P = S2.P; S3.S2 = S2                       # one shared noise hook (set_noise must act on the P that bc2_learn/measure use)
S7 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s7", "stage7_observable_first.py")); spec.loader.exec_module(S7)
P = S2.P


def g_meas(lams, w, t, shots, rng):
    con = P.CONTRAST(abs(t))
    re = (2 * rng.binomial(shots, float(np.clip(0.5 + 0.5 * con * np.sum(w * np.cos(lams * t)), 0, 1))) / shots - 1) / con
    im = (2 * rng.binomial(shots, float(np.clip(0.5 + 0.5 * con * np.sum(w * np.sin(-lams * t)), 0, 1))) / shots - 1) / con
    return re + 1j * im                          # estimate of sum_j w_j e^{-i lam_j t}


def h_meas(lams, w, t, shots, rng):
    con = P.CONTRAST(abs(t))
    re = (2 * rng.binomial(shots, float(np.clip(0.5 + 0.5 * con * np.sum(w * lams * np.cos(lams * t)), 0, 1))) / shots - 1) / con
    im = (2 * rng.binomial(shots, float(np.clip(0.5 + 0.5 * con * np.sum(w * lams * np.sin(-lams * t)), 0, 1))) / shots - 1) / con
    return re + 1j * im                          # estimate of sum_j w_j lam_j e^{-i lam_j t}


def qks(lams, U, b, v, K, Delta, shots_total, rng, thr=1e-2):
    states = {"b": (b, 1.0), "v": (v, 1.0), "p": ((b + v) / np.linalg.norm(b + v), np.linalg.norm(b + v)**2), "i": ((b + 1j * v) / np.linalg.norm(b + 1j * v), np.linalg.norm(b + 1j * v)**2)}
    W = {k: S7.weights(U, s[0]) for k, s in states.items()}
    lags = np.arange(-(K - 1), K) * Delta
    n_pts = len(lags) * 2 * 2 + K * 3 * 2                                 # g and h at all lags (2 quadratures) + three other states at t_k
    spp = max(4, shots_total // n_pts)
    g0 = {t: g_meas(lams, W["b"], t, spp, rng) for t in lags}
    h0 = {t: h_meas(lams, W["b"], t, spp, rng) for t in lags}
    S = np.array([[g0[lags[(l - k) + K - 1]] for l in range(K)] for k in range(K)])
    H = np.array([[h0[lags[(l - k) + K - 1]] for l in range(K)] for k in range(K)])
    S = (S + S.conj().T) / 2; H = (H + H.conj().T) / 2
    r = np.array([np.conj(g0[lags[k + K - 1]]) for k in range(K)])
    # regularise: project onto the S-eigenspace above threshold
    ev, V = np.linalg.eigh(S); keep = ev > thr * ev.max()
    Vk = V[:, keep] / np.sqrt(ev[keep])
    Ht = Vk.conj().T @ H @ Vk; rt = Vk.conj().T @ r
    c = Vk @ np.linalg.solve(Ht, rt)
    C = complex(np.sum(c * np.array([g0[lags[k + K - 1]] for k in range(K)])))
    E = complex((c.conj() @ H @ c) / (c.conj() @ S @ c))
    gv = []
    for k in range(K):                                                     # <v|e^{-iAt_k}|b> by polarisation
        t = k * Delta; gg = {s: g_meas(lams, W[s], t, spp, rng) * states[s][1] for s in ("b", "v", "p", "i")}
        gv.append(((gg["p"] - gg["b"] - gg["v"]) + 1j * (gg["i"] - gg["b"] - gg["v"])) / 2)
    L = complex(np.sum(c * np.array(gv)))
    return C.real, E.real, L, int(keep.sum())


def run():
    out = {}; probs = S2.problems(); rng0 = np.random.default_rng(5)
    for name in ("P3 Poisson N=8", "P3 Poisson N=4", "P4 Toeplitz N=8", "P5 random 4x4"):
        a, b = probs[name]; lams, U = np.linalg.eigh(a); N = len(lams)
        v = rng0.normal(size=N) + 0j; v /= np.linalg.norm(v)
        x = np.linalg.solve(a, b); C_true = float(np.real(np.vdot(b, x))); E_true = float(np.real(np.vdot(x, a @ x) / np.vdot(x, x))); L_true = complex(np.vdot(v, x))
        res = {}
        for T2 in (np.inf, 300.0):
            S3.set_noise(T2)
            for shots in (10_000, 40_000, 160_000):
                best = None
                for K, Delta, thr in ((N, 2.0, 1e-2), (N + 4, 2.0, 1e-2), (N + 4, 4.0, 1e-2), (N + 4, 4.0, 1e-3), (2 * N, 3.0, 1e-3)):   # grid of Krylov sizes / spacings / thresholds
                    eC, eE, eL, dims = [], [], [], []
                    for rep in range(4):
                        rng = np.random.default_rng(3000 + rep)
                        C, E, L, dim = qks(lams, U, b, v, K, Delta, shots, rng, thr=thr)
                        eC.append(abs(C - C_true) / abs(C_true)); eE.append(abs(E - E_true) / abs(E_true)); eL.append(abs(L - L_true) / abs(L_true)); dims.append(dim)
                    cand = {"K": K, "Delta": Delta, "thr": thr, "rel err C": float(np.median(eC)), "rel err E": float(np.median(eE)), "rel err L": float(np.median(eL)), "kept dim": float(np.median(dims)), "max time": float((K - 1) * Delta)}
                    if best is None or cand["rel err C"] + cand["rel err L"] < best["rel err C"] + best["rel err L"]: best = cand
                key = f"T2={'inf' if not np.isfinite(T2) else T2}, shots={shots}"
                res[key] = best; print(name, key, {k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in best.items()})
        out[name] = res
    json.dump(out, open("results/stage7b_qks.json", "w"), indent=1)


if __name__ == "__main__":
    run()
