"""Stage 7: observable-first spectral-measure solver vs model-free randomized Fourier estimator.
Quantities of interest for x = A^{-1} b (A Hermitian positive definite, lam_max = 1):
  compliance  C = <b|A^{-1}|b>            = sum_j w_j / lam_j          (function of the spectral measure)
  energy      E = <x|A|x>/<x|x>           = (sum_j w_j/lam_j) / (sum_j w_j/lam_j^2)
  functional  L = <v|A^{-1}|b>            = sum_j <v|u_j><u_j|b> / lam_j  (needs cross weights c_j)
Spectral method: learn {lam_j} and the diagonal weights w_j with the projective estimator (BC2) on |b>, and the
  cross weights c_j by the polarisation identity from runs on (|b>+|v>)/sqrt2 and (|b>+i|v>)/sqrt2
  (weights w^+_j, w^i_j): c_j = w^+_j - (w_j + w^v_j)/2 + i (w^i_j - (w_j + w^v_j)/2).
  Weights come from the coarse-QPE label counts; eigenvalues from the refinement.
Randomized Fourier method (model-free, same primitive): A^{-1} = (1/sqrt(2 pi)) int_0^T dy int dz z e^{-z^2/2} e^{-i y z A}
  truncated at y <= Y = kappa log(1/eps); sample (y, z), Hadamard test of e^{-i y z A} between the input states.
  Unbiased up to truncation; variance ~ Y^2 / shots.  Dephasing contrast on each evolution of duration |yz|.
Output: results/stage7_results.json
"""
import json, importlib.util, numpy as np
S2 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s2", "stage2_noise_compare.py")); spec.loader.exec_module(S2)
S3 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s3", "stage3_backend_and_baselines.py")); spec.loader.exec_module(S3)
S3.P = S2.P; S3.S2 = S2                       # one shared noise hook (set_noise must act on the P that bc2_learn/measure use)
P, PD = S2.P, S2.PD
T0 = 2 * np.pi / 1.25


def weights(U, psi):
    return np.abs(U.conj().T @ psi)**2


def qpe_kernel(mu, n_c, T2, rng, n_mc=200):
    """Expected clock-outcome distribution P_j(y) for each learned eigenvalue (dephasing averaged)."""
    M = 2**n_c; K = np.zeros((M, len(mu)))
    for j, m_ in enumerate(mu):
        amp = S3.qpe_amplitudes(m_, n_c, T2, rng, n_mc)
        pr = np.abs(amp)**2; pr /= pr.sum(1, keepdims=True)
        K[:, j] = (1 - PD) * pr.mean(0) + PD / M
    return K


def weights_nnls(counts, kernel):
    """Non-negative least squares: counts/shots ~ sum_j w_j P_j(y)."""
    from scipy.optimize import nnls
    w, _ = nnls(kernel, counts / counts.sum())
    return w / max(w.sum(), 1e-12)


def learn_measure(lams, U, psi, n_c, T2, shots, rng, mu_shared=None):
    """Projective estimator on input psi. If mu_shared is given, only the weights are estimated
    (NNLS on the coarse-QPE counts with the QPE kernel of the shared eigenvalues)."""
    w = weights(U, psi)
    if mu_shared is None:
        mu, _, counts = S2.bc2_learn(lams, w, n_c, T2, shots, rng, return_counts=True)
        mu = np.sort(mu)
    else:
        mu = mu_shared
        _, y, M, t0 = S2.qpe_sample_labels(lams, w, n_c, T2, shots, rng); counts = np.bincount(y, minlength=M)
    K = qpe_kernel(mu, n_c, T2, rng)
    return mu, weights_nnls(counts, K)


def spectral_qois(lams, U, b, v, n_c, T2, shots, rng):
    """Eigenvalues from the |b> run; weights of |b>, |v>, (|b>+|v>)/sqrt2, (|b>+i|v>)/sqrt2 by NNLS (shots/4 each)."""
    s = shots // 4
    mu_b, w_b = learn_measure(lams, U, b, n_c, T2, s, rng)
    bp = (b + v) / np.linalg.norm(b + v); bi = (b + 1j * v) / np.linalg.norm(b + 1j * v)
    _, wv = learn_measure(lams, U, v, n_c, T2, s, rng, mu_shared=mu_b)
    _, wp = learn_measure(lams, U, bp, n_c, T2, s, rng, mu_shared=mu_b)
    _, wi = learn_measure(lams, U, bi, n_c, T2, s, rng, mu_shared=mu_b)
    np2, ni2 = np.linalg.norm(b + v)**2, np.linalg.norm(b + 1j * v)**2
    c = (wp * np2 - w_b - wv) / 2 + 1j * (wi * ni2 - w_b - wv) / 2            # <v|u_j><u_j|b>
    C = float(np.sum(w_b / mu_b)); E = float(np.sum(w_b / mu_b) / np.sum(w_b / mu_b**2)); L = complex(np.sum(c / mu_b))
    return C, E, L, mu_b, w_b


def fourier_functional(lams, U, b, v, T2, shots, rng, eps=1e-3):
    """Model-free randomized Fourier estimate of <v|A^{-1}|b>, Hadamard tests with the polarisation trick:
    <v|e^{-iAt}|b> from runs on the four superposition states.  Returns estimate and max evolution time."""
    kappa = 1.0 / lams.min(); Y = kappa * np.log(1 / eps)
    bp = (b + v) / np.linalg.norm(b + v); bi = (b + 1j * v) / np.linalg.norm(b + 1j * v)
    states = {"b": (b, 1.0), "v": (v, 1.0), "p": (bp, np.linalg.norm(b + v)**2), "i": (bi, np.linalg.norm(b + 1j * v)**2)}
    W = {k: weights(U, s[0]) for k, s in states.items()}
    n = shots // 4; est = 0j; tmax = 0.0
    for _ in range(n):
        y = rng.uniform(0, Y); z = rng.normal(); t = y * z; tmax = max(tmax, abs(t))
        con = P.CONTRAST(abs(t))
        g = {}
        for k, (s_, nrm) in states.items():   # single-shot estimates of Re/Im <s|e^{-iAt}|s> = sum w cos/sin, corrected
            re = (2 * rng.binomial(1, np.clip(0.5 + 0.5 * con * np.sum(W[k] * np.cos(lams * t)), 0, 1)) - 1) / con
            im = (2 * rng.binomial(1, np.clip(0.5 + 0.5 * con * np.sum(W[k] * np.sin(-lams * t)), 0, 1)) - 1) / con
            g[k] = (re + 1j * im) * nrm
        gvb = ((g["p"] - g["b"] - g["v"]) + 1j * (g["i"] - g["b"] - g["v"])) / 2
        est += 1j * Y * z * gvb                                                    # A^{-1} = i int_0^inf dy E_z[z e^{-iyzA}];  y ~ U[0,Y]
    return est / n, tmax


def run():
    out = {}; probs = S2.problems()
    rng0 = np.random.default_rng(5)
    for name in ("P3 Poisson N=8", "P3 Poisson N=4", "P4 Toeplitz N=8", "P5 random 4x4"):
        a, b = probs[name]; lams, U = np.linalg.eigh(a); N = len(lams)
        v = rng0.normal(size=N) + 0j; v /= np.linalg.norm(v)
        x = np.linalg.solve(a, b)
        C_true = float(np.real(np.vdot(b, x))); E_true = float(np.real(np.vdot(x, a @ x) / np.vdot(x, x))); L_true = complex(np.vdot(v, x))
        n_gap = int(np.ceil(np.log2(2 * 1.25 / max(np.diff(lams).min(), 1e-3))))
        res = {"C_true": C_true, "E_true": E_true, "L_true": [L_true.real, L_true.imag], "kappa": float(1 / lams.min())}
        for T2 in (np.inf, 300.0):
            S3.set_noise(T2)
            n_c = max(2, min(n_gap, int(np.floor(np.log2(T2 / T0))) if np.isfinite(T2) else n_gap, 6))   # <= 6 bits (resource-matched)
            for shots in (10_000, 40_000, 160_000):
                eC, eE, eL, eF, tF = [], [], [], [], []
                for rep in range(4):
                    rng = np.random.default_rng(2000 + rep)
                    C, E, L, mu, wb = spectral_qois(lams, U, b, v, n_c, T2, shots, rng)
                    eC.append(abs(C - C_true) / abs(C_true)); eE.append(abs(E - E_true) / abs(E_true)); eL.append(abs(L - L_true) / abs(L_true))
                    Lf, tmax = fourier_functional(lams, U, b, v, T2, shots, rng); eF.append(abs(Lf - L_true) / abs(L_true)); tF.append(tmax)
                key = f"T2={'inf' if not np.isfinite(T2) else T2}, shots={shots}, n_c={n_c}"
                res[key] = {"spectral: rel err compliance": float(np.median(eC)), "spectral: rel err energy": float(np.median(eE)),
                            "spectral: rel err functional <v|x>": float(np.median(eL)),
                            "fourier: rel err functional <v|x>": float(np.median(eF)), "fourier: median max evolution time": float(np.median(tF))}
                print(name, key, {k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in res[key].items()})
        out[name] = res
    json.dump(out, open("results/stage7_results.json", "w"), indent=1)


if __name__ == "__main__":
    run()
