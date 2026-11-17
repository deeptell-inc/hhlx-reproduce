"""Stage 5: problems whose spectrum cannot be placed on one clock (P3 Poisson N=8).
(a) Multi-pass clock: greedily split the learned spectrum into groups that share a t0 placing every
    member within `tol` bins of an integer; each group is one HHL pass (valid for LINEAR functionals
    of x, which add across passes; not for the state itself).
(b) Method B, spectrum-aware QSVT: polynomial p with p(mu_j) ~ 1/mu_j on the LEARNED eigenvalues,
    degree <= m-1, no clock, no leakage.  x_hat ~ sum_j beta_j p(lambda_j) u_j.  QSVT needs |p|<=1 on
    [-1,1], so the normalisation overhead is max_{[-1,1]}|p| / max_j |p(lambda_j)|; the postselection
    probability scales as its inverse square.  Variants: exact Lagrange interpolation (Chebyshev basis),
    and a regularised fit (nodes weighted, plus a bounded-ness penalty on [-1,1]).
    Reference: standard QSVT inverse polynomial = Chebyshev least-squares fit of 1/x on [lam_min, 1],
    minimal degree reaching F >= 0.999 and 0.9999.
Output: results/stage5_results.json
"""
import json, importlib.util, numpy as np
from numpy.polynomial import chebyshev as C
S2 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s2", "stage2_noise_compare.py")); spec.loader.exec_module(S2)
S3 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s3", "stage3_backend_and_baselines.py")); spec.loader.exec_module(S3)
S3.P = S2.P; S3.S2 = S2                       # one shared noise hook (set_noise must act on the P that bc2_learn/measure use)
P, SHOTS = S2.P, S2.SHOTS
XX = np.linspace(-1, 1, 4001)


def fidelity_poly(a, b, p):
    lams, U = np.linalg.eigh(a); beta = U.conj().T @ b
    x = np.linalg.solve(a, b); x /= np.linalg.norm(x)
    xh = U @ (beta * p(lams)); xh /= np.linalg.norm(xh)
    return float(abs(np.vdot(x, xh))**2)


def overhead(p, lams):
    return float(np.max(np.abs(p(XX))) / np.max(np.abs(p(lams))))


def lagrange_cheb(mu):
    c = C.chebfit(mu, 1.0 / mu, deg=len(mu) - 1)                   # exact interpolation in the Chebyshev basis
    return lambda x: C.chebval(x, c)


def regularised_poly(mu, w_est, deg, lam_bound=1e-3):
    """Weighted LS: fit 1/x at the learned nodes (weights ~ sqrt(w_j)/mu_j^2, the solution-error weights of
    Prop. 3) plus a weak penalty keeping |p| small on [-1,1] outside the spectrum."""
    xs_pen = np.linspace(-1, 1, 400)
    A = np.vstack([C.chebvander(mu, deg) * (np.sqrt(w_est) / mu**2)[:, None], np.sqrt(lam_bound) * C.chebvander(xs_pen, deg)])
    y = np.concatenate([(1.0 / mu) * (np.sqrt(w_est) / mu**2), np.zeros(len(xs_pen))])
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda x: C.chebval(x, c)


def standard_qsvt_poly(lam_min, deg):
    xs = np.linspace(lam_min, 1.0, 2000)
    c = C.chebfit(xs, 1.0 / xs, deg=deg)
    return lambda x: C.chebval(x, c)


def multipass_groups(mu, M, tol=0.1):
    """Greedy: repeatedly pick the t0 that places the most remaining eigenvalues within tol bins of integers."""
    remaining = list(range(len(mu))); groups = []
    t0_grid = np.linspace(0.3, 1.0, 2001) * (2 * np.pi / 1.25)
    while remaining:
        best, bg, bt = -1, [], None
        for t0 in t0_grid:
            if mu.max() * t0 >= 2 * np.pi * 0.98: continue
            pos = mu[remaining] * t0 * M / (2 * np.pi)
            ok = [remaining[i] for i in range(len(remaining)) if abs(pos[i] - round(pos[i])) <= tol and round(pos[i]) >= 1]
            if len(ok) > best: best, bg, bt = len(ok), ok, t0
        if not bg: bg, bt = [remaining[0]], None
        groups.append((bg, bt)); remaining = [r for r in remaining if r not in bg]
    return groups


def run():
    out = {}
    probs = S2.problems()
    for name in ("P3 Poisson N=8", "P3 Poisson N=4", "P4 Toeplitz N=8", "P5 random 4x4"):
        a, b = probs[name]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; m = len(lams)
        n_gap = int(np.ceil(np.log2(2 * 1.25 / max(np.diff(lams).min(), 1e-3))))
        res = {"lams": lams.round(5).tolist(), "w": w.round(4).tolist()}
        # (a) multi-pass grouping with the n_c used at T2=300 in stage 4 (5 bits) and with n_gap bits
        for n_c in sorted({5, min(n_gap, 10)}):
            g = multipass_groups(lams, 2**n_c)
            res[f"multipass n_c={n_c}"] = {"passes": len(g), "group sizes": [len(x[0]) for x in g]}
        # standard QSVT reference: minimal degree for F >= 0.999 / 0.9999 (lam_min known exactly)
        ref = {}
        for target in (0.999, 0.9999):
            for d in range(2, 200):
                p = standard_qsvt_poly(lams.min(), d)
                if fidelity_poly(a, b, p) >= target:
                    ref[f"F>={target}"] = {"degree": d, "overhead": overhead(p, lams)}; break
        res["standard QSVT 1/x"] = ref
        # (b) method B with learned spectra: noiseless BC2 and T2=300 BC2, 10 reps each
        for T2 in (np.inf, 300.0):
            S3.set_noise(T2)
            n_c = max(2, min(n_gap, int(np.floor(np.log2(T2 / (2 * np.pi / 1.25)))) if np.isfinite(T2) else n_gap))
            rows = {"lagrange": [], "lagrange overhead": [], "regularised deg m-1": [], "regularised overhead": [], "regularised deg 2m": [], "learn rel err max": []}
            for rep in range(10):
                mu, _ = S2.bc2_learn(lams, w, n_c, T2, SHOTS, np.random.default_rng(1200 + rep))
                near = np.array([np.argmin(np.abs(mu - l)) for l in lams]); rows["learn rel err max"].append(float(np.max(np.abs(mu[near] - lams) / lams)))
                w_mu = np.array([w[np.argmin(np.abs(lams - x))] for x in mu])
                pl = lagrange_cheb(mu); rows["lagrange"].append(fidelity_poly(a, b, pl)); rows["lagrange overhead"].append(overhead(pl, lams))
                pr = regularised_poly(mu, w_mu, len(mu) - 1); rows["regularised deg m-1"].append(fidelity_poly(a, b, pr)); rows["regularised overhead"].append(overhead(pr, lams))
                pr2 = regularised_poly(mu, w_mu, 2 * len(mu)); rows["regularised deg 2m"].append(fidelity_poly(a, b, pr2))
            res[f"method B, T2={'inf' if not np.isfinite(T2) else T2}, n_c={n_c}"] = {k: {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))} for k, v in rows.items()}
        # (b') sensitivity: exact spectrum perturbed by relative Gaussian error sigma, Lagrange vs regularised
        sens = {}
        for sig in (1e-4, 1e-3, 1e-2):
            Fl, Fr, Ol = [], [], []
            for rep in range(20):
                mu = lams * (1 + sig * np.random.default_rng(rep).normal(size=m))
                pl, pr = lagrange_cheb(mu), regularised_poly(mu, w, m - 1)
                Fl.append(fidelity_poly(a, b, pl)); Fr.append(fidelity_poly(a, b, pr)); Ol.append(overhead(pl, lams))
            sens[f"sigma={sig}"] = {"lagrange F": float(np.median(Fl)), "regularised F": float(np.median(Fr)), "lagrange overhead": float(np.median(Ol))}
        res["sensitivity to learning error"] = sens
        out[name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k not in ("lams", "w")}, indent=None)[:1500])
    json.dump(out, open("results/stage5_results.json", "w"), indent=1)


if __name__ == "__main__":
    run()
