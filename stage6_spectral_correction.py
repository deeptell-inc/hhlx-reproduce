"""Stage 6.
(1) Suresh 2026 spectral correction (min-norm correction in the odd Chebyshev basis, K smallest
    eigenvalues, degree unchanged) applied to a loose odd-parity bounded base polynomial, versus
(2) our node-aware LP with the SAME constraints (odd parity, |p| <= 1 on [-1,1]).
    Both evaluated on the TRUE spectrum with learned nodes from noiseless BC2.  Reported: minimal degree
    for F >= 0.999 / 0.9999, subnormalisation tau = max_{[-1,1]} |p| / max_j |p(lambda_j)|.
(3) Proposition 10 check: 1 - F <= sum_j w_j [(d^2 + lam_min/lam_j^2) |delta_j|]^2 / ||x_hat||^2 for
    node perturbations, and the per-component decomposition of 1-F when a weak component is missed.
Output: results/stage6_results.json
"""
import json, importlib.util, numpy as np
from numpy.polynomial import chebyshev as C
from scipy.optimize import linprog
S2 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s2", "stage2_noise_compare.py")); spec.loader.exec_module(S2)
S3 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s3", "stage3_backend_and_baselines.py")); spec.loader.exec_module(S3)
S3.P = S2.P; S3.S2 = S2                       # one shared noise hook (set_noise must act on the P that bc2_learn/measure use)
XG = np.linspace(-1, 1, 1601)


def odd_vander(x, n):                          # T_1, T_3, ..., T_{2n-1}
    V = C.chebvander(x, 2 * n - 1); return V[:, 1::2]


def poly_from_odd(c):
    full = np.zeros(2 * len(c)); full[1::2] = c
    return lambda x: C.chebval(x, full)


def lp_odd(n, nodes=None, weights=None, interval=None):
    """Odd-parity LP: variables c (n odd coefficients) [+ slack]; |p| <= 1 on [-1,1]."""
    V = odd_vander(XG, n)
    if nodes is not None:
        m = len(nodes); Vn = odd_vander(nodes, n); y = nodes.min() / nodes
        cost = np.concatenate([np.zeros(n), weights])
        A = np.vstack([np.hstack([Vn, -np.eye(m)]), np.hstack([-Vn, -np.eye(m)]), np.hstack([V, np.zeros((len(XG), m))]), np.hstack([-V, np.zeros((len(XG), m))])])
        bnd = np.concatenate([y, -y, np.ones(len(XG)), np.ones(len(XG))])
        r = linprog(cost, A_ub=A, b_ub=bnd, bounds=[(None, None)] * n + [(0, None)] * m, method="highs")
    else:
        a = interval; xs = np.linspace(a, 1.0, 400); Vi = odd_vander(xs, n); y = a / xs
        cost = np.concatenate([np.zeros(n), [1.0]])
        A = np.vstack([np.hstack([Vi, -np.ones((len(xs), 1))]), np.hstack([-Vi, -np.ones((len(xs), 1))]), np.hstack([V, np.zeros((len(XG), 1))]), np.hstack([-V, np.zeros((len(XG), 1))])])
        bnd = np.concatenate([y, -y, np.ones(len(XG)), np.ones(len(XG))])
        r = linprog(cost, A_ub=A, b_ub=bnd, bounds=[(None, None)] * n + [(0, None)], method="highs")
    return r.x[:n]


def spectral_correction(c0, nodes, lam_min):
    """Suresh 2026 Sec. 4.1: min-norm Delta c in the same odd basis so that lam_k p(lam_k) = lam_min
    (target lam_min/x so that the polynomial stays O(1)); K = len(nodes)."""
    n = len(c0); B = odd_vander(nodes, n)                     # (K, n)
    r = lam_min - nodes * (B @ c0)                              # residuals lam_k p0(lam_k) - target
    Lam = np.diag(nodes); G = Lam @ B @ B.T @ Lam
    alpha = np.linalg.solve(G, r)
    return c0 + B.T @ (nodes * alpha)


def fidelity_poly(a, b, p):
    lams, U = np.linalg.eigh(a); beta = U.conj().T @ b
    x = np.linalg.solve(a, b); x /= np.linalg.norm(x)
    xh = U @ (beta * p(lams)); xh /= np.linalg.norm(xh)
    return float(abs(np.vdot(x, xh))**2)


def tau(p, lams):
    return float(np.max(np.abs(p(XG))) / np.max(np.abs(p(lams))))


SEEDS = range(1300, 1340)                       # learning runs behind the degree table (the first one feeds the rest of run())


def seed_study(a, b, lams, w, n_c, degs, c_int):
    """Minimal degrees over independent learning runs: spectral correction at all learned nodes and node-aware LP.
    None = target not reached up to the largest degree.  tau: subnormalisation at the F >= 0.9999 degree (0.999 if not reached)."""
    lmin = lams.min(); R = {k: {"deg F>=0.999": [], "deg F>=0.9999": [], "tau F>=0.999": [], "tau F>=0.9999": []} for k in ("spectral correction", "node LP")}; R["nodes learned"] = []
    for seed in SEEDS:
        mu = np.sort(S2.bc2_learn(lams, w, n_c, np.inf, S2.SHOTS, np.random.default_rng(seed))[0]); R["nodes learned"].append(len(mu))
        om = np.sqrt(np.array([w[np.argmin(np.abs(lams - x))] for x in mu])) / mu**2
        for key in ("spectral correction", "node LP"):
            first = {0.999: (None, None), 0.9999: (None, None)}
            for d in degs:
                nn = (d + 1) // 2
                try: p = poly_from_odd(spectral_correction(c_int[d], mu, lmin) if key == "spectral correction" else lp_odd(nn, nodes=mu, weights=om))
                except np.linalg.LinAlgError: continue
                F = fidelity_poly(a, b, p)
                for t in first:
                    if first[t][0] is None and F >= t: first[t] = (d, tau(p, lams))
                if first[0.9999][0] is not None: break
            for t in first: R[key][f"deg F>={t}"].append(first[t][0]); R[key][f"tau F>={t}"].append(first[t][1])
    return R


def run():
    out = {}; probs = S2.problems(); S3.set_noise(np.inf)
    for name in ("P3 Poisson N=8", "P3 Poisson N=4", "P4 Toeplitz N=8", "P5 random 4x4"):
        a, b = probs[name]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; m = len(lams); lmin = lams.min()
        n_gap = int(np.ceil(np.log2(2 * 1.25 / max(np.diff(lams).min(), 1e-3)))); n_c = min(n_gap, 10)
        mu, _ = S2.bc2_learn(lams, w, n_c, np.inf, S2.SHOTS, np.random.default_rng(1300)); mu = np.sort(mu)
        w_mu = np.array([w[np.argmin(np.abs(lams - x))] for x in mu]); om = np.sqrt(w_mu) / mu**2
        degs = list(range(3, 80, 2)); res = {"kappa": float(lams.max() / lmin), "m": m}
        c_int = {d: lp_odd((d + 1) // 2, interval=lmin) for d in degs}
        res["seed study"] = seed_study(a, b, lams, w, n_c, degs, c_int) | {"seeds": len(SEEDS), "d_max": degs[-1]}
        # (1) spectral correction of an odd bounded interval base polynomial, K = all learned, K = ceil(m/2) smallest
        for K in (len(mu), int(np.ceil(m / 2))):
            Fs, taus = [], []
            for d in degs:
                n = (d + 1) // 2
                c0 = lp_odd(n, interval=lmin)
                nodes = mu[:K]
                try: c = spectral_correction(c0, nodes, lmin)
                except np.linalg.LinAlgError: Fs.append(np.nan); taus.append(np.nan); continue
                p = poly_from_odd(c); Fs.append(fidelity_poly(a, b, p)); taus.append(tau(p, lams))
            res[f"spectral-correction K={K}"] = {"deg F>=0.999": next((d for d, F in zip(degs, Fs) if F >= 0.999), None),
                                                 "deg F>=0.9999": next((d for d, F in zip(degs, Fs) if F >= 0.9999), None),
                                                 "tau at that degree": next((t for d, F, t in zip(degs, Fs, taus) if F >= 0.9999), None),
                                                 "F vs deg": {str(d): round(F, 5) for d, F in zip(degs, Fs) if d in (3, 5, 7, 9, 11, 15, 19, 25, 31, 41, 61, 79)}}
        # (2) node-aware odd LP (same constraints) and odd interval LP reference
        Fn, tn, Fi = [], [], []
        for d in degs:
            n = (d + 1) // 2
            pn = poly_from_odd(lp_odd(n, nodes=mu, weights=om)); Fn.append(fidelity_poly(a, b, pn)); tn.append(tau(pn, lams))
            Fi.append(fidelity_poly(a, b, poly_from_odd(lp_odd(n, interval=lmin))))
        res["node-aware odd LP"] = {"deg F>=0.999": next((d for d, F in zip(degs, Fn) if F >= 0.999), None), "deg F>=0.9999": next((d for d, F in zip(degs, Fn) if F >= 0.9999), None),
                                    "tau at that degree": next((t for d, F, t in zip(degs, Fn, tn) if F >= 0.9999), None), "F vs deg": {str(d): round(F, 5) for d, F in zip(degs, Fn) if d in (3, 5, 7, 9, 11, 15, 19, 25, 31, 41, 61, 79)}}
        res["interval odd LP (standard)"] = {"deg F>=0.999": next((d for d, F in zip(degs, Fi) if F >= 0.999), None), "deg F>=0.9999": next((d for d, F in zip(degs, Fi) if F >= 0.9999), None),
                                             "F vs deg": {str(d): round(F, 5) for d, F in zip(degs, Fi) if d in (3, 5, 7, 9, 11, 15, 19, 25, 31, 41, 61, 79)}}
        # (3) Proposition 10: perturbation bound, node-aware odd LP at the F>=0.9999 degree (true nodes)
        d_star = next((d for d, F in zip(degs, Fn) if F >= 0.9999), degs[-1]); n = (d_star + 1) // 2
        prop10 = {}
        for sig in (1e-4, 1e-3, 1e-2):
            lhs, rhs = [], []
            for rep in range(10):
                nodes = lams * (1 + sig * np.random.default_rng(rep).normal(size=m)); delta = nodes - lams
                p = poly_from_odd(lp_odd(n, nodes=nodes, weights=np.sqrt(w) / nodes**2))
                F = fidelity_poly(a, b, p); lhs.append(1 - F)
                x = np.linalg.solve(a, b); beta = U.conj().T @ b
                xh_norm2 = np.sum(w * p(lams)**2)
                r_node = np.abs(p(nodes) - lmin / nodes)                                   # node residual r_j
                rhs.append(float(np.sum(w * (r_node + (d_star**2 * np.max(np.abs(p(XG))) + lmin / (lams * nodes)) * np.abs(delta))**2) / xh_norm2))
            prop10[f"sigma={sig}"] = {"1-F median": float(np.median(lhs)), "bound median": float(np.median(rhs)), "bound holds": bool(np.all(np.array(lhs) <= np.array(rhs) + 1e-12))}
        res["Prop10 (degree %d)" % d_star] = prop10
        # missed-component decomposition with learned nodes (Toeplitz has a missed weak component)
        pL = poly_from_odd(lp_odd(n, nodes=mu, weights=om))
        beta = U.conj().T @ b; xh = beta * pL(lams); xh /= np.linalg.norm(xh); xb = (beta / lams); xb /= np.linalg.norm(xb)
        comp = {f"lam={lams[j]:.4f}, w={w[j]:.3f}": float(abs(xh[j] - xb[j])**2) for j in range(m)}
        res["per-component |x_hat - x|^2 (learned nodes, degree %d)" % d_star] = {"1-F": 1 - fidelity_poly(a, b, pL), **comp}
        out[name] = res
        print(name, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "F vs deg"}) for k, v in res.items()})[:1400])
    json.dump(out, open("results/stage6_results.json", "w"), indent=1)


if __name__ == "__main__":
    run()
