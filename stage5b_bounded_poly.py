"""Stage 5b: fair spectrum-aware vs interval QSVT polynomials under the QSVT constraint |p| <= 1 on [-1,1].
Both are linear programs in Chebyshev coefficients c (degree d):
  node-aware : minimise sum_j omega_j |p(mu_j) - lam_min/mu_j|   (omega_j = sqrt(w_j)/mu_j^2, Prop. 3 weights)
  interval   : minimise max_{x in [lam_min,1]} |p(x) - lam_min/x|   (standard QSVT inversion target)
subject to -1 <= p(x_k) <= 1 on a grid of [-1,1].  Fidelity uses the TRUE eigenvalues; overhead = 1 by construction.
Also reports the Markov-inequality lower bound d >= sqrt(kappa/2) for any bounded polynomial with p(lam_min)=1, slope 1/lam_min.
Output: results/stage5b_results.json
"""
import json, importlib.util, numpy as np
from numpy.polynomial import chebyshev as C
from scipy.optimize import linprog
S2 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s2", "stage2_noise_compare.py")); spec.loader.exec_module(S2)
S3 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s3", "stage3_backend_and_baselines.py")); spec.loader.exec_module(S3)
S3.P = S2.P; S3.S2 = S2                       # one shared noise hook (set_noise must act on the P that bc2_learn/measure use)
XG = np.linspace(-1, 1, 1201)


def bounded_poly(deg, nodes=None, weights=None, interval=None):
    V = C.chebvander(XG, deg); n = deg + 1
    if nodes is not None:                                   # node-aware LP: vars [c (n), e (m)]
        m = len(nodes); Vn = C.chebvander(nodes, deg); y = nodes.min() / nodes
        cost = np.concatenate([np.zeros(n), weights])
        A = []; bnd = []
        A.append(np.hstack([Vn, -np.eye(m)])); bnd.append(y)            # p - e <= y
        A.append(np.hstack([-Vn, -np.eye(m)])); bnd.append(-y)          # -p - e <= -y
        A.append(np.hstack([V, np.zeros((len(XG), m))])); bnd.append(np.ones(len(XG)))
        A.append(np.hstack([-V, np.zeros((len(XG), m))])); bnd.append(np.ones(len(XG)))
        r = linprog(cost, A_ub=np.vstack(A), b_ub=np.concatenate(bnd), bounds=[(None, None)] * n + [(0, None)] * m, method="highs")
        c = r.x[:n]
    else:                                                   # interval LP: vars [c (n), t]
        lam_min = interval; xs = np.linspace(lam_min, 1.0, 400); Vi = C.chebvander(xs, deg); y = lam_min / xs
        cost = np.concatenate([np.zeros(n), [1.0]])
        A = [np.hstack([Vi, -np.ones((len(xs), 1))]), np.hstack([-Vi, -np.ones((len(xs), 1))]),
             np.hstack([V, np.zeros((len(XG), 1))]), np.hstack([-V, np.zeros((len(XG), 1))])]
        bnd = [y, -y, np.ones(len(XG)), np.ones(len(XG))]
        r = linprog(cost, A_ub=np.vstack(A), b_ub=np.concatenate(bnd), bounds=[(None, None)] * n + [(0, None)], method="highs")
        c = r.x[:n]
    return lambda x: C.chebval(x, c)


def fidelity_poly(a, b, p):
    lams, U = np.linalg.eigh(a); beta = U.conj().T @ b
    x = np.linalg.solve(a, b); x /= np.linalg.norm(x)
    xh = U @ (beta * p(lams)); xh /= np.linalg.norm(xh)
    return float(abs(np.vdot(x, xh))**2)


def run():
    out = {}
    probs = S2.problems(); S3.set_noise(np.inf)
    for name in ("P3 Poisson N=8", "P3 Poisson N=4", "P4 Toeplitz N=8", "P5 random 4x4"):
        a, b = probs[name]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2
        kappa = lams.max() / lams.min()
        n_gap = int(np.ceil(np.log2(2 * 1.25 / max(np.diff(lams).min(), 1e-3)))); n_c = min(n_gap, 10)
        mu, _ = S2.bc2_learn(lams, w, n_c, np.inf, S2.SHOTS, np.random.default_rng(1300))   # learned spectrum (noiseless)
        w_mu = np.array([w[np.argmin(np.abs(lams - x))] for x in mu]); om = np.sqrt(w_mu) / mu**2
        degs = list(range(2, 61, 2))
        F_node, F_node_true, F_int = [], [], []
        for d in degs:
            F_node.append(fidelity_poly(a, b, bounded_poly(d, nodes=mu, weights=om)))
            F_node_true.append(fidelity_poly(a, b, bounded_poly(d, nodes=lams, weights=np.sqrt(w) / lams**2)))
            F_int.append(fidelity_poly(a, b, bounded_poly(d, interval=lams.min())))
        def first(Fs, target):
            for d, F in zip(degs, Fs):
                if F >= target: return d
            return None
        res = {"kappa": float(kappa), "m": int(len(lams)), "markov lower bound sqrt(kappa/2)": float(np.sqrt(kappa / 2)),
               "learned max rel err": float(np.max(np.abs(np.sort(mu)[:len(lams)] - lams) / lams)) if len(mu) == len(lams) else None,
               "degree for F>=0.999": {"node-aware (learned)": first(F_node, 0.999), "node-aware (true)": first(F_node_true, 0.999), "interval (standard)": first(F_int, 0.999)},
               "degree for F>=0.9999": {"node-aware (learned)": first(F_node, 0.9999), "node-aware (true)": first(F_node_true, 0.9999), "interval (standard)": first(F_int, 0.9999)},
               "F vs degree": {str(d): {"node-aware (learned)": round(fn, 5), "node-aware (true)": round(ft, 5), "interval": round(fi, 5)} for d, fn, ft, fi in zip(degs, F_node, F_node_true, F_int)}}
        out[name] = res
        print(name, {k: v for k, v in res.items() if k != "F vs degree"})
        print("   F vs degree (learned / true / interval):", [(d, round(fn, 4), round(ft, 4), round(fi, 4)) for d, fn, ft, fi in zip(degs, F_node, F_node_true, F_int) if d in (2, 4, 6, 8, 10, 14, 20, 30, 40, 60)])
    json.dump(out, open("results/stage5b_results.json", "w"), indent=1)


if __name__ == "__main__":
    run()
