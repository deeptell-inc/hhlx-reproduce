"""Stage 12: NISQ-era methods and decoded quantum interferometry as further baselines of stages 9-11.

  vqls  variational quantum linear solver (Bravo-Prieto et al.): the same system A x = b, the state from layers of R_Y
        (and R_Z for a complex system) with a CNOT chain, global cost 1 - |<b|A|psi>|^2 / <psi|A^2|psi>.  Problems: stage 2
        (the systems of stage 9), the DC power-flow cases of stage 10, the ridge systems of stage 11; the dimension is
        padded to a power of two with a unit block.
  qrc   quantum reservoir computing (Fujii-Nakajima): 4 qubits, 4 virtual nodes, <Z> features, the ridge readout of stage 11.
  qcl   quantum circuit learning (Mitarai et al.): a 4-qubit circuit that re-uploads the last four inputs, output a <Z_0> + c.
  dqi   decoded quantum interferometry (Jordan et al., arXiv:2408.08292) on an optimal-polynomial-intersection instance:
        a polynomial of degree < N_DQI over F_p maps the level of the last input to the level of the target; the set of
        an input level holds the R_DQI output levels around the median training target.  The output distribution is the
        exact one of the ideal algorithm (amplitude a function of the number of satisfied constraints), by enumeration.

Cost unit: circuit executions (state preparations), against the 10^4 x mult shots of the projective estimator.  Each
method has a row without shot noise (exact expectation values, the ceiling of the method) and rows within that number of
executions.  None of these circuits contains the Hamiltonian evolution to which the dephasing time T2 of stages 2-11
refers, so T2 is not applied; the readout contrast 1 - PD of stage 2 is (calibrated, as there).  Stages 2-11 are not
modified.  Run from the repository root:
    python stage12_nisq_baselines.py                       # self-checks
    python stage12_nisq_baselines.py vqls "P3 Poisson N=8" # also: case9, lorenz, ...; writes results/stage12_vqls.<name>.json
    python stage12_nisq_baselines.py qrc lorenz            # qrc | qcl | dqi <series>; writes results/stage12_<method>.<series>.json
    python stage12_nisq_baselines.py table                 # all rows next to the stored rows of stages 9-11
    python stage12_nisq_baselines.py manifest              # hashes of code and results; sh run_stage12_all.sh runs all of it (~40 min)
"""
import sys, glob, json, importlib.util, numpy as np
from math import comb
from scipy.optimize import minimize


def _load(name, path): m = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location(name, path)); spec.loader.exec_module(m); return m


S11 = _load("s11", "stage11_time_series.py"); GRID = _load("s10grid", "stage10_power_grid.py"); S2 = S11.S9.S2   # stage 11 replaces the problems of its own copy of stage 10
C0, SHOTS, REPS, MULTS = 1 - S2.PD, S2.SHOTS, 10, (1, 10)
est = lambda v, s, rng: (2 * rng.binomial(s, (1 + C0 * np.clip(v, -1, 1)) / 2) / s - 1) / C0     # s-shot estimate of expectation values


# ---------------------------------------------------------------- circuits (batches of state vectors) and optimisers
def _rot(psi, q, n, h, z=False):
    """R_Y (R_Z if z) on qubit q of a batch of n-qubit states; h holds half the rotation angle of every state."""
    p = psi.reshape(len(psi), *(2,) * n); a, b = np.take(p, 0, q + 1), np.take(p, 1, q + 1); sh = (-1,) + (1,) * (n - 1); c, s = np.cos(h).reshape(sh), np.sin(h).reshape(sh)
    return np.stack((a * (c - 1j * s), b * (c + 1j * s)) if z else (c * a - s * b, s * a + c * b), q + 1).reshape(len(psi), -1)


def _cx(n):
    """Permutation of the basis states by a chain of CNOT gates (control q, target q + 1).  With a CZ chain the R_Y
    layers reach only n (n + 1) / 2 of the 2^n - 1 directions of the real states (rank of the tangent space)."""
    b = (np.arange(2**n)[:, None] >> np.arange(n)[::-1]) & 1
    for q in range(n - 1): b[:, q + 1] ^= b[:, q]
    return b @ (1 << np.arange(n)[::-1])


def ansatz(theta, n, cplx):
    """States of the layered circuit for a batch of parameter vectors theta (batch, k n L), k = 2 for a complex system."""
    k = 2 if cplx else 1; psi = np.zeros((len(theta), 2**n), complex); psi[:, 0] = 1; h = theta / 2; cx = _cx(n); j = 0
    for l in range(theta.shape[1] // (k * n)):
        if l: psi = psi[:, cx]
        for q in range(n):
            for z in range(k): psi = _rot(psi, q, n, h[:, j], bool(z)); j += 1
    return psi


def shifted(theta): return np.vstack([theta, theta + np.pi * np.eye(len(theta))])     # d psi / d theta_k = psi(theta + pi e_k) / 2 for these rotations


def spsa(cost, th, K, rng, a=1.0, c=0.2):      # step and perturbation sizes: the better ones of a small trial on the 4 x 4 Poisson system
    for k in range(K):
        d = rng.choice([-1.0, 1.0], len(th)); ck = c / (k + 1)**0.101
        th = th - a / (k + 11)**0.602 * np.clip((cost(th + ck * d) - cost(th - ck * d)) / (2 * ck), -5, 5) * d
    return th


def pauli(M, n):
    """Coefficients Tr(P M) / 2^n of a 2^n x 2^n matrix on the Pauli strings (array of shape (4,)*n, order I X Y Z)."""
    K = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1j, -1j, 0], [1, 0, 0, -1]]) / 2
    T = M.reshape((2,) * (2 * n)).transpose([i for q in range(n) for i in (q, n + q)]).reshape((4,) * n)
    for q in range(n): T = np.moveaxis(np.tensordot(K, T, axes=(1, q)), 0, q)
    return T


# ---------------------------------------------------------------- variational quantum linear solver
def linear_problem(name):
    a, b = (S11.problem(name) if name in S11.SERIES else GRID.problem(name) if name in GRID.CASES else S2.problems()[name])[:2]
    n = max(1, int(np.ceil(np.log2(len(b))))); A = np.eye(2**n, dtype=complex); A[:len(b), :len(b)] = a; B = np.zeros(2**n, complex); B[:len(b)] = b
    return A, B, n, bool(np.abs(A.imag).max() + np.abs(B.imag).max() > 1e-12)


def vqls(name, rng, budget=None):
    """One run from random angles.  budget None: exact cost and gradient, L-BFGS-B; `executions` is then the number of
    (circuit, Pauli term) pairs it evaluates, the executions at one shot each.  Otherwise SPSA within `budget`
    executions: <b|A|psi> from one Hadamard test per Pauli term of A, <psi|A^2|psi> from the Pauli terms of A^2,
    s shots per term with s the largest number that leaves 500 iterations (at least 1).  Returns None when one
    iteration does not fit.  The fidelity of the final state is computed exactly."""
    A, B, n, cplx = linear_problem(name); k = 2 if cplx else 1; q = A @ B; M = A @ A; x = np.linalg.solve(A, B); x /= np.linalg.norm(x)
    P = k * n * (int(np.ceil((k * 2**n - k) / (k * n))) + 1); th = rng.uniform(-np.pi, np.pi, P)       # one layer more than the count of free amplitudes asks for
    al, ga = pauli(A, n).real, pauli(M, n).real; mA, mM = np.abs(al) > 1e-12, np.abs(ga) > 1e-12; mM[(0,) * n] = False; terms = int(k * mA.sum() + mM.sum())
    def exact(t):
        psi = ansatz(shifted(t), n, cplx); p0, dp = psi[0], psi[1:] / 2; beta = np.vdot(q, p0); Mp = M @ p0; D = np.vdot(p0, Mp).real
        return 1 - abs(beta)**2 / D, -(2 * (np.conj(beta) * (dp @ q.conj())).real * D - abs(beta)**2 * 2 * (dp @ Mp.conj()).real) / D**2
    if budget is None:
        r = minimize(exact, th, jac=True, method="L-BFGS-B", options={"maxiter": 4000, "maxfun": 20000, "ftol": 1e-15, "gtol": 1e-11}); th, ex, it = r.x, r.nfev * (P + 1) * terms, r.nit
    else:
        s = max(1, budget // (2 * terms * 500)); K = int(budget // (2 * terms * s)); ex, it = 2 * terms * s * K, K
        if K < 1: return None
        def noisy(t):
            p0 = ansatz(t[None], n, cplx)[0]; tb = 2**n * pauli(np.outer(p0, B.conj()), n)[mA]; u = 2**n * pauli(np.outer(p0, p0.conj()), n).real
            beta = al[mA] @ (est(tb.real, s, rng) + (1j * est(tb.imag, s, rng) if cplx else 0)); D = ga[(0,) * n] + ga[mM] @ est(u[mM], s, rng)
            return 1 - abs(beta)**2 / max(D, 1e-3)
        th = spsa(noisy, th, K, rng)
    psi = ansatz(th[None], n, cplx)[0]; Ap = A @ psi
    return {"1-F": float(1 - abs(np.vdot(x, psi))**2), "cost": float(exact(th)[0]), "executions": int(ex), "iterations": int(it), "terms": terms, "x": psi * np.vdot(Ap, B) / np.vdot(Ap, Ap).real}


def run_vqls(name):
    out = {"problem": name, "qubits": linear_problem(name)[2]}
    for label, budget in [("exact cost", None)] + [(f"{SHOTS * m} executions", SHOTS * m) for m in (MULTS if name in S11.SERIES else MULTS[:1]) + (1000,)]:    # the last row: a thousand times the budget
        runs = [vqls(name, np.random.default_rng(1200 + k), budget) for k in range(REPS)]
        if runs[0] is None: out[label] = {"not runnable": "one SPSA iteration needs more executions"}; continue
        r = {q: [x[q] for x in runs] for q in ("1-F", "cost", "executions", "iterations", "terms")}
        if name in S11.SERIES:
            p = S11.system(name); W = [p["scale"] * x["x"][:S11.N].real for x in runs]; ref = S11.readout(name, np.linalg.eigvalsh(p["a"]))
            r["NMSE"] = [S11.nmse(name, w) for w in W]; r["dev"] = [float(np.mean((p["Zt"] @ (w - ref))**2) / np.var(p["yt"])) for w in W]
        out[label] = _summary(r)
    _write(out, f"results/stage12_vqls.{name.replace(' ', '_')}.json")


def _summary(r): return {q: float(np.median(v)) for q, v in r.items()} | {"runs": r}


def _write(out, path):
    for k, v in out.items():
        if isinstance(v, dict): print(out.get("problem", ""), "|", k, {q: (float(f"{x:.3g}") if isinstance(x, float) else x) for q, x in v.items() if q != "runs"}, flush=True)
    json.dump(out, open(path, "w"), indent=1)


# ---------------------------------------------------------------- forecasting models
def _series(case): u, y, last = S11.SERIES[case](S11.WASH + S11.NTR + S11.NTE); return u, y, last, slice(S11.WASH, S11.WASH + S11.NTR), slice(S11.WASH + S11.NTR, None)


def _fit(case, X):
    """Forecast error of the ridge readout of stage 11 on the features X, and its difference from the forecasts of the echo state network."""
    u, y, last, tr, te = _series(case); p = S11.ridge(X, y, last); f = p["Zt"] @ np.linalg.solve(p["G"] + p["alpha"] * np.eye(X.shape[1]), p["g"]) + p["ym"]
    return _errors(case, f)


def _errors(case, f):
    p = S11.system(case); ref = p["Zt"] @ S11.readout(case, np.linalg.eigvalsh(p["a"])) + p["ym"]
    return {"NMSE": float(np.mean((f - p["yt"])**2) / np.var(p["yt"])), "dev": float(np.mean((f - ref)**2) / np.var(p["yt"]))}


NQ, NV, TAU = 4, 4, 1.0                             # qubits and virtual nodes of the reservoir, time between inputs


def qrc_features(case):
    """<Z_i> of a 4-qubit reservoir at NV times between inputs: qubit 0 is reset to sqrt(1-s)|0> + sqrt(s)|1> with the
    input s scaled to [0, 1], then H = sum_{i<j} J_ij X_i X_j + sum_i Z_i (J uniform in [-1, 1], seed 2) acts for TAU."""
    u = _series(case)[0]; rng = np.random.default_rng(2); s = np.clip((u - u.min()) / (u.max() - u.min()), 0, 1); d = 2**NQ
    op = lambda m, i: np.kron(np.kron(np.eye(2**i), m), np.eye(2**(NQ - i - 1))); X, Z = np.array([[0, 1], [1, 0]]), np.diag([1.0, -1.0])
    H = sum(rng.uniform(-1, 1) * op(X, i) @ op(X, j) for i in range(NQ) for j in range(i + 1, NQ)) + sum(op(Z, i) for i in range(NQ))
    e, V = np.linalg.eigh(H); U = (V * np.exp(-1j * e * TAU / NV)) @ V.conj().T; zs = np.array([np.diag(op(Z, i)) for i in range(NQ)]); rho = np.eye(d) / d; F = np.zeros((len(u), NQ * NV))
    for t in range(len(u)):
        phi = np.array([np.sqrt(1 - s[t]), np.sqrt(s[t])]); rho = np.kron(np.outer(phi, phi), np.einsum("aiaj->ij", rho.reshape(2, d // 2, 2, d // 2)))
        for v in range(NV): rho = U @ rho @ U.conj().T; F[t, v * NQ:(v + 1) * NQ] = zs @ np.diag(rho).real
    return F


def run_qrc(case):
    F = qrc_features(case); n_circ = (S11.NTR + S11.NTE) * NV; out = {"problem": case, "features": F.shape[1], "circuits": n_circ, "exact expectation values": _fit(case, F) | {"executions": None}}
    for m in MULTS:                                # every measured step and virtual node is one circuit; all qubits are read out together
        s = SHOTS * m // n_circ
        if s < 1: out[f"{SHOTS * m} executions"] = {"not runnable": f"{n_circ} circuits to measure"}; continue
        runs = [_fit(case, est(F, s, np.random.default_rng(1200 + k))) for k in range(REPS)]
        out[f"{SHOTS * m} executions"] = _summary({q: [r[q] for r in runs] for q in runs[0]}) | {"shots per circuit": int(s), "executions": int(s * n_circ)}
    _write(out, f"results/stage12_qrc.{case}.json")


LAYERS, BATCH = 3, 100                              # re-uploading layers of the circuit; minibatch of the shot-limited training


def qcl_data(case):
    """The last NQ inputs scaled to [-0.9, 0.9] (training range) and the standardised target; training and test parts."""
    u, y, last, tr, te = _series(case); lo, hi = u[:tr.stop].min(), u[:tr.stop].max(); s = np.clip(1.8 * (u - lo) / (hi - lo) - 0.9, -0.9, 0.9)
    X = np.stack([np.roll(s, j) for j in range(NQ)], 1); m, sd = y[tr].mean(), y[tr].std(); return X[tr], (y[tr] - m) / sd, X[te], m, sd


def qcl_z(theta, X):
    """<Z_0> for every parameter vector (rows of theta) and every sample (rows of X): array (len(theta), len(X))."""
    B, M = len(theta), len(X); psi = np.zeros((B * M, 2**NQ), complex); psi[:, 0] = 1; h = np.repeat(theta / 2, M, 0); ry, rz = np.tile(np.arcsin(X) / 2, (B, 1)), np.tile(np.arccos(X**2) / 2, (B, 1)); cx = _cx(NQ); j = 0
    for l in range(LAYERS):
        for q in range(NQ): psi = _rot(_rot(psi, q, NQ, ry[:, q]), q, NQ, rz[:, q], True)
        psi = psi[:, cx]
        for q in range(NQ):
            for z in range(2): psi = _rot(psi, q, NQ, h[:, j], bool(z)); j += 1
    return (np.abs(psi)**2 @ (1 - 2 * ((np.arange(2**NQ) >> (NQ - 1)) & 1))).reshape(B, M)


def qcl(case, rng, budget=None):
    """One training run.  budget None: exact expectation values and gradient, L-BFGS-B.  Otherwise SPSA on minibatches
    of BATCH samples with two independent shots each: the loss is the mean product of the two one-shot residuals, an
    unbiased estimate of the squared residual (the square of one one-shot residual is not: its expectation carries the
    extra term a^2 (1/C0^2 - <Z>^2), which pulls the output scale a to zero).  A loss value costs 2 BATCH executions, an
    SPSA step two loss values.  The test forecasts use exact expectation values in both cases."""
    Xtr, ytr, Xte, m, sd = qcl_data(case); P = 2 * NQ * LAYERS; th = np.concatenate([rng.uniform(-np.pi, np.pi, P), [1.0, 0.0]])
    def loss(t):                                  # <Z> is quadratic in the state: d<Z>/d theta_k = [<Z>(theta + pi/2 e_k) - <Z>(theta - pi/2 e_k)] / 2
        zp = qcl_z(np.vstack([t[:P], t[:P] + np.pi / 2 * np.eye(P), t[:P] - np.pi / 2 * np.eye(P)]), Xtr); z = zp[0]; res = t[P] * z + t[P + 1] - ytr
        return np.mean(res**2), np.concatenate([2 * t[P] * ((zp[1:P + 1] - zp[P + 1:]) / 2) @ res / len(res), [2 * np.mean(res * z), 2 * np.mean(res)]])
    if budget is None:
        r = minimize(loss, th, jac=True, method="L-BFGS-B", options={"maxiter": 300}); th, ex = r.x, r.nfev * (2 * P + 1) * len(ytr)
    else:
        K = budget // (4 * BATCH); ex = 4 * BATCH * K
        def noisy(t):
            i = rng.integers(0, len(ytr), BATCH); z = qcl_z(t[None, :P], Xtr[i])[0]; res = lambda: t[P] * est(z, 1, rng) + t[P + 1] - ytr[i]
            return np.mean(res() * res())
        th = spsa(noisy, th, K, rng)
    return _errors(case, m + sd * (th[P] * qcl_z(th[None, :P], Xte)[0] + th[P + 1])) | {"training MSE": float(loss(th)[0]), "executions": int(ex)}


def run_qcl(case):
    out = {"problem": case}
    for label, budget in [("exact expectation values", None)] + [(f"{SHOTS * m} executions", SHOTS * m) for m in MULTS]:
        runs = [qcl(case, np.random.default_rng(1200 + k), budget) for k in range(REPS)]; out[label] = _summary({q: [r[q] for r in runs] for q in runs[0]})
    _write(out, f"results/stage12_qcl.{case}.json")


# ---------------------------------------------------------------- decoded quantum interferometry (optimal polynomial intersection)
P_DQI, N_DQI, R_DQI = 31, 5, (5, 15)               # field size, number of coefficients, sizes of the constraint sets (15 of 31 is the balanced case of the paper)
LAG = {"narma2": 1, "mackey-glass": 0, "lorenz": 0} # the input the polynomial reads: the last one the target depends on


def dqi_weights(m, l, p, r):
    """Optimal amplitudes of the degree-l DQI state as a function of the number s of satisfied constraints, and the
    expected s (Lemma 9.2 of Jordan et al.): principal eigenvector of the tridiagonal matrix A^(m,l,d)."""
    d = (p - 2 * r) / np.sqrt(r * (p - r)); k = np.arange(1, l + 1); a = np.sqrt(k * (m - k + 1.0)); e, v = np.linalg.eigh(np.diag(np.arange(l + 1) * d) + np.diag(a, 1) + np.diag(a, -1))
    w = v[:, -1] * np.sign(v[0, -1]); gp, gm = np.sqrt((p - r) / r), -np.sqrt(r / (p - r)); s = np.arange(m + 1)       # g_i on and off its set: mean 0 and mean square 1 over F_p
    ek =lambda k: np.array([sum(comb(int(si), j) * comb(int(m - si), k - j) * gp**j * gm**(k - j) for j in range(k + 1)) for si in s])
    return sum(w[k] * ek(k) / np.sqrt(comb(m, k)) for k in range(l + 1)), m * r / p + np.sqrt(r * (p - r)) / p * e[-1]     # amplitude times p^(n/2)


def dqi_instance(case, r, p=P_DQI, n=N_DQI):
    """Number of satisfied constraints and squared test error (in units of the target variance) of every polynomial."""
    u, y, last, tr, te = _series(case); x = np.roll(u, LAG[case]); lo, hi, ylo, yhi = x[tr].min(), x[tr].max(), y[tr].min(), y[tr].max(); m = p - 1
    lev = lambda x: np.clip(((x - lo) / (hi - lo) * m).astype(int), 0, m - 1); res = lambda y: np.clip(((y - ylo) / (yhi - ylo) * p).astype(int), 0, p - 1); val = ylo + (np.arange(p) + 0.5) * (yhi - ylo) / p
    med = np.array([np.median(res(y[tr])[lev(x[tr]) == i]) if np.any(lev(x[tr]) == i) else np.nan for i in range(m)]); ok = np.flatnonzero(np.isfinite(med))
    c = np.clip(np.round(med[ok[np.argmin(np.abs(np.arange(m)[:, None] - ok[None]), 1)]]).astype(int), r // 2, p - 1 - r // 2)       # an empty level takes the set of the nearest one
    sat = np.abs(np.arange(p)[None] - c[:, None]) <= r // 2; E = np.array([[np.sum((val[v] - y[te][lev(x[te]) == i])**2) for v in range(p)] for i in range(m)]) / (len(y[te]) * np.var(y[te]))
    pts = np.arange(1, p); V = np.zeros((1, m), np.int64)
    for j in range(n - 1): V = ((V[:, None, :] + np.arange(p)[None, :, None] * pow_mod(pts, j, p)[None, None, :]) % p).reshape(-1, m)   # the coefficient of y^j is the j-th digit from the left
    S, ERR, rows = [], [], np.arange(m)[None]
    for cn in range(p): W = (V + cn * pow_mod(pts, n - 1, p)[None]) % p; S.append(sat[rows, W].sum(1).astype(np.uint8)); ERR.append(E[rows, W].sum(1).astype(np.float32))
    return np.stack(S, 1).ravel(), np.stack(ERR, 1).ravel(), float(E[np.arange(m), c].sum())


def pow_mod(y, j, p): return np.array([pow(int(v), j, p) for v in y])


def run_dqi(case):
    out = {"problem": case, "p": P_DQI, "coefficients": N_DQI, "degree of the DQI polynomial": (l := (N_DQI - 1) // 2)}; m = P_DQI - 1
    for r in R_DQI:
        S, ERR, table = dqi_instance(case, r); amp, s_theory = dqi_weights(m, l, P_DQI, r); cnt = np.bincount(S, minlength=m + 1); pr = cnt * amp**2 / P_DQI**N_DQI
        assert abs(pr.sum() - 1) < 1e-9 and abs(pr @ np.arange(m + 1) - s_theory) < 1e-6, (pr.sum(), pr @ np.arange(m + 1), s_theory)     # Lemmas 9.1 and 9.2
        order = np.argsort(S, kind="stable"); start = np.concatenate([[0], np.cumsum(cnt)]); best = S.max(); mean_err = np.bincount(S, weights=ERR, minlength=m + 1) / np.maximum(cnt, 1)
        row = {"set size": r, "satisfied, DQI expectation": float(s_theory / m), "satisfied, random": r / P_DQI, "satisfied, optimum": float(best / m),
               "NMSE, DQI expectation": float(pr @ mean_err), "NMSE, random polynomial": float(ERR.mean()), "NMSE, optimum (mean over the optimal polynomials)": float(mean_err[best]), "NMSE, all constraints met at the centre": table}
        for mult in (0,) + MULTS:                  # one sample, then the best (most constraints satisfied) of 10^4 x mult samples
            runs = {"satisfied": [], "NMSE": []}
            for k in range(REPS):
                rng = np.random.default_rng(1200 + k); sc = rng.choice(m + 1, max(1, SHOTS * mult), p=pr); top = sc.max(); i = order[start[top] + rng.integers(0, cnt[top])]
                runs["satisfied"].append(float(top / m)); runs["NMSE"].append(float(ERR[i]))
            row[f"{max(1, SHOTS * mult)} executions"] = _summary(runs)
        out[f"sets of {r}"] = row
    json.dump(out, open(f"results/stage12_dqi.{case}.json", "w"), indent=1)
    for k, v in out.items():
        if isinstance(v, dict): print(case, "|", k, {q: ({a: float(f"{b:.3g}") for a, b in x.items() if a != "runs"} if isinstance(x, dict) else float(f"{x:.3g}")) for q, x in v.items()}, flush=True)


# ---------------------------------------------------------------- tables
def _g(x): return "-" if x is None else f"{x:.2g}"


def _ref(rows):                                    # stored medians of stages 9-11 at 10^4 projective shots: projective, and the sequential row (delta 1e-3) with the lowest 1-F
    return ([r["1-F"] for k, r in rows.items() if k.startswith("projective (")][0], min(r["1-F"] for k, r in rows.items() if k.startswith("sequential (budget x1.0, delta=0.001")))


def table():
    P2 = {k: v for f in sorted(glob.glob("results/stage9_compare.part*.json")) for k, v in json.load(open(f)).items()}
    print("Linear systems: median 1-F over ten runs.  Projective and sequential: stored rows at 10^4 projective shots (T2 = inf / 300).  VQLS: no T2 (see the module text).\n")
    print("| system | qubits | Pauli terms | projective | sequential | VQLS, exact cost | its (circuit, term) pairs | " + " | ".join(f"VQLS, {SHOTS * m:.0e} executions" for m in MULTS + (1000,)) + " |\n|" + "---|" * 10)
    for name in list(P2) + list(GRID.CASES) + list(S11.SERIES):
        v = json.load(open(f"results/stage12_vqls.{name.replace(' ', '_')}.json"))
        fn = f"results/stage{11 if name in S11.SERIES else 10}_compare.{name}.json"; ref = P2[name] if name in P2 else json.load(open(fn))["6-bit clock"] if glob.glob(fn) else None
        ref = [_ref(ref[k]) if ref else (None, None) for k in ("T2=inf", "T2=300.0")]              # stage 10 has no learning runs on 118 buses (its spectrum is not sparse)
        cell = lambda m: _g(v.get(f"{SHOTS * m} executions", {}).get("1-F")) if "not runnable" not in v.get(f"{SHOTS * m} executions", {}) else "not runnable"
        print(f"| {name} | {v['qubits']} | {int(v['exact cost']['terms'])} | {_g(ref[0][0])} / {_g(ref[1][0])} | {_g(ref[0][1])} / {_g(ref[1][1])} | {_g(v['exact cost']['1-F'])} | {_g(v['exact cost']['executions'])} | " + " | ".join(cell(m) for m in MULTS + (1000,)) + " |")
    print("\nForecasts: median over ten runs; `difference` is the mean squared difference from the forecasts of the exact ridge readout of the echo state network, in units of the target variance.\n")
    print("| series | method | setting | executions | 1-F | forecast NMSE | difference |\n|" + "---|" * 7)
    fc = json.load(open("results/stage11_forecast.json"))
    for case in S11.SERIES:
        rows = [("echo state network, exact ridge readout", "kappa <= 100", None, 0.0, fc[case]["ridge at this kappa"], 0.0), ("last known value", "", None, None, fc[case]["persistence"], None)]
        for key in ("T2=inf", "T2=300.0"):
            for tag, x10 in (("10^4 projective shots", False), ("ten times that time", True)):
                for m in ("projective", "sequential"):
                    c = [(k, r) for k, r in fc[case].items() if isinstance(r, dict) and k.startswith(f"{key}, {m}") and k.endswith("x10 time") == x10 and ("x10" in k or " (" in k and "x0.1" not in k and "delta=0.0001" not in k and "no target" not in k and "with target" not in k)]
                    k, r = min(c, key=lambda kr: kr[1]["1-F"]); rows.append((f"{m} HHL", f"{key}, {tag}" + (f", {k.split('sig_id=')[1][:6]}" if "sig_id" in k else ""), None, r["1-F"], r["NMSE"], r["dev"]))
        for meth in ("vqls", "qrc", "qcl"):
            for k, r in json.load(open(f"results/stage12_{meth}.{case}.json")).items():
                if isinstance(r, dict): rows.append((meth.upper(), k, r.get("executions"), r.get("1-F"), r.get("NMSE"), r.get("dev")) if "not runnable" not in r else (meth.upper(), k, None, None, None, None))
        for k, r in json.load(open(f"results/stage12_dqi.{case}.json")).items():
            if isinstance(r, dict):
                rows += [(f"DQI, {k}", f"expectation of one sample (satisfied {r['satisfied, DQI expectation']:.2f}, random {r['satisfied, random']:.2f}, optimum {r['satisfied, optimum']:.2f})", 1, None, r["NMSE, DQI expectation"], None)]
                rows += [(f"DQI, {k}", f"best of the samples (satisfied {x['satisfied']:.2f})", int(kk.split()[0]), None, x["NMSE"], None) for kk, x in r.items() if kk.endswith("executions") and kk != "1 executions"]
                rows += [(f"optimal polynomial, {k}", "exhaustive search", None, None, r["NMSE, optimum (mean over the optimal polynomials)"], None), (f"all constraints met, {k}", "", None, None, r["NMSE, all constraints met at the centre"], None)]
        for meth, setting, ex, f1, e, dv in rows: print(f"| {case} | {meth} | {setting} | {_g(ex)} | {_g(f1)} | {'not runnable' if e is None else _g(e)} | {_g(dv)} |")


def run_manifest():
    """SHA-256 of this script, of stage 11 (imported for its problems and its ridge readout) and of every stage-12
    result file, with modification times (checked by paper/check_numbers.py; the manifests of stages 9 and 10 cover
    the other imported scripts and the grid data)."""
    import hashlib, os, time
    ent = lambda p: {"sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "mtime": os.path.getmtime(p)}
    res = sorted(p for p in glob.glob("results/stage12_*.json") if not p.endswith("manifest.json"))
    json.dump({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": {p: ent(p) for p in ("stage12_nisq_baselines.py", "stage11_time_series.py")}, "results": {p: ent(p) for p in res}},
              open("results/stage12_manifest.json", "w"), indent=1); print("stage-12 manifest:", len(res), "result files")


# ---------------------------------------------------------------- self-checks
def demo():
    rng = np.random.default_rng(0); n = 3; M = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)); c = pauli(M, n)
    z, a_, b_, y = 0.3, 4.0, -0.2, 0.7; d1, d2 = est(np.full(400_000, z), 1, rng), est(np.full(400_000, z), 1, rng); r1, r2 = a_ * d1 + b_ - y, a_ * d2 + b_ - y
    assert abs(np.mean(r1 * r2) - (a_ * z + b_ - y)**2) < 0.1 and abs(np.mean(r1**2) - (a_ * z + b_ - y)**2 - a_**2 * (1 / C0**2 - z**2)) < 0.1   # the shot-limited loss of the circuit learning
    mats = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    assert np.allclose(sum(c[i, j, k] * np.kron(np.kron(mats[i], mats[j]), mats[k]) for i in range(4) for j in range(4) for k in range(4)), M)       # Pauli decomposition
    th = rng.uniform(-3, 3, 12); psi = ansatz(shifted(th), 3, True); eps = 1e-6; fd = (ansatz((th + eps * np.eye(12)[5])[None], 3, True)[0] - psi[0]) / eps
    assert np.allclose(np.linalg.norm(psi, axis=1), 1) and np.allclose(psi[6] / 2, fd, atol=1e-5)                                                       # shift rule of the state
    for n, c in ((4, False), (3, True)): J = ansatz(shifted(rng.uniform(-3, 3, (2 if c else 1) * n * 6)), n, c)[1:]; assert np.linalg.matrix_rank(np.hstack([J.real, J.imag]), tol=1e-8) == (2 if c else 1) * 2**n - 1   # the circuit reaches every direction
    r = vqls("P3 Poisson N=4", np.random.default_rng(1)); assert r["1-F"] < 1e-8 and r["cost"] < 1e-10, r                                             # a well-conditioned system is solved
    A, B, n, _ = linear_problem("case14"); assert A.shape == (16, 16) and np.allclose(np.linalg.solve(A, B)[13:], 0)                                  # padding keeps the solution
    p, nn, r, m = 7, 3, 2, 6; sat = (np.arange(p)[None] - rng.integers(0, p, m)[:, None]) % p < r; xs = np.stack(np.meshgrid(*[np.arange(p)] * nn, indexing="ij"), -1).reshape(-1, nn)
    s = sat[np.arange(m)[None], (xs @ np.array([pow_mod(np.arange(1, p), j, p) for j in range(nn)])) % p].sum(1); amp, mean = dqi_weights(m, 1, p, r); pr = amp[s]**2 / p**nn
    assert abs(pr.sum() - 1) < 1e-12 and abs(pr @ s - mean) < 1e-12 and mean > m * r / p                                                                # DQI state: unit norm and the expectation of Lemma 9.2
    t = rng.uniform(-3, 3, 2 * NQ * LAYERS); X = rng.uniform(-0.9, 0.9, (5, NQ)); z = qcl_z(np.vstack([t, t + np.pi / 2 * np.eye(len(t))[3], t - np.pi / 2 * np.eye(len(t))[3]]), X)
    assert np.all(np.abs(z) <= 1) and np.allclose((z[1] - z[2]) / 2, (qcl_z((t + eps * np.eye(len(t))[3])[None], X)[0] - z[0]) / eps, atol=1e-5)       # parameter-shift rule of <Z>
    print("stage 12 self-check passed")


if __name__ == "__main__":
    if len(sys.argv) == 1: demo()
    elif sys.argv[1] == "table": table()
    elif sys.argv[1] == "manifest": run_manifest()
    else: {"vqls": run_vqls, "qrc": run_qrc, "qcl": run_qcl, "dqi": run_dqi}[sys.argv[1]](sys.argv[2])
