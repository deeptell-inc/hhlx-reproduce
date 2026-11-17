"""Numerical checks for T5_stage1_theory_2026-11-04.md (one runnable check per proposition)."""
import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("pilot", "pilot_bayes_clock.py")
pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
rng = np.random.default_rng(0)


def fim_closed(lams, w, t, theta):                     # Proposition 1
    phi = lams * t + theta
    p = np.sum(w * (1 + np.cos(phi)) / 2)
    v = w * np.sin(phi)
    return t**2 * np.outer(v, v) / (4 * p * (1 - p))


def fim_numeric(lams, w, t, theta, h=1e-6):
    def p(l): return np.sum(w * (1 + np.cos(l * t + theta)) / 2)
    grad = np.array([(p(lams + h * e) - p(lams - h * e)) / (2 * h) for e in np.eye(len(lams))])
    pp = p(lams)
    return np.outer(grad, grad) / (pp * (1 - pp))


# ---- Prop 1: closed-form FIM equals the numerical Bernoulli FIM ----
lams = np.array([0.42, 0.72, 0.83, 0.92]); w = np.array([0.04, 0.344, 0.26, 0.356])
for _ in range(20):
    t, th = rng.uniform(1, 500), rng.uniform(0, 2 * np.pi)
    assert np.allclose(fim_closed(lams, w, t, th), fim_numeric(lams, w, t, th), rtol=1e-5, atol=1e-9)
print("Prop 1 OK: rank-1 closed-form FIM matches finite differences")

# ---- Prop 2: averaged FIM is sandwiched between (t^2/2) W^2 and kappa_p t^2 W^2, off-diagonals vanish ----
T, D = 200.0, 200.0
ts = rng.uniform(T, T + D, 4000); ths = rng.uniform(0, 2 * np.pi, 4000)
Ibar = np.mean([fim_closed(lams, w, t, th) for t, th in zip(ts, ths)], axis=0)
kappa_p = np.mean([1 / (4 * (p := np.sum(w * (1 + np.cos(lams * t + th)) / 2)) * (1 - p)) for t, th in zip(ts, ths)])
t2 = np.mean(ts**2)
lo, hi = t2 / 2 * w**2, kappa_p * t2 * w**2
d = np.diag(Ibar)
assert np.all(d >= 0.9 * lo) and np.all(d <= 1.1 * hi), (d / lo, d / hi)
corr = Ibar / np.sqrt(np.outer(d, d)); off = np.abs(corr - np.eye(len(d))).max()   # normalised cross-information
assert off < 0.2, off
print(f"Prop 2 OK: diag(Ibar)/[(t^2/2)w^2] = {np.round(d/lo,2)}, kappa_p={kappa_p:.2f}, max |corr| off-diag = {off:.3f}")

# ---- Prop 3: 1-F <= ||d||^2/||x_hat||^2 for random vectors ----
for _ in range(200):
    n = rng.integers(2, 9)
    x = rng.normal(size=n) + 1j * rng.normal(size=n); dvec = rng.normal(size=n) * rng.uniform(0, 1) + 0j
    xh = x + dvec
    F = abs(np.vdot(x, xh))**2 / (np.linalg.norm(x)**2 * np.linalg.norm(xh)**2)
    assert 1 - F <= np.linalg.norm(dvec)**2 / np.linalg.norm(xh)**2 + 1e-12
print("Prop 3 OK: fidelity bound holds on 200 random instances")

# ---- Prop 2/6/7 vs the pilot: rms error of tracked eigenvalues against the CRB of the shots actually taken ----
log = []
_measure = pilot.measure
def logged_measure(lams_, w_, t, theta, shots, depol=0.0):
    log.append((t, theta, shots)); return _measure(lams_, w_, t, theta, shots, depol)
pilot.measure = logged_measure
pilot.rng = np.random.default_rng(7); a, b = pilot.random_problem()
lams_t, U = np.linalg.eigh(a); w_t = np.abs(U.conj().T @ b)**2
errs, crbs, Tmax = [], [], []
for rep in range(6):
    log.clear(); pilot.rng = np.random.default_rng(1000 + rep)
    mu0, we, s1, ser = pilot.stage1a(lams_t, w_t, 1.0, 4)
    mu, sig, used, tracked = pilot.stage1b(lams_t, w_t, mu0, we, ser)
    near = np.array([np.argmin(np.abs(mu - l)) for l in lams_t])
    errs.append(mu[near] - lams_t)
    I = sum(s * fim_closed(lams_t, w_t, t, th) for t, th, s in log)
    crbs.append(np.sqrt(np.diag(np.linalg.pinv(I))))
    Tmax.append(max(t for t, _, _ in log))
errs, crb = np.array(errs), np.mean(crbs, axis=0)
rms = np.sqrt(np.mean(errs**2, axis=0))
learn = w_t >= 0.2                                     # clearly tracked components
ratio = rms[learn] / crb[learn]
print(f"Pilot vs CRB (seed 7, 6 repeats): w={np.round(w_t,3)}  rms={rms}  CRB={crb}  rms/CRB={np.round(ratio,2)}  T_max~{np.mean(Tmax):.0f}")
assert np.all(ratio >= 0.5) and np.all(ratio <= 4.0), ratio
# information law sigma ~ c/(w T sqrt(S)) with c = O(1): check c from the CRB itself
S_tot = sum(s for _, _, s in log)
c_est = crb[learn] * w_t[learn] * np.mean(Tmax) * np.sqrt(S_tot)
print(f"Prop 2 scaling: c = CRB * w * T_max * sqrt(S_tot) = {np.round(c_est,2)} (expected O(1), larger than 1 because most shots were at t << T_max)")
assert np.all(c_est > 0.3) and np.all(c_est < 30), c_est
print("ALL CHECKS PASSED")
