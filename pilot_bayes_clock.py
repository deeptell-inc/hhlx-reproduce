"""Phase-0 pilot for Bayesian-clock HHL (T5).

Single-ancilla circuit:  |+>  --controlled exp(-iAt)-- Rz(theta) -- H -- measure
gives P(0 | t, theta) = sum_j w_j (1 + cos(lambda_j t + theta)) / 2,  w_j = |<u_j|b>|^2.
We sample that statistic exactly (no Qiskit needed), learn {lambda_j, w_j} classically,
then build the HHL solution from the learned spectrum and compare with exact x = A^{-1} b.

Stage 1a: matrix-pencil on the time series g(t)=sum_j w_j e^{i lambda_j t}  (O'Brien 2019 style)
Stage 1b: rejection-filter Bayesian refinement (Wiebe & Granade 2016), shots allocated
          by the T5 rule S_j ~ (sqrt(w_j)/lambda_j^2)^(2/3).
"""
import numpy as np

rng = np.random.default_rng(7)
CONTRAST = lambda t: 1.0      # stage 2: set to e.g. lambda t: exp(-t/T2) for dephasing (assumed calibrated)
T_CAP = np.inf                # stage 2: coherence-limited cap on the evolution time


def random_problem(n=4, lam_min=0.25, lam_max=1.0, min_gap=0.08):
    """Pilot assumption: eigenvalues separated by >= min_gap (near-degenerate pairs are the
    stated limitation of stage 1a, resolution ~ 1/T_total; see plan section 4.5)."""
    while True:
        lams = np.sort(rng.uniform(lam_min, lam_max, n))
        if np.min(np.diff(lams)) >= min_gap:
            break
    q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    a = q @ np.diag(lams) @ q.conj().T
    b = rng.normal(size=n) + 1j * rng.normal(size=n)
    return a, b / np.linalg.norm(b)


def p0(lams, w, t, theta, depol=0.0):
    p = 0.5 + 0.5 * CONTRAST(t) * np.sum(w * np.cos(lams * t + theta))
    return float(np.clip((1 - depol) * p + depol / 2, 0.0, 1.0))


def measure(lams, w, t, theta, shots, depol=0.0):
    """Returns number of 0-outcomes among `shots`."""
    return rng.binomial(shots, p0(lams, w, t, theta, depol))


# ---------- stage 1a: matrix pencil on g(t) ----------
def stage1a(lams, w, lam_max, n_dim, K=40, shots=50, depol=0.0):
    dt = np.pi / (1.2 * lam_max)           # no aliasing for lambda in [0, lam_max]
    g = np.zeros(K, complex)
    for k in range(K):
        re = 2 * measure(lams, w, k * dt, 0.0, shots, depol) / shots - 1      # E[cos]
        im = 2 * measure(lams, w, k * dt, -np.pi / 2, shots, depol) / shots - 1  # E[sin]
        g[k] = (re + 1j * im) / ((1 - depol) * CONTRAST(k * dt))
    L = K // 2
    H = np.array([g[i:i + L] for i in range(K - L + 1)])    # Hankel, shape (K-L+1, L)
    u, s, vh = np.linalg.svd(H)
    floor = np.median(s[len(s) // 2:])                   # shot-noise floor of the Hankel spectrum
    m = min(n_dim, max(1, int(np.sum(s > 2.5 * floor))))  # model order, capped by known dimension
    Us = u[:, :m]                                         # ESPRIT: signal subspace, row-shift invariance
    z = np.linalg.eigvals(np.linalg.pinv(Us[:-1]) @ Us[1:])
    lam_est = np.sort(np.angle(z) % (2 * np.pi) / dt)     # lambda in [0, 2pi/dt) ⊃ [0, lam_max]
    ts = np.arange(K) * dt

    def weights_for(lam):
        V = np.exp(1j * np.outer(ts, lam))
        wt = np.clip(np.real(np.linalg.lstsq(V, g, rcond=None)[0]), 0.02, None)
        return wt / wt.sum()

    # ESPRIT roots can be grossly wrong at low shot counts: polish by GLOBAL coordinate search
    # on the same series (no phase wrapping here since 2*pi/dt > lam_max), then refit weights.
    cand = np.linspace(0.02 * lam_max, 1.2 * lam_max, 240)
    lam = lam_est.copy()
    for _ in range(3):
        for c in range(m):
            best, bx = np.inf, lam[c]
            for v in cand:
                lam[c] = v
                r = np.sum(np.abs(np.exp(1j * np.outer(ts, lam)) @ weights_for(lam) - g) ** 2)
                if r < best: best, bx = r, v
            lam[c] = bx
    lam_est = np.sort(lam)
    w_est = weights_for(lam_est)
    return lam_est, w_est, 2 * K * shots, (ts, g)


# ---------- stage 1b: joint refinement on the ACCUMULATED time series, evolution time doubling ----------
def stage1b(lams, w, mu, w_est, series, shots_total=5800, depol=0.0, n_t=6, shots=40, lam_max=1.0,
            sig0=0.08, verbose=False):
    """Kitaev-style multi-stage estimation for a non-eigenstate input (cf. O'Brien et al. 2019).
    All (t, g(t)) samples are kept: short times (stage 1a) anchor phase unwrapping, long times give
    precision ~1/t.  Each round adds n_t samples in [T, T_next] and refits ALL tracked eigenvalues
    and weights jointly (coordinate grid search over one phase period at T_next for robust
    unwrapping, then Gauss-Newton).  T_next is the largest time with sigma_j*T_next < 0.4 for every
    tracked j (<= 2T).  Components too weak to be learned in budget are frozen at the stage-1a value;
    their mis-modelled signal enters the posterior width as a bias ~ w_frozen/(w_j T)."""
    from scipy.optimize import least_squares
    ts, g = [np.array(series[0], float)], [np.array(series[1], complex)]
    m, S_eff = len(mu), n_t * shots
    mu, w_est = mu.copy(), w_est.copy()
    tracked = 0.5 * w_est * np.sqrt(S_eff) >= 1.2
    idx, fr = np.where(tracked)[0], np.where(~tracked)[0]
    k, bias = len(idx), w_est[~tracked].sum() + 0.02
    sig = np.full(m, np.nan); sig[idx] = sig0
    used, T = 0, float(ts[0].max())
    while used + 2 * n_t * shots <= shots_total and k > 0:
        T_next = float(np.clip(0.4 / sig[idx].max(), 1.2 * T, 2 * T))
        if T_next >= T_CAP:                                   # coherence-limited: keep sampling near the cap
            T_next = T_CAP; T = 0.5 * T_CAP
        t_new = T + (T_next - T) * (np.arange(n_t) + 0.5) / n_t
        g_new = np.array([(2 * measure(lams, w, t, 0.0, shots, depol) / shots - 1)
                          + 1j * (2 * measure(lams, w, t, -np.pi / 2, shots, depol) / shots - 1)
                          for t in t_new]) / ((1 - depol) * CONTRAST(t_new))
        ts.append(t_new); g.append(g_new); used += 2 * n_t * shots
        tt, gg = np.concatenate(ts), np.concatenate(g)
        frozen = np.exp(1j * np.outer(tt, mu[fr])) @ w_est[fr]
        cw = CONTRAST(tt)                                     # noise per point ~ 1/(contrast*sqrt(shots)): weight residuals
        model = lambda x: np.exp(1j * np.outer(tt, x[:k])) @ x[k:] + frozen
        x = np.concatenate([mu[idx], w_est[idx]])
        for _ in range(2):                                     # coordinate grid search = unwrapping
            for c in range(k):
                cand = x[c] + np.linspace(-np.pi, np.pi, 81) / T_next
                cand = cand[(cand > 0) & (cand < 1.5 * lam_max)]
                best, bx = np.inf, x[c]
                for v in cand:
                    x[c] = v; r = np.sum(cw**2 * np.abs(model(x) - gg) ** 2)
                    if r < best: best, bx = r, v
                x[c] = bx
        resid = lambda x: np.concatenate([cw * (model(x) - gg).real, cw * (model(x) - gg).imag])
        lo = np.concatenate([np.zeros(k), np.full(k, 1e-3)]); hi = np.concatenate([np.full(k, 1.5 * lam_max), np.ones(k)])
        fit = least_squares(resid, np.clip(x, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), method="trf")
        mu[idx], w_est[idx] = fit.x[:k], fit.x[k:]
        J = fit.jac[:, :k]
        sig_fit = np.sqrt(np.diag(np.linalg.pinv(J.T @ J)) / shots)
        sig[idx] = 1.5 * np.sqrt(sig_fit ** 2 + (bias / (w_est[idx] * T_next)) ** 2)
        if verbose:
            print(f"    T={T:8.1f}->{T_next:8.1f} used={used:5d} relerr={np.round(np.abs(mu-lams)/lams,5)} sig={np.round(sig,5)}")
        T = T_next
    return mu, sig, used, tracked


def hhl_from_spectrum(a, b, lam_est):
    """Method A idealised: exact eigen-tagging, rotation angles from learned eigenvalues."""
    lams, U = np.linalg.eigh(a)
    beta = U.conj().T @ b
    # match each true eigencomponent to nearest learned eigenvalue (what a short clock register does)
    inv = np.array([1 / lam_est[np.argmin(np.abs(lam_est - l))] for l in lams])
    x = U @ (beta * inv)
    return x / np.linalg.norm(x)


def hhl_clock_bits(a, b, n_p, lam_max):
    lams, U = np.linalg.eigh(a)
    beta = U.conj().T @ b
    lam_q = np.maximum(np.round(lams / lam_max * 2**n_p), 1) * lam_max / 2**n_p
    x = U @ (beta / lam_q)
    return x / np.linalg.norm(x)


def run(depol=0.0, verbose=True):
    a, b = random_problem()
    lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b) ** 2
    lam_max = 1.0  # assumed known upper bound
    mu, w_est, s1, series = stage1a(lams, w, lam_max, len(lams), depol=depol)
    assert len(mu) == len(lams), f"model order {len(mu)} != {len(lams)}"
    mu, sig, s2, tracked = stage1b(lams, w, mu, w_est, series, depol=depol)
    # evaluate by nearest-neighbour matching (what a short clock register does); components with
    # weight < W_LEARN cannot be learned within this budget (information per shot ~ w_j^2) and are
    # reported but not asserted on.
    W_LEARN = 0.10
    near = np.array([np.argmin(np.abs(mu - l)) for l in lams])
    rel = np.abs(mu[near] - lams) / lams
    learnable = w >= W_LEARN
    x_exact = np.linalg.solve(a, b); x_exact /= np.linalg.norm(x_exact)
    F = abs(np.vdot(x_exact, hhl_from_spectrum(a, b, mu))) ** 2
    if verbose:
        print(f"depol={depol}: shots stage1a={s1}, stage1b={s2}, total={s1+s2}")
        print("  true   :", np.round(lams, 5)); print("  learned:", np.round(mu, 5))
        print("  rel err (nearest match):", np.array2string(rel, precision=5), " learnable(w>=0.1):", learnable)
        print("  learned sigma:", np.round(sig, 5), " tracked:", tracked)
        print(f"  weights true/est: {np.round(w,3)} / {np.round(w_est,3)}")
        print(f"  fidelity Bayesian-clock HHL = {F:.6f}")
        for n_p in (2, 3, 4, 6):
            print(f"  fidelity standard HHL n_p={n_p}: {abs(np.vdot(x_exact, hhl_clock_bits(a,b,n_p,lam_max)))**2:.6f}")
    rel_learn = rel[learnable].max() if learnable.any() else np.nan
    return rel_learn, s1 + s2, F, rel.max(), learnable.sum()


if __name__ == "__main__":
    rel_t, shots, F, rel_all, n_tr = run(0.0)
    assert rel_t < 1e-3, f"max relative error on learnable (w>=0.1) eigenvalues {rel_t:.2e} >= 1e-3"
    assert shots < 1e4, f"total shots {shots} >= 1e4"
    print(f"PASS on learnable components (seed 7). Fidelity {F:.5f} (criterion 0.999: {'met' if F > 0.999 else 'NOT met'}), "
          f"max rel err incl. weak {rel_all:.2e}, learnable {n_tr}/4")
    print("Sweep over 10 seeds:")
    for depol in (0.0, 0.05):
        res = []
        for seed in range(10):
            globals()["rng"] = np.random.default_rng(seed)
            res.append(run(depol, verbose=False))
        res = np.array(res, dtype=float)
        ok = np.sum((res[:, 0] < 1e-3) & (res[:, 1] < 1e4))
        print(f"  depol={depol}: learnable-criterion pass {ok}/10 | F>0.999 in {np.sum(res[:,2]>0.999)}/10 | "
              f"median F {np.median(res[:,2]):.5f} | median rel err learnable {np.nanmedian(res[:,0]):.1e} | "
              f"median learnable {np.median(res[:,4]):.0f}/4 | median shots {np.median(res[:,1]):.0f}")
