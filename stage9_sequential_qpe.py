"""Stage 9: single-copy sequential Bayesian phase estimation ("spectral sampling") for the learning stage.

Idea.  Keep ONE copy of |b> and measure it repeatedly with a single ancilla (Ramsey / Hadamard test of
e^{-iAt}).  With ancilla-only noise every Kraus operator is diagonal in the eigenbasis of A, so the record of a
copy is distributed EXACTLY as if nature drew j ~ w_j once and all outcomes came from lambda_j
(measurement back-action; primitive of Jin-Zhang-Ma 2023).  Hence single-eigenvalue Bayesian estimation
applies verbatim to each copy, and the copies are i.i.d. samples of the spectral measure:
  identify : per copy, exact grid Bayes; every measurement (t, theta) maximises the expected reduction of the
             posterior variance per unit evolution time, in closed form from posterior Fourier moments (see
             identify); stop when sigma <= sig_id (resolution chosen from the gap to be resolved, not from a
             clock size) AND the posterior mass farther than 3 sig_id from the estimate is <= p_out.  p_out is a
             calibrated mislabel probability: it replaces the 1/d^2 tails of a clock register.  After the first
             batch the prior is the learned mixture plus a flat mass u for undiscovered components (a sequential
             test among known components instead of a full search);
  detect   : density peaks of the per-copy estimates -> components (m is NOT assumed), weights = counts;
  refine   : the same copy continues at the frontier of its component's pooled posterior, only while that
             component is above its precision target (need-based allocation, <= n_ref measurements per copy).
Noise: contrast c(t) = (1-p_d) exp(-t/T2) on every Ramsey measurement (same calibrated model as stages 2-8);
optional system-lifetime check: the copy jumps to a fresh eigencomponent (j ~ w) at rate 1/T_sys.
Costs reported: total evolution time, max single evolution time (ancilla coherence), max per-copy time
(system lifetime), copies.   Output: results/stage9_*.json
"""
import json, os, sys, warnings, importlib.util, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # spurious Accelerate BLAS warnings on macOS
S2 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s2", "stage2_noise_compare.py")); spec.loader.exec_module(S2)
PD = S2.PD
LAM_HI = 1.25                                     # prior support (0, 1.25), as the clock range of stages 2-8


def contrast(t, T2):
    t = np.asarray(t, dtype=float)
    return (1 - PD) * (np.exp(-t / T2) if np.isfinite(T2) else np.ones_like(t))


def identify(j, lams, w, T2, sig_id, t_cap, rng, eps=0.02, n_max=300, T_sys=np.inf, p_out=1e-4, prior=None, c_true=None):
    """Vectorised over copies.  j: (B,) component each copy has collapsed to.  Returns per-copy posterior mean,
    converged flag, evolution time spent, and the (possibly jumped) component index.
    Design rule: with q_t = E_post[(lambda - mu) e^{i lambda t}] and z_t = E_post[e^{i lambda t}], a measurement at
    (t, theta) lowers the posterior variance of lambda on average by R = c(t)^2/4 * Re(e^{i theta} q_t)^2 / (P0 P1).
    Each measurement takes the pair (t, theta) that maximises R/t (variance reduction per unit evolution time) over
    the candidate times and 16 phases; c is the contrast of the likelihood used in the update, (1-eps) c(t).  For a
    Gaussian posterior and constant contrast this is sigma t = 0.71 at the fringe slope; for a multimodal posterior
    or an informative prior it is the time and phase that split the surviving modes.  c_true (default: the assumed
    contrast) is the contrast the outcomes are sampled with, for miscalibration tests.
    Once sigma <= sig_id, residual far mass is removed by verification measurements (theta = -mu t, t maximising
    the decay rate of the far mass)."""
    j = j.copy(); B = len(j); g = np.arange(sig_id / 10, LAM_HI, sig_id / 5)
    p0 = np.full(len(g), 1.0 / len(g))
    if prior is not None and len(prior[0]):                            # components found so far + flat mass u for the rest
        cen, wc, u = prior; k = np.clip(np.round((cen - g[0]) / (g[1] - g[0])).astype(int), 0, len(g) - 1)
        p0 = u * p0; np.add.at(p0, k, (1 - u) * wc / wc.sum())
    m0 = p0 @ g; post = np.tile(p0, (B, 1)); mu = np.full(B, m0); sig = np.full(B, np.sqrt(p0 @ (g - m0)**2))
    t_top = min(t_cap, 1.5 / sig_id)                                   # the grid resolves fringes up to ~1.5/sig_id
    tk = t_top / 2**(np.arange(int(2 * np.log2(t_top)) + 1) / 2)       # candidate times, factor sqrt(2) apart, down to ~1
    E = np.exp(1j * tk[:, None] * g[None, :]); Et = E.T
    ck = (1 - eps) * contrast(tk, T2); T = np.zeros(B); act = np.ones(B, bool); ver = np.zeros(B, bool)   # contrast of the tempered likelihood
    ct = contrast(tk, T2) if c_true is None else c_true(tk); eth = np.exp(1j * np.linspace(0, np.pi, 16, endpoint=False))
    short = tk <= 0.5 / sig_id                                         # verification keeps the resolved core in phase
    for _ in range(n_max):
        i = np.flatnonzero(act)
        if not len(i): break
        z = post[i] @ Et; q = (post[i] * (g[None, :] - mu[i][:, None])) @ Et        # E[e^{i lam t}], E[(lam - mu) e^{i lam t}]
        P0 = 0.5 + 0.5 * ck[None, :, None] * np.real(z[:, :, None] * eth)          # (copy, time, phase)
        R = ck[None, :, None]**2 / 4 * np.real(q[:, :, None] * eth)**2 / np.maximum(P0 * (1 - P0), 1e-9) / tk[None, :, None]
        best = np.argmax(R.reshape(len(i), -1), axis=1); ks, th = best // 16, np.angle(eth[best % 16])   # variance reduction per time
        v = np.flatnonzero(ver[i])                                     # core resolved, far mass left: kill it fastest
        if len(v):
            far = np.abs(g[None, :] - mu[i[v]][:, None]) > 3 * sig_id; pf = post[i[v]] * far
            r = np.real((pf @ Et) / pf.sum(1, keepdims=True) * np.exp(-1j * mu[i[v]][:, None] * tk[None, :]))
            kv = np.argmax(np.where(short, -np.log((1 + ck * r) / (1 + ck)) / tk, -np.inf), axis=1)   # far-mass decay rate
            ks[v] = kv; th[v] = -mu[i[v]] * tk[kv]                     # theta = -mu t: the core answers 0 with prob (1+c)/2
        t, c = tk[ks], ck[ks]
        if np.isfinite(T_sys):                                         # system error: lose the eigenstate
            jump = rng.uniform(size=len(i)) > np.exp(-t / T_sys)
            j[i[jump]] = rng.choice(len(lams), size=int(jump.sum()), p=w)
        d = rng.uniform(size=len(i)) < 0.5 + 0.5 * ct[ks] * np.cos(lams[j[i]] * t + th)
        like = 0.5 + 0.5 * c[:, None] * np.real(E[ks] * np.exp(1j * th)[:, None])   # c already contains the tempering
        like = np.where(d[:, None], like, 1 - like)
        new = post[i] * like; new /= new.sum(1, keepdims=True); post[i] = new
        mu[i] = new @ g; sig[i] = np.sqrt(np.maximum(new @ g**2 - mu[i]**2, 1e-30)); T[i] += t
        out = (new * (np.abs(g[None, :] - mu[i][:, None]) > 3 * sig_id)).sum(1)       # posterior mass off the estimate
        act[i] = (sig[i] > sig_id) | (out > p_out)                     # stop only when a wrong fringe is excluded
        ver[i] = (sig[i] <= sig_id) & (out > p_out)
    return mu, ~act, T, j


def cluster(mu, sig_id, n_min=4):
    """Components = density peaks of the per-copy estimates, found by peeling: take the fullest window (+-2 sig_id,
    >= n_min copies), set its centre to the mean of its copies, remove every copy within +-4 sig_id, repeat.
    Returns sorted centres and the label of every copy (-1 = none)."""
    h = sig_id / 2; nb = int(np.ceil(LAM_HI / h)) + 1; x = mu[mu >= 0]; rem = np.ones(len(x), bool); cen = []
    xb = np.clip((x / h).astype(int), 0, nb - 1)                       # bin of every copy
    while True:
        win = np.convolve(np.bincount(xb[rem], minlength=nb), np.ones(9), mode="same")
        k = int(np.argmax(win))
        if win[k] < n_min: break
        c = x[rem & (np.abs(xb - k) <= 4)].mean(); cen.append(c); rem &= np.abs(x - c) > 4 * sig_id   # exactly the copies counted in the window
    cen = np.sort(np.array(cen))
    if not len(cen): return cen, np.full(len(mu), -1)
    lab = np.argmin(np.abs(mu[:, None] - cen[None, :]), axis=1)
    lab[np.abs(mu - cen[lab]) > 4 * sig_id] = -1
    return cen, lab


class Posterior:
    """Pooled 1-D grid posterior of one component (the BC2 refinement rule, shared by all its copies)."""
    def __init__(self, mu, sig):
        self.mu, self.sig, self.n = mu, sig, 0; self._regrid()

    def _regrid(self):
        self.g = np.linspace(self.mu - 25 * self.sig, self.mu + 25 * self.sig, 2001)
        self.p = np.exp(-0.5 * ((self.g - self.mu) / self.sig)**2); self.p /= self.p.sum()

    def update(self, lam_true, T2, t_cap, rng, eps=0.05):
        t = min(0.6 / self.sig, t_cap); th = -self.mu * t - np.pi / 2; c = contrast(t, T2)
        d = rng.uniform() < 0.5 + 0.5 * c * np.cos(lam_true * t + th)
        like = 0.5 + 0.5 * c * np.cos(self.g * t + th); like = (1 - eps) * (like if d else 1 - like) + eps / 2
        new = self.p * like; self.p = new / new.sum(); self.n += 1
        self.mu = float(self.p @ self.g); self.sig = float(np.sqrt(max(self.p @ (self.g - self.mu)**2, 1e-30)))
        if self.sig < 10 * (self.g[1] - self.g[0]): self._regrid()
        return t


def carry_over(old, lab_prev, lab, cen, sig_id, f_min=0.9):
    """Pooled posteriors after re-clustering.  A new cluster inherits an old posterior only if the two are the same
    set of copies: among the copies that already had a label, at least f_min of the new cluster's members come from
    the old cluster AND at least f_min of the old cluster's members are in the new one.  A strong cluster that absorbs
    a weak one therefore keeps its own posterior, a merger of comparable clusters and a split keep none (unless one
    part holds f_min of the members), and everything else starts afresh at the cluster mean.  Returns (posteriors, number inherited, number created)."""
    post, used, n_prev = {}, set(), len(lab_prev); kept = made = 0
    for c, x in enumerate(cen):
        was = lab_prev[lab[:n_prev] == c]; was = was[was >= 0]; k = int(np.bincount(was).argmax()) if len(was) else -1
        if k in old and k not in used and np.mean(was == k) >= f_min and np.mean(lab[:n_prev][lab_prev == k] == c) >= f_min:
            post[c] = old[k]; used.add(k); kept += 1
        else: post[c] = Posterior(float(x), sig_id); made += 1
        if not post[c].n: post[c].mu = float(x)                        # never refined: follow the cluster mean
    return post, kept, made


def learn(lams, w, T2, budget, rng, sig_id, delta=1e-3, t_cap=None, n_ref=16, batch=500, T_sys=np.inf, u=0.05):
    """Run copies one after another until the total evolution time reaches `budget` (never beyond: a copy that would
    not finish is not counted and its time is charged).  Returns learned eigenvalues, weights, stats.
    delta: relative precision target of every component (refinement stops there; 0 = refine on every copy); "auto"
    sets it from the sensitivity of the compliance C = sum_j w_j/lambda_j: the eigenvalue term
    (w_c/lambda_c)^2 delta_c^2 of each component is kept below a quarter of its 1/m share of the multinomial
    variance (sum_j w_j/lambda_j^2 - C^2)/S of the weights.
    Batches hold at most 500 copies and at most a quarter of the remaining time, so that refinement and the update of
    the prior are interleaved with identification also when the budget is small."""
    t_cap = t_cap or (T2 if np.isfinite(T2) else 2000.0)
    MU, J, post, T_tot, T_copy_max, n_ref_meas = np.empty(0), np.empty(0, int), {}, 0.0, 0.0, 0
    lab, cen, est, out_of_time = np.empty(0, int), np.empty(0), 13.0 / sig_id, False
    T_ident = T_refine = 0.0; n_kept = n_made = 0
    while T_tot < budget and not out_of_time:
        left = budget - T_tot; B = int(np.clip(0.25 * left / est, 20, batch))
        j0 = rng.choice(len(lams), size=B, p=w)
        prior = (np.array([post[c].mu for c in sorted(post)]), np.bincount(lab[lab >= 0], minlength=len(post)) + 1.0, u) if post else None
        mu, ok, T, j = identify(j0, lams, w, T2, sig_id, t_cap, rng, T_sys=T_sys, prior=prior)
        keep = int(np.searchsorted(np.cumsum(T), left, side="right"))  # copies completed before the time is up
        if keep < B: mu, ok, T, j, B, out_of_time = mu[:keep], ok[:keep], T[:keep], j[:keep], keep, True
        if B == 0: break
        est = 0.5 * est + 0.5 * float(T.mean())
        mu[~ok] = -1.0                                                 # not converged within n_max: no label
        MU, J = np.append(MU, mu), np.append(J, j)
        lab_prev = lab; cen, lab = cluster(MU, sig_id)
        post, kept, made = carry_over(post, lab_prev, lab, cen, sig_id); n_kept += kept; n_made += made
        T_ref = np.zeros(B); n_c_ = np.bincount(lab[lab >= 0], minlength=len(cen))
        if delta == "auto":                                            # eigenvalue variance of each component <= a quarter of its 1/m share
            mu_c = np.array([post[c].mu for c in range(len(cen))]); wc = n_c_ / max(n_c_.sum(), 1)   # of the multinomial variance of C
            V = max(np.sum(wc / mu_c**2) - np.sum(wc / mu_c)**2, 1e-12)
            tgt = np.clip(0.5 * np.sqrt(V / (len(cen) * max(n_c_.sum(), 1))) * mu_c / np.maximum(wc, 1e-12), 1e-4, 0.3)
        else: tgt = np.full(len(cen), delta)
        for b in range(B):                                             # refinement on the copies of this batch
            c = lab[len(MU) - B + b]
            if c < 0: continue
            for _ in range(n_ref):
                t_next = min(0.6 / post[c].sig, t_cap)
                if post[c].sig <= tgt[c] * post[c].mu or T_tot + T.sum() + T_ref.sum() + t_next > budget: break
                if np.isfinite(T_sys) and rng.uniform() > np.exp(-t_next / T_sys): j[b] = rng.choice(len(lams), p=w)
                T_ref[b] += post[c].update(lams[j[b]], T2, t_cap, rng); n_ref_meas += 1
        T_tot += float(T.sum() + T_ref.sum()); T_copy_max = max(T_copy_max, float((T + T_ref).max())); T_ident += float(T.sum()); T_refine += float(T_ref.sum())
    if out_of_time: T_tot = float(budget)                              # the unfinished copy used up the rest
    mu_hat = np.array([post[c].mu if post[c].n else cen[c] for c in range(len(cen))]); cnt = np.bincount(lab[lab >= 0], minlength=len(cen))
    return mu_hat, cnt / max(int(np.sum(MU >= 0)), 1), {"T_tot": T_tot, "copies": len(MU), "T_copy_max": T_copy_max, "t_max": float(t_cap),
                                              "unlabelled": float(np.mean(lab < 0)) if len(lab) else 0.0, "n_ref_meas": n_ref_meas, "counts": cnt.tolist(), "sig_id": sig_id,
                                              "converged": float(np.mean(MU >= 0)) if len(MU) else 1.0, "weight_sum": float(cnt.sum() / max(int(np.sum(MU >= 0)), 1)),
                                              "T_ident": T_ident, "T_refine": T_refine, "centres": [float(x) for x in cen], "mu": [float(x) for x in mu_hat],
                                              "posteriors_inherited": n_kept, "posteriors_created": n_made}


def sig_id_for(lams, T2, floor=0.0):
    """Identification resolution: an eighth of the smallest gap, not finer than `floor` (a user choice: the cost per
    copy is ~10/sig_id from a flat prior and ~4/sig_id once the components are known) nor than what t <= T2 resolves."""
    s = max(np.diff(lams).min() / 8, floor)
    return float(max(s, 0.6 / T2)) if np.isfinite(T2) else float(s)


FLOOR6 = 0.005                                    # a quarter of the bin of a 6-bit clock register (2 pi / 64 t0 = 0.0195)


def survey(lams, w, T2, budget, rng, sig_id, centres, t_cap=None, batch=500):
    """Identification only (no refinement) on another input state: weights on the given centres, within `budget`."""
    t_cap = t_cap or (T2 if np.isfinite(T2) else 2000.0)
    cnt, n_ok, T_tot, est, out_of_time = np.zeros(len(centres)), 0, 0.0, 13.0 / sig_id, False
    while T_tot < budget and not out_of_time:
        left = budget - T_tot; B = int(np.clip(0.25 * left / est, 20, batch))
        mu, ok, T, _ = identify(rng.choice(len(lams), size=B, p=w), lams, w, T2, sig_id, t_cap, rng, prior=(centres, cnt + 1.0, 0.2))
        keep = int(np.searchsorted(np.cumsum(T), left, side="right"))
        if keep < B: mu, ok, T, out_of_time = mu[:keep], ok[:keep], T[:keep], True
        if not len(T): break
        est = 0.5 * est + 0.5 * float(T.mean())
        lab = np.argmin(np.abs(mu[:, None] - centres[None, :]), axis=1); hit = ok & (np.abs(mu - centres[lab]) <= 4 * sig_id)
        cnt += np.bincount(lab[hit], minlength=len(centres)); n_ok += int(ok.sum()); T_tot += float(T.sum())
    return cnt / max(n_ok, 1), float(budget) if out_of_time else T_tot


def qois(lams, U, b, v, T2, budget, rng, sig_id, delta):
    """Compliance, energy and <v|A^-1|b> from the learned measure; cross weights by polarisation (budget/4 per state)."""
    wt = lambda psi: np.abs(U.conj().T @ psi)**2
    mu, wb, st = learn(lams, wt(b), T2, budget / 4, rng, sig_id, delta)
    bp, bi = b + v, b + 1j * v; np2, ni2 = np.linalg.norm(bp)**2, np.linalg.norm(bi)**2
    sv = [survey(lams, wt(psi / np.linalg.norm(psi)), T2, budget / 4, rng, sig_id, mu) for psi in (v, bp, bi)]
    wv, wp, wi = (x[0] for x in sv); st["T_used"] = st["T_tot"] + sum(x[1] for x in sv)
    c = (wp * np2 - wb - wv) / 2 + 1j * (wi * ni2 - wb - wv) / 2
    return float(np.sum(wb / mu)), float(np.sum(wb / mu) / np.sum(wb / mu**2)), complex(np.sum(c / mu)), mu, wb, st


def truth(a, b, v):
    x = np.linalg.solve(a, b)
    return float(np.real(np.vdot(b, x))), float(np.real(np.vdot(x, a @ x) / np.vdot(x, x))), complex(np.vdot(v, x))


T0 = 2 * np.pi / 1.25


def set_noise(T2):
    """Calibrated contrast used by the projective estimator of stage 2 (module S2)."""
    S2.P.CONTRAST = (lambda t, T2=T2: (1 - PD) * np.exp(-np.asarray(t) / T2)) if np.isfinite(T2) else (lambda t: (1 - PD) + 0 * np.asarray(t))
    S2.P.T_CAP = T2 if np.isfinite(T2) else np.inf


def n_c_for(lams, T2):
    n_gap = int(np.ceil(np.log2(2 * 1.25 / max(np.diff(lams).min(), 1e-3))))
    return max(2, min(n_gap, int(np.floor(np.log2(T2 / T0))) if np.isfinite(T2) else n_gap, 6))


def projective(lams, w, T2, shots, rng, delta=None):
    """Projective estimator of stage 2 with kernel-inverted weights (as stage 7); returns mu, weights, time used."""
    from scipy.optimize import nnls
    S3 = importlib.util.module_from_spec(sp := importlib.util.spec_from_file_location("s3", "stage3_backend_and_baselines.py")); sp.loader.exec_module(S3)
    n_c = n_c_for(lams, T2); S2.T_ACC[0] = 0.0
    mu, _, counts = S2.bc2_learn(lams, w, n_c, T2, shots, rng, return_counts=True, delta=delta); T_used = S2.T_ACC[0]; mu = np.sort(mu)
    K = np.zeros((2**n_c, len(mu)))
    for k_, m_ in enumerate(mu):
        pr = np.abs(S3.qpe_amplitudes(m_, n_c, T2, rng, 200))**2; K[:, k_] = (1 - PD) * (pr / pr.sum(1, keepdims=True)).mean(0) + PD / 2**n_c
    wh, _ = nnls(K, counts / counts.sum())
    return mu, wh / max(wh.sum(), 1e-12), T_used


def _summ(r):
    """Median, worst and the per-repetition values of every recorded quantity."""
    out = {k: float(np.median(v)) for k, v in r.items()}
    out["1-F_max"] = float(np.max(r["1-F"])); out["reps"] = len(r["1-F"]); out["runs"] = {k: [float(x) for x in v] for k, v in r.items()}
    return out


def _compare_one(name, reps=10):
    """Projective estimator (10^4 shots) vs sequential estimator at the same total evolution time, and two ablations
    of the allocation: projective with the precision target (label-only shots once a component is at the target, shot
    number chosen to fill the same time) and sequential without a target (one refinement measurement on every copy)."""
    a, b = S2.problems()[name]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; C0, E0, _ = truth(a, b, b); res = {}
    err = lambda mu, wh: (1 - S2.fidelity(a, b, mu), abs(np.sum(wh / mu) - C0) / C0, abs(np.sum(wh / mu) / np.sum(wh / mu**2) - E0) / E0)
    for T2 in (np.inf, 300.0):
        set_noise(T2); n_c = n_c_for(lams, T2); key = f"T2={'inf' if not np.isfinite(T2) else T2}"; res[key] = {}
        r = {"1-F": [], "eC": [], "eE": [], "T": [], "m": []}
        for k_ in range(reps):
            mu, wh, T = projective(lams, w, T2, 10_000, np.random.default_rng(900 + k_)); f, c, e = err(mu, wh)
            r["1-F"].append(f); r["eC"].append(c); r["eE"].append(e); r["T"].append(T); r["m"].append(len(mu))
        T_ref = float(np.median(r["T"])); res[key][f"projective (n_c={n_c}, 1e4 shots)"] = _summ(r) | {"system_time_per_copy": (2**n_c - 1) * T0 + (T2 if np.isfinite(T2) else 2000.0)}
        r = {"1-F": [], "eC": [], "eE": [], "T": [], "m": [], "shots": []}     # ablation: projective with the target, same time
        for k_ in range(reps):
            lo, hi, shots = 0, None, 10_000                            # largest shot number whose run fits into T_ref (bisection)
            for _ in range(14):
                run = projective(lams, w, T2, shots, np.random.default_rng(900 + k_), delta=1e-3)
                if run[2] <= T_ref: lo, best = shots, run
                else: hi = shots
                if hi is not None and hi - lo <= max(1, lo // 200): break
                shots = min(2 * shots, 600_000) if hi is None else (lo + hi) // 2
                if hi is None and lo >= 600_000: break
            mu, wh, T = best; f, c, e = err(mu, wh)                    # the recorded run is the one executed with `lo` shots
            r["1-F"].append(f); r["eC"].append(c); r["eE"].append(e); r["T"].append(T); r["m"].append(len(mu)); r["shots"].append(lo)
        res[key]["projective with target delta=0.001"] = _summ(r)
        variants = [(sid, frac, delta, 16) for sid in sorted({sig_id_for(lams, T2, FLOOR6), sig_id_for(lams, T2)}, reverse=True) for frac in (1.0, 0.1) for delta in (1e-3, 1e-4)]
        variants.append((sig_id_for(lams, T2, FLOOR6), 1.0, 0.0, 1))    # ablation: sequential without a target
        for sid, frac, delta, n_ref in variants:
            r = {"1-F": [], "eC": [], "eE": [], "T": [], "m": [], "copies": [], "Tcopy": []}; rec = []
            for k_ in range(reps if sid >= FLOOR6 else reps // 2):     # 5 repetitions at the full Toeplitz resolution
                mu, wh, st = learn(lams, w, T2, frac * T_ref, np.random.default_rng(700 + k_), sid, delta, n_ref=n_ref); f, c, e = err(mu, wh)
                r["1-F"].append(f); r["eC"].append(c); r["eE"].append(e); r["T"].append(st["T_tot"]); r["m"].append(len(mu)); r["copies"].append(st["copies"]); r["Tcopy"].append(st["T_copy_max"])
                rec.append({k: st[k] for k in ("T_ident", "T_refine", "converged", "unlabelled", "weight_sum", "counts", "centres", "mu", "posteriors_inherited", "posteriors_created")})
            lab = "no target, 1 refinement per copy" if n_ref == 1 else f"delta={delta}"
            res[key][f"sequential (budget x{frac}, {lab}, sig_id={sid:.4f})"] = _summ(r) | {"T_target": frac * T_ref, "records": rec}
        for k, v in res[key].items(): print(name, key, k, {kk: float(f"{vv:.3g}") for kk, vv in v.items() if isinstance(vv, (int, float))}, flush=True)
    return name, res


def _qoi_one(name, reps=6):
    """Observables at equal total evolution time, six repetitions each: projective estimator (stage 7, time measured
    on the same runs) and sequential estimator with the time budget set to the median projective time."""
    S7 = importlib.util.module_from_spec(sp := importlib.util.spec_from_file_location("s7", "stage7_observable_first.py")); sp.loader.exec_module(S7)
    S7.S3.set_noise(np.inf)
    a, b = S2.problems()[name]; lams, U = np.linalg.eigh(a); rng0 = np.random.default_rng(5)
    for nm in ("P3 Poisson N=8", "P3 Poisson N=4", "P4 Toeplitz N=8", "P5 random 4x4"):   # same |v> as stage 7 (drawn in this order)
        v = rng0.normal(size=len(S2.problems()[nm][0])) + 0j; v /= np.linalg.norm(v)
        if nm == name: break
    C0, E0, L0 = truth(a, b, v); res = {}; e3 = lambda C, E, L: (abs(C - C0) / abs(C0), abs(E - E0) / abs(E0), abs(L - L0) / abs(L0))
    for T2 in (np.inf, 300.0):
        S7.S3.set_noise(T2); n_c = n_c_for(lams, T2); sid = sig_id_for(lams, T2, FLOOR6)
        for shots in (10_000, 160_000):
            p = {"eC": [], "eE": [], "eL": [], "T": []}; q = {"eC": [], "eE": [], "eL": [], "T": [], "copies": []}
            for k_ in range(reps):
                S7.S2.T_ACC[0] = 0.0; C, E, L, _, _ = S7.spectral_qois(lams, U, b, v, n_c, T2, shots, np.random.default_rng(2000 + k_))
                x = e3(C, E, L); p["eC"].append(x[0]); p["eE"].append(x[1]); p["eL"].append(x[2]); p["T"].append(S7.S2.T_ACC[0])
            T_ref = float(np.median(p["T"]))
            for k_ in range(reps):
                C, E, L, mu, wb, st = qois(lams, U, b, v, T2, T_ref, np.random.default_rng(2500 + k_), sid, "auto")
                x = e3(C, E, L); q["eC"].append(x[0]); q["eE"].append(x[1]); q["eL"].append(x[2]); q["T"].append(st["T_used"]); q["copies"].append(st["copies"])
            key = f"T2={'inf' if not np.isfinite(T2) else T2}, shots={shots}, n_c={n_c}"; med = lambda d: {k: float(np.median(v)) for k, v in d.items()} | {"runs": {k: [float(x) for x in v] for k, v in d.items()}}
            res[key] = {"T_tot": T_ref, "sig_id": sid, "projective": med(p), "sequential": med(q)}
            print(name, key, {m: {k: float(f"{v:.3g}") for k, v in res[key][m].items() if k != "runs"} for m in ("projective", "sequential")}, flush=True)
    return name, res


def _run_part(fn, stem):
    """One system per OS process (argv[2] = index); parts are merged by the table scripts."""
    k = int(sys.argv[2]); name, res = fn(list(S2.problems())[k])
    json.dump({name: res}, open(f"results/{stem}.part{k}.json", "w"), indent=1)


def run_compare(): _run_part(_compare_one, "stage9_compare")
def run_qoi(): _run_part(_qoi_one, "stage9_qoi")


def _detect_one(args):
    """One run of the planted-component test (spectrum of stage 8): detection and false clusters at 10^3, 10^4, 10^5 copies."""
    w_plant, lam_p, seed, sizes = args
    others = np.array([0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00])
    lams = np.sort(np.append(others, lam_p)); w = np.full(8, (1 - w_plant - 0.5) / 6)
    w[np.argmin(np.abs(lams - 0.75))] = 0.5; w[np.argmin(np.abs(lams - lam_p))] = w_plant; w /= w.sum()
    sid = sig_id_for(lams, np.inf); rng = np.random.default_rng(seed); MU, T, out, cen = np.empty(0), 0.0, [], np.empty(0)
    for S in sizes:
        while len(MU) < S:
            prior = (cen, np.array([np.sum(np.abs(MU - c) < 4 * sid) for c in cen]) + 1.0, 0.05) if len(cen) else None
            mu, ok, t, _ = identify(rng.choice(8, size=min(1000, S - len(MU)), p=w), lams, w, np.inf, sid, 2000.0, rng, prior=prior)
            mu[~ok] = -1.0; MU = np.append(MU, mu); T += float(t.sum()); cen, _ = cluster(MU, sid)
        out.append({"S": S, "T_tot": T, "detected": bool(len(cen) and np.min(np.abs(cen - lam_p)) < 4 * sid),
                    "false": int(sum(np.min(np.abs(lams - c)) > 4 * sid for c in cen)), "m_hat": len(cen)})
    return out


def run_detect(runs=12, big=4):
    """argv[2] = index of (placement, weight); one OS process each.  All runs go to 10^4 copies, the first `big` to 10^5."""
    combos = [(place, lam_p, w_plant) for place, lam_p in (("far", 0.375), ("adjacent", 0.81)) for w_plant in (0.001, 0.003, 0.01, 0.03)]
    k = int(sys.argv[2]); place, lam_p, w_plant = combos[k]
    rs = [_detect_one((w_plant, lam_p, 50 * r + 7, (1000, 10_000, 100_000) if r < big else (1000, 10_000))) for r in range(runs)]
    rows = []
    for i, S in enumerate((1000, 10_000, 100_000)):
        sub = [r[i] for r in rs if len(r) > i]
        rows.append({"w": w_plant, "S": S, "S_w": S * w_plant, "runs": len(sub), "P_detect": float(np.mean([r["detected"] for r in sub])),
                     "false_clusters_mean": float(np.mean([r["false"] for r in sub])), "T_tot": float(np.mean([r["T_tot"] for r in sub]))})
    for row in rows: print(place, row, flush=True)
    json.dump({place: rows}, open(f"results/stage9_detect.part{k}.json", "w"), indent=1)


def lam_table(mu, M, t0, half):
    """Clock value -> rotation eigenvalue: the nearest learned eigenvalue within half+0.5 bins, else NaN (filtered)."""
    pos = mu * M * t0 / (2 * np.pi); lam_y = np.full(M, np.nan)
    for y in range(M):
        d = np.abs(((y - pos + M / 2) % M) - M / 2); k = int(np.argmin(d))
        if d[k] <= half + 0.5: lam_y[y] = mu[k]
    return lam_y


def backend_fidelity(a, b, n_c, T2, lam_of_y, rng, n_mc, window=None, t0=T0):
    """Joint fidelity of the solver with an n_c-bit clock for the physical dephased ensemble: ONE clock phase-kick
    trajectory per sample acts on all eigencomponents, and the heralded fidelity is E|<x,0|y_s>|^2 / E||y_s||^2.
    Returns (fidelity, Monte Carlo standard error from 8 blocks, success probability for rotation C/lambda with
    C = the smallest rotated eigenvalue)."""
    lams, U = np.linalg.eigh(a); beta = U.conj().T @ b; x = np.linalg.solve(a, b); xb = U.conj().T @ (x / np.linalg.norm(x))
    M = 2**n_c; k = np.arange(M); bits = ((k[:, None] >> np.arange(n_c)) & 1); dur = 2.0**np.arange(n_c) * t0
    g = np.full(M, 1 / np.sqrt(M)) if window is None else window
    cy = np.where(np.isnan(lam_of_y), 0.0, 1.0 / np.where(np.isnan(lam_of_y), 1.0, lam_of_y))
    xi = rng.normal(0, np.sqrt(2 * dur / T2), size=(n_mc, n_c)) if np.isfinite(T2) else np.zeros((1, n_c))
    kick = xi @ bits.T                                                 # (sample, k): the same kicks for every eigencomponent
    amp = np.fft.fft(g[None, None, :] * np.exp(1j * (k[None, None, :] * (lams * t0)[None, :, None] + kick[:, None, :])), axis=2) / np.sqrt(M)
    P = np.abs(amp)**2; num = np.abs((P @ cy) @ (np.conj(xb) * beta))**2; nrm = (P @ cy**2) @ np.abs(beta)**2
    F = float(num.mean() / nrm.mean()); blocks = [num[i::8].mean() / nrm[i::8].mean() for i in range(8)] if len(num) >= 16 else [F]
    return F, float(np.std(blocks) / np.sqrt(len(blocks))), float(nrm.mean() * np.nanmin(lam_of_y)**2)


def run_backend(reps=10):
    """Coherent back end with a real clock.  "clock": spectrum learned by the projective estimator, uniform clock
    (learned angles, optimised t0, no learning, t0 from the exact spectrum).  "window": spectrum learned by the
    sequential estimator, uniform / sine / DPSS clock states, with the rotation half-width as an explicit option."""
    from scipy.signal.windows import dpss
    S4 = importlib.util.module_from_spec(sp := importlib.util.spec_from_file_location("s4", "stage4_mmqcels_clockopt.py")); sp.loader.exec_module(S4)
    out = {}
    for name, (a, b) in S2.problems().items():
        lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; res = {}
        for T2 in (np.inf, 300.0):
            set_noise(T2); n0 = n_c_for(lams, T2); nmc = 256 if np.isfinite(T2) else 1; R = {}
            def put(key, val):
                for suffix, v in zip(("", " |mc", " |p"), val): R.setdefault(key + suffix, []).append(v)
            for k_ in range(reps):
                rng = np.random.default_rng(1500 + k_); M = 2**n0
                mu_p, w_p, _ = projective(lams, w, T2, 10_000, rng); R.setdefault("clock: idealised", []).append(S2.fidelity(a, b, mu_p))
                put("clock: learned angles", backend_fidelity(a, b, n0, T2, lam_table(mu_p, M, T0, 1), rng, nmc))
                t0 = S4.optimise_t0(mu_p, w_p, M); put("clock: +t0", backend_fidelity(a, b, n0, T2, lam_table(mu_p, M, t0, 1), rng, nmc, t0=t0))
                ly = 2 * np.pi * np.arange(M) / (M * T0); put("clock: no learning", backend_fidelity(a, b, n0, T2, np.where(ly >= lams.min(), ly, np.nan), rng, nmc))
                t0s = S4.optimise_t0(lams, w, M); ly = 2 * np.pi * np.arange(M) / (M * t0s)
                put("clock: exact-spectrum t0", backend_fidelity(a, b, n0, T2, np.where(ly >= lams.min() * 0.9, ly, np.nan), rng, nmc, t0=t0s))
                mu, wh, _ = learn(lams, w, T2, 2e6, rng, sig_id_for(lams, T2, FLOOR6))
                for n_c in (n0, n0 + 1):
                    M = 2**n_c; sine = np.sqrt(2 / M) * np.sin(np.pi * (np.arange(M) + 0.5) / M)   # clock state of the original HHL paper
                    wins = {"uniform": None, "sine": sine} | {f"DPSS NW={nw}": dpss(M, nw) / np.linalg.norm(dpss(M, nw)) for nw in (1.5, 2.0)}
                    for wn, half in (("uniform", 1), ("uniform", 2), ("sine", 2), ("DPSS NW=1.5", 2), ("DPSS NW=2.0", 2), ("DPSS NW=2.0", 1)):
                        put(f"window: {wn} +-{half} n_c={n_c}", backend_fidelity(a, b, n_c, T2, lam_table(mu, M, T0, half), rng, nmc, window=wins[wn]))
                    t0 = S4.optimise_t0(mu, wh, M); put(f"window: uniform+t0opt +-1 n_c={n_c}", backend_fidelity(a, b, n_c, T2, lam_table(mu, M, t0, 1), rng, nmc, t0=t0))
            key = f"T2={'inf' if not np.isfinite(T2) else T2}"; res[key] = {k: float(np.median(v)) for k, v in R.items()} | {"n_c": n0, "n_mc": nmc, "reps": reps, "runs": {k: [float(x) for x in v] for k, v in R.items()}}
            print(name, key, {k: round(v, 5) for k, v in res[key].items() if k.startswith(("clock", "window")) and "|" not in k}, flush=True)
        out[name] = res; json.dump(out, open("results/stage9_backend.json", "w"), indent=1)


def run_lifetime(reps=10):
    """System-lifetime requirement: the copy loses its eigenstate at rate 1/T_sys (re-drawn j ~ w). P3 Poisson N=8."""
    a, b = S2.problems()["P3 Poisson N=8"]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; C0 = truth(a, b, b)[0]
    sid = sig_id_for(lams, np.inf); out = {"sig_id": sid}
    for T_sys in (np.inf, 1e6, 3e5, 1e5, 3e4, 1e4):
        r = {"1-F": [], "eC": [], "m": [], "false": [], "unlabelled": []}
        for rep in range(reps):
            mu, wh, st = learn(lams, w, np.inf, 2e6, np.random.default_rng(1700 + rep), sid, T_sys=T_sys)
            r["1-F"].append(1 - S2.fidelity(a, b, mu)); r["eC"].append(abs(np.sum(wh / mu) - C0) / C0); r["m"].append(len(mu))
            r["false"].append(int(sum(np.min(np.abs(lams - c)) > 4 * sid for c in mu))); r["unlabelled"].append(st["unlabelled"])
        out[f"T_sys={T_sys}"] = {k: float(np.median(v)) for k, v in r.items()} | {"1-F_max": float(np.max(r["1-F"])), "false_mean": float(np.mean(r["false"]))}
        print("T_sys", T_sys, out[f"T_sys={T_sys}"], flush=True)
    json.dump(out, open("results/stage9_lifetime.json", "w"), indent=1)


def run_cost(copies=24000):
    """Identification cost and mislabel rate versus the allowed mislabel probability p_out (P3 Poisson N=8), for a
    flat prior, the prior LEARNED by a preceding run of the estimator, and the exact spectrum as prior (reference);
    and the mislabel rate when the true contrast differs from the assumed one."""
    a, b = S2.problems()["P3 Poisson N=8"]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; out = {}
    def stats(T2, p_out, prior, c_true=None, seed=31):
        sid = sig_id_for(lams, T2); cap = T2 if np.isfinite(T2) else 2000.0; rng = np.random.default_rng(seed); err, ok, T = [], [], []
        for _ in range(copies // 2000):
            j = rng.choice(len(lams), size=2000, p=w)
            mu, o, t, _ = identify(j, lams, w, T2, sid, cap, rng, p_out=p_out, prior=prior, c_true=c_true)
            err.append((mu - lams[j]) / sid); ok.append(o); T.append(t * sid)
        err, ok, T = np.concatenate(err), np.concatenate(ok), np.concatenate(T)
        return {"sig_id": sid, "copies": copies, "converged": float(ok.mean()), "T_id*sig_id mean": float(T.mean()), "T_id*sig_id p99": float(np.percentile(T, 99)),
                "rms err/sig_id": float(np.sqrt(np.mean(err[ok]**2))), "frac |err|>3 sig_id": float(np.mean(np.abs(err[ok]) > 3)), "frac |err|>4 sig_id": float(np.mean(np.abs(err[ok]) > 4))}
    for T2 in (np.inf, 300.0):
        sid = sig_id_for(lams, T2); mu_l, w_l, st = learn(lams, w, T2, 5e6, np.random.default_rng(30), sid, "auto")
        priors = {"flat": None, "learned": (mu_l, np.array(st["counts"]) + 1.0, 0.05), "exact": (lams, w, 0.05)}
        out[f"learned prior, T2={'inf' if not np.isfinite(T2) else T2}"] = {"components": len(mu_l), "copies": st["copies"], "T_tot": st["T_tot"]}
        for mode, prior in priors.items():
            for p_out in (1e-2, 1e-3, 1e-4, 1e-5):
                key = f"T2={'inf' if not np.isfinite(T2) else T2}, prior={mode}, p_out={p_out:g}"; out[key] = stats(T2, p_out, prior)
                print(key, {k: float(f"{v:.3g}") for k, v in out[key].items()}, flush=True)
    cases = {"calibrated, T2=300": (300.0, None), "T2 300 -> 100": (300.0, lambda t: (1 - PD) * np.exp(-t / 100.0)),
             "calibrated, no dephasing": (np.inf, None), "p_d 0.02 -> 0.10": (np.inf, lambda t: 0.90 + 0 * t), "p_d 0.02 -> 0.20": (np.inf, lambda t: 0.80 + 0 * t),
             "T2 inf -> 1000": (np.inf, lambda t: (1 - PD) * np.exp(-t / 1000.0))}
    for lab, (T2, c_true) in cases.items():                            # assumed -> true contrast, flat prior
        for p_out in (1e-3, 1e-4):
            key = f"miscalibration: {lab}, p_out={p_out:g}"; out[key] = stats(T2, p_out, None, c_true)
            print(key, {k: float(f"{v:.3g}") for k, v in out[key].items()}, flush=True)
    json.dump(out, open("results/stage9_cost.json", "w"), indent=1)


def run_crb(shots=10_000, n_design=40_000):
    """Local Cramer-Rao bound on the relative error of the compliance C = sum_j w_j/lambda_j from MIXTURE data (fresh
    copy per Hadamard test, both quadratures), for Gaussian time designs t ~ |N(0, T^2)| including the T = 2000 design
    of the QMEGS-type baseline.  Weights live on the simplex; eigenvalues known, or unknown (nuisance parameters)."""
    out = {}
    for name, (a, b) in S2.problems().items():
        lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; m = len(lams); C = np.sum(w / lams); res = {}
        gw = 1 / lams[:-1] - 1 / lams[-1]; gl = -w / lams**2             # gradient of C: free weights (w_m = 1 - sum), eigenvalues
        for T in (30.0, 100.0, 300.0, 1000.0, 2000.0):
            t = np.abs(np.random.default_rng(11).normal(0, T, n_design)); c = contrast(t, np.inf); I = np.zeros((2 * m - 1, 2 * m - 1))
            for th in (0.0, -np.pi / 2):
                ph = np.outer(t, lams) + th; p = 0.5 + 0.5 * c * (np.cos(ph) @ w)
                dw = 0.5 * c[:, None] * (np.cos(ph[:, :-1]) - np.cos(ph[:, -1:])); dl = -0.5 * c[:, None] * w[None, :] * t[:, None] * np.sin(ph)
                G = np.hstack([dw, dl]); I += (G / (p * (1 - p))[:, None]).T @ G
            I *= (shots / 2) / n_design                                 # shots/2 settings per quadrature
            known = np.sqrt(gw @ np.linalg.solve(I[:m - 1, :m - 1], gw)) / C
            g = np.concatenate([gw, gl]); unknown = np.sqrt(g @ np.linalg.solve(I, g)) / C
            res[f"T={T:g}"] = {"eigenvalues known": float(known), "eigenvalues unknown": float(unknown)}
        # c-optimal design for C with known eigenvalues: weights xi over candidate settings (t <= 2000, 16 phases),
        # multiplicative updates xi_i <- xi_i sqrt(d_i), d_i = (g^T I^-1 f_i)^2 / [p_i (1 - p_i)]
        tc = np.concatenate([np.arange(0.25, 300, 0.25), np.arange(300, 2001, 2.5)]); F, V = [], []
        for th in np.arange(16) * np.pi / 16:
            ph = np.outer(tc, lams) + th; p = 0.5 + 0.5 * (1 - PD) * (np.cos(ph) @ w)
            F.append(0.5 * (1 - PD) * (np.cos(ph[:, :-1]) - np.cos(ph[:, -1:]))); V.append(p * (1 - p))
        F, V = np.vstack(F), np.concatenate(V); xi = np.full(len(V), 1.0 / len(V))
        for _ in range(6000):
            h = F @ np.linalg.solve((F * (xi / V)[:, None]).T @ F, gw); xi = xi * np.sqrt(h**2 / V); xi /= xi.sum()
        res["optimised design (eigenvalues known)"] = {"eigenvalues known": float(np.sqrt(gw @ np.linalg.solve((F * (xi / V)[:, None]).T @ F, gw) / shots) / C),
                                                        "settings with 99% of the weight": int(np.searchsorted(np.cumsum(np.sort(xi)[::-1]), 0.99) + 1)}
        out[name] = res; print(name, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in res.items()}, flush=True)
    json.dump({"shots": shots, "systems": out}, open("results/stage9_crb.json", "w"), indent=1)


def run_regress():
    """The stage 2-8 results were produced before two hooks were added to stage2_noise_compare.py (time accumulator,
    optional precision target) and one to stage3 (clock window).  Re-run the projective estimator with the stored
    seeds and compare with results/stage2_results.json; the record is written to results/stage9_regression.json."""
    ref = json.load(open("results/stage2_results.json")); out = {}
    for name in ("P3 Poisson N=8", "P4 Toeplitz N=8", "P3 Poisson N=4", "P5 random 4x4"):
        a, b = S2.problems()[name]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2
        for row in ref[name]["rows"]:
            if row["T2"] not in ("inf", 300.0): continue
            T2 = np.inf if row["T2"] == "inf" else row["T2"]; set_noise(T2)
            F = [S2.fidelity(a, b, S2.bc2_learn(lams, w, row["n_c"], T2, S2.SHOTS, np.random.default_rng(300 + r))[0]) for r in range(3)]
            out[f"{name}, T2={row['T2']}"] = {"stored F_median": row["BC2"]["F_median"], "recomputed": float(np.median(F)), "stored F_min": row["BC2"]["F_min"], "recomputed min": float(np.min(F)),
                                              "identical": bool(np.median(F) == row["BC2"]["F_median"] and np.min(F) == row["BC2"]["F_min"])}
            print(name, row["T2"], out[f"{name}, T2={row['T2']}"], flush=True)
    assert all(v["identical"] for v in out.values()); json.dump(out, open("results/stage9_regression.json", "w"), indent=1)


def run_pairs(reps=10):
    """Toeplitz system, the runs of the matched-time table (seeds 700+rep, sig_id = 0.005, target 1e-3): in how many
    runs are the eigenvalues 0.9214 and 0.9375 reported as two components, and how far is every reported eigenvalue
    from the nearest true one?"""
    a, b = S2.problems()["P4 Toeplitz N=8"]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; out = {}
    T_ref = {k: next(v for kk, v in x.items() if kk.startswith("sequential (budget x1.0, delta=0.001, sig_id=0.0050"))["T_target"]
             for k, x in json.load(open("results/stage9_compare.part3.json"))["P4 Toeplitz N=8"].items()}
    for T2 in (np.inf, 300.0):
        key = f"T2={'inf' if not np.isfinite(T2) else T2}"; sep, dev = [], []
        for k_ in range(reps):
            mu, wh, st = learn(lams, w, T2, T_ref[key], np.random.default_rng(700 + k_), sig_id_for(lams, T2, FLOOR6), 1e-3)
            sep.append(bool(np.min(np.abs(mu - lams[5])) < 0.005 and np.min(np.abs(mu - lams[6])) < 0.005 and np.sum(np.abs(mu - 0.93) < 0.02) == 2))
            dev.append(float(np.max(np.min(np.abs(mu[:, None] - lams[None, :]), axis=1))))
        out[key] = {"pair separated": int(sum(sep)), "runs": reps, "max distance of a reported eigenvalue from the spectrum": float(max(dev))}
        print(key, out[key], flush=True)
    json.dump(out, open("results/stage9_pairs.json", "w"), indent=1)


def run_accept():
    """Acceptance alpha of the 5-bit uniform register in the planted-component test of stage 8: probability that a
    copy of the planted component is read in its peak bin, and in the +-1-bin window, with the leakage floor."""
    M, out = 32, {}; k = np.arange(M)
    for place, lam_p in (("far", 0.375), ("adjacent", 0.81)):
        P = np.abs(np.fft.fft(np.exp(1j * k * lam_p * T0)) / M)**2; y = int(np.argmax(P))
        out[place] = {"lam_p": lam_p, "peak bin": float(P[y]), "+-1 window": float(P[[(y - 1) % M, y, (y + 1) % M]].sum())}
    out["floors"] = json.load(open("results/stage8_floors.json")); print(out)
    json.dump(out, open("results/stage9_accept.json", "w"), indent=1)


def run_manifest():
    """SHA-256 of the code and of every stage-9 result file, with modification times; run after all experiments.
    paper/check_numbers.py asserts that the hashes match and that every result file is newer than the code."""
    import hashlib, glob, time
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    code = ["stage9_sequential_qpe.py", "stage2_noise_compare.py", "stage3_backend_and_baselines.py", "stage4_mmqcels_clockopt.py", "stage7_observable_first.py", "pilot_bayes_clock.py"]
    res = sorted(p for p in glob.glob("results/stage9_*.json") if not p.endswith("manifest.json"))
    tables = {"stage4_mmqcels_clockopt.py": "results/stage4_results.json", "stage6_spectral_correction.py": "results/stage6_results.json",
              "stage8b_floors_and_detection_runs.py": "results/stage8_floors.json"}             # other results quoted in the manuscript, with their scripts
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": {p: {"sha256": h(p), "mtime": os.path.getmtime(p)} for p in code},
           "results": {p: {"sha256": h(p), "mtime": os.path.getmtime(p)} for p in res},
           "tables": {r: {"sha256": h(r), "mtime": os.path.getmtime(r), "script": c, "script_sha256": h(c), "script_mtime": os.path.getmtime(c)} for c, r in tables.items()}
           | {"results/adjudication_P1.txt": {"sha256": h("results/adjudication_P1.txt"), "mtime": os.path.getmtime("results/adjudication_P1.txt"), "script": "stage8b_floors_and_detection_runs.py",
                                              "script_sha256": h("stage8b_floors_and_detection_runs.py"), "script_mtime": os.path.getmtime("stage8b_floors_and_detection_runs.py")}}}
    json.dump(out, open("results/stage9_manifest.json", "w"), indent=1); print("manifest:", len(res), "result files")


def check_reduction():
    """Exact check of the premise: for an adaptive sequence of Ramsey measurements on ONE copy, the probability of every
    record equals sum_j w_j prod_i p(d_i | lambda_j) (state-vector propagation with back-action vs mixture formula)."""
    import itertools
    rng = np.random.default_rng(3); lams = np.array([0.21, 0.5, 0.93]); beta = rng.normal(size=3) + 1j * rng.normal(size=3)
    beta /= np.linalg.norm(beta); w = np.abs(beta)**2
    setting = lambda rec: (1.0 + 2.3 * len(rec) + 1.7 * sum(rec), 0.4 - 0.9 * sum(rec))     # (t, theta) depends on the record so far
    for rec in itertools.product((0, 1), repeat=5):
        psi, mix = beta.copy(), w.copy()
        for i, d in enumerate(rec):
            t, th = setting(rec[:i]); ph = np.exp(1j * (lams * t + th))
            psi = 0.5 * (1 + (-1)**d * ph) * psi                       # Kraus operator of outcome d, diagonal in the eigenbasis
            mix = mix * (0.5 + 0.5 * (-1)**d * np.cos(lams * t + th))
        assert abs(np.vdot(psi, psi).real - mix.sum()) < 1e-12
        # ancilla dephasing: a phase kick +-xi0 with equal probability before every measurement (contrast cos xi0)
        xi0 = 0.7; tot = 0.0; mixn = w.copy()
        for kicks in itertools.product((-xi0, xi0), repeat=len(rec)):
            psi = beta.copy()
            for i, d in enumerate(rec):
                t, th = setting(rec[:i]); psi = 0.5 * (1 + (-1)**d * np.exp(1j * (lams * t + th + kicks[i]))) * psi
            tot += np.vdot(psi, psi).real / 2**len(rec)
        for i, d in enumerate(rec):
            t, th = setting(rec[:i]); mixn = mixn * (0.5 + 0.5 * np.cos(xi0) * (-1)**d * np.cos(lams * t + th))
        assert abs(tot - mixn.sum()) < 1e-12
    print("reduction check ok (32 records, without and with ancilla dephasing)")


def check_identity():
    """Deterministic tests of carry_over: an unchanged cluster keeps its posterior; a weak cluster swallowed by a strong
    one does not pass its posterior on (the merged cluster keeps the strong one's); a split starts afresh."""
    mk = lambda mu: (lambda p: (setattr(p, "n", 5), p)[1])(Posterior(mu, 1e-3))
    prev = np.array([0] * 100 + [1] * 5 + [2] * 50); old = {0: mk(0.30), 1: mk(0.32), 2: mk(0.70)}
    post, kept, made = carry_over(old, prev, np.concatenate([prev, [0, 2]]), [0.30, 0.32, 0.70], 0.01)          # unchanged
    assert post[0] is old[0] and post[1] is old[1] and post[2] is old[2] and (kept, made) == (3, 0)
    merged = np.array([0] * 105 + [1] * 50); post, kept, made = carry_over(old, prev, merged, [0.301, 0.70], 0.01)   # weak 1 merges into strong 0
    assert post[0] is old[0] and post[1] is old[2] and abs(post[0].mu - 0.30) < 1e-12 and (kept, made) == (2, 0)
    prev2 = np.array([0] * 5 + [1] * 100 + [2] * 50); old2 = {0: mk(0.28), 1: mk(0.30), 2: mk(0.70)}                  # the refined posterior is the weak one's
    post, kept, made = carry_over(old2, prev2, merged, [0.2995, 0.70], 0.01)
    assert post[0] is old2[1] and post[0] is not old2[0]
    split = np.array([0] * 60 + [1] * 40 + [2] * 5 + [3] * 50); post, kept, made = carry_over(old, prev, split, [0.295, 0.305, 0.32, 0.70], 0.01)   # cluster 0 splits
    assert post[0] is not old[0] and post[1] is not old[0] and post[0].n == 0 and post[1].n == 0 and post[2] is old[1] and post[3] is old[2]
    even = np.array([0] * 105 + [1] * 50); prev3 = np.array([0] * 58 + [1] * 47 + [2] * 50); old3 = {0: mk(0.30), 1: mk(0.31), 2: mk(0.70)}       # 55:45 merger: neither is the same set
    post, kept, made = carry_over(old3, prev3, even, [0.305, 0.70], 0.01)
    assert post[0] is not old3[0] and post[0] is not old3[1] and post[0].n == 0 and post[1] is old3[2] and (kept, made) == (1, 1)
    chip = np.array([0] * 91 + [1] * 9 + [2] * 5 + [3] * 50); post, kept, made = carry_over(old, prev, chip, [0.30, 0.31, 0.32, 0.70], 0.01)        # 91:9 split: the large part is still the old set
    assert post[0] is old[0] and post[1] is not old[0] and post[1].n == 0
    print("identity check ok (unchanged, absorbed weak cluster, 55:45 merger, 60:40 split, 91:9 split)")


def demo():                                        # self-check: two-level spectrum, weights and eigenvalues recovered
    lams, w = np.array([0.3, 0.8]), np.array([0.2, 0.8])
    mu, wh, st = learn(lams, w, np.inf, 3e5, np.random.default_rng(0), sig_id=0.02)
    assert len(mu) == 2 and np.allclose(mu, lams, rtol=3e-3) and abs(wh[0] - 0.2) < 0.05, (mu, wh)
    print("demo ok", mu, wh, st)


if __name__ == "__main__":
    check_reduction(); check_identity(); demo()
    {"compare": run_compare, "qoi": run_qoi, "detect": run_detect, "backend": run_backend, "lifetime": run_lifetime, "cost": run_cost, "crb": run_crb, "regress": run_regress, "accept": run_accept, "pairs": run_pairs, "manifest": run_manifest}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: None)()
