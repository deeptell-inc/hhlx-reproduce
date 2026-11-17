"""Detection probability of a planted weak component versus shots: mixture (QMEGS-type) vs projective counting.
Spectrum: 7 fixed well-separated eigenvalues + one planted component of weight w at lambda_p; the remaining
weight is spread over the others with one strong component (w=0.5). Two placements of lambda_p: far from the
strong component (>=4 bins at n_c=5) and adjacent (1.5 bins), the latter exposing the clock-leakage floor.
Detection: a learned eigenvalue within 0.01 of lambda_p (mixture: QMEGS peak), or a coarse-QPE peak within
+-1 bin with >= 3 counts (projective).  Output: results/stage8_detection.json, results/stage8_detection.png
"""
import json, importlib.util, numpy as np
S2 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s2", "stage2_noise_compare.py")); spec.loader.exec_module(S2)
S3 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("s3", "stage3_backend_and_baselines.py")); spec.loader.exec_module(S3); S3.P = S2.P
S3.set_noise(np.inf); P = S2.P
n_c = 5; M = 2**n_c; t0 = 2 * np.pi / 1.25; binw = 2 * np.pi / (M * t0)      # 0.039
others = np.array([0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00])
w_strong_idx = 4                                                          # lambda = 0.75 carries 0.5


def spectrum(w_plant, lam_p):
    lams = np.sort(np.append(others, lam_p)); w = np.full(len(lams), (1 - w_plant - 0.5) / 6)
    w[np.argmin(np.abs(lams - 0.75))] = 0.5; w[np.argmin(np.abs(lams - lam_p))] = w_plant
    return lams, w / w.sum()


def detect_mixture(lams, w, lam_p, S, rng):
    mu, _ = S3.qmegs_learn(lams, w, np.inf, S, rng, m=len(lams))
    return bool(np.min(np.abs(mu - lam_p)) < 0.01)


def detect_projective(lams, w, lam_p, S, rng):
    j, y, M_, t0_ = S2.qpe_sample_labels(lams, w, n_c, np.inf, S, rng)
    counts = np.bincount(y, minlength=M_); yp = int(round(lam_p * M_ * t0_ / (2 * np.pi)))
    # greedy peaks as in bc2_learn
    peaks = []
    for yy in np.argsort(-counts):
        if counts[yy] < 3: break
        if all(min(abs(yy - q), M_ - abs(yy - q)) > 1 for q in peaks): peaks.append(yy)
        if len(peaks) == len(lams): break
    return any(min(abs(yp - q), M_ - abs(yp - q)) <= 1 for q in peaks)


def run():
    out = {}
    for place, lam_p in (("far", 0.375), ("adjacent", 0.81)):             # 0.81 is 1.5 bins from 0.75 (w=0.5)
        rows = []
        for w_plant in (0.001, 0.003, 0.01, 0.03, 0.1):
            lams, w = spectrum(w_plant, lam_p)
            for S in (1000, 10000, 100000):
                dm = np.mean([detect_mixture(lams, w, lam_p, S, np.random.default_rng(10 * r)) for r in range(12)])
                dp = np.mean([detect_projective(lams, w, lam_p, S, np.random.default_rng(10 * r + 1)) for r in range(12)])
                rows.append({"w": w_plant, "S": S, "w_sqrtS": w_plant * np.sqrt(S), "sqrt_wS": np.sqrt(w_plant * S), "P_detect mixture": float(dm), "P_detect projective": float(dp)})
                print(place, w_plant, S, f"w*sqrtS={w_plant*np.sqrt(S):.2f} sqrt(wS)={np.sqrt(w_plant*S):.1f}  mixture={dm:.2f} projective={dp:.2f}", flush=True)
        out[place] = rows
    json.dump(out, open("results/stage8_detection.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for k, place in enumerate(("far", "adjacent")):
        rows = out[place]
        ax[k].semilogx([r["w_sqrtS"] for r in rows], [r["P_detect mixture"] for r in rows], "o", label="mixture (QMEGS-type) vs $w\\sqrt{S}$")
        ax[k].semilogx([r["sqrt_wS"] for r in rows], [r["P_detect projective"] for r in rows], "s", label="projective counting vs $\\sqrt{wS}$")
        ax[k].set_xlabel("detection SNR"); ax[k].set_title(f"planted component {place} from the strong one"); ax[k].grid(alpha=.3)
    ax[0].set_ylabel("detection probability (12 runs)"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("results/stage8_detection.png", dpi=130)


if __name__ == "__main__":
    run()
