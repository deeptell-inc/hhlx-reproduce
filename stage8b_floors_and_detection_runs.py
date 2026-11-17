"""Inputs of Fig. 2 and of the appendix "Detection runs".

floors():          leakage weight of the other components into the +-1 bins of the planted component (5-bit clock, no
                   dephasing) for the two placements of stage 8  ->  results/stage8_floors.json
detection_runs():  Poisson N=8 without dephasing, three runs each at 10^4 and 10^5 shots: which components the
                   MM-QCELS-type, QMEGS-type and projective estimators detect (learned eigenvalue within 0.01)
                                                                 ->  results/adjudication_P1.txt
Run from the repository root:  python stage8b_floors_and_detection_runs.py [floors|runs]   (no argument: both)
"""
import sys, json, importlib.util, numpy as np
load = lambda n, f: (lambda sp: (lambda m: (sp.loader.exec_module(m), m)[1])(importlib.util.module_from_spec(sp)))(importlib.util.spec_from_file_location(n, f))
S2 = load("s2", "stage2_noise_compare.py"); S3 = load("s3", "stage3_backend_and_baselines.py"); S3.P = S2.P; S3.S2 = S2
S4 = load("s4", "stage4_mmqcels_clockopt.py"); S4.P = S2.P; S4.S2 = S2; S4.S3 = S3      # one noise object and one pilot RNG for all three estimators


def floors(path="results/stage8_floors.json"):
    S8 = load("s8", "stage8_detection.py"); n_c = 5; M = 32; t0 = 2 * np.pi / 1.25; rng = np.random.default_rng(0); out = {}
    for place, lam_p in (("far", 0.375), ("adjacent", 0.81)):
        lams, w = S8.spectrum(0.0, lam_p); yp = int(round(lam_p * M * t0 / (2 * np.pi))); leak = 0.0     # planted weight 0: leakage of the others
        for l, ww in zip(lams, w):
            if abs(l - lam_p) < 1e-9: continue
            amp = S3.qpe_amplitudes(l, n_c, np.inf, rng, 1)[0]; pr = np.abs(amp)**2; pr /= pr.sum()
            leak += ww * pr[[(yp - 1) % M, yp, (yp + 1) % M]].sum()
        out[place] = float(leak); print(place, "bin", yp, "leakage weight into +-1 bins (floor):", round(leak, 4))
    json.dump(out, open(path, "w"))


def detection_runs(path="results/adjudication_P1.txt"):
    a, b = S2.problems()["P3 Poisson N=8"]; lams, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; S3.set_noise(np.inf); f = open(path, "w")
    def say(*x): print(*x, flush=True); print(*x, file=f, flush=True)
    def report(tag, mu):
        mu = np.sort(np.asarray(mu)); near = np.array([np.argmin(np.abs(mu - l)) for l in lams]); detected = np.abs(mu[near] - lams) < 0.01
        say(f"{tag}: F={S2.fidelity(a, b, mu):.4f} detected={detected.astype(int)} missed(w,lam)={[(round(float(w[i]), 3), round(float(lams[i]), 3)) for i in np.where(~detected)[0]]}")
    say("true lams", np.round(lams, 4)); say("true w   ", np.round(w, 4))
    for shots in (10_000, 100_000):
        for rep in range(3): report(f"MM-QCELS S={shots:6d} rep{rep}", S4.mm_qcels(lams, w, np.inf, shots, np.random.default_rng(900 + rep)))
        for rep in range(3): report(f"QMEGS    S={shots:6d} rep{rep}", S3.qmegs_learn(lams, w, np.inf, shots, np.random.default_rng(900 + rep))[0])
        for rep in range(3): report(f"BC2      S={shots:6d} rep{rep}", S2.bc2_learn(lams, w, 5, np.inf, shots, np.random.default_rng(900 + rep))[0])
    for shots in (10_000, 100_000):
        S_pt = shots / (2 * 12 * 6)
        say(f"S={shots}: mixture per-point noise 1/sqrt(S_pt)={1 / np.sqrt(S_pt):.3f} vs weights sorted {np.round(np.sort(w), 3)}; projective counts for w=0.032: {0.032 * shots:.0f}+-{np.sqrt(0.032 * shots):.0f}")
    f.close()


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    if len(sys.argv) == 1 or sys.argv[1] == "floors": floors()
    if len(sys.argv) == 1 or sys.argv[1] == "runs": detection_runs()
