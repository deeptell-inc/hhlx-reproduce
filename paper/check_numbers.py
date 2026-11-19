"""Spot check of the stage-9 numbers quoted in the prose of main.tex against the result files (run from paper/).
Each quoted value must equal the stored value rounded to the quoted number of significant figures; ranges must
contain every stored value they summarise; every matched-time run must have used no more than its time budget.
This is a hand-written list of the quantitative statements of the abstract and Secs. II-VI on the sequential
estimator and the tapered clock, not an automatic extraction of every number in the text."""
import json, os, numpy as np
from make_tables import load, _cmp

C, Q, D = load("stage9_compare"), load("stage9_qoi"), load("stage9_detect")
B, L, K, CRB, ACC, REG, PAIRS = (json.load(open(f"../results/stage9_{n}.json")) for n in ("backend", "lifetime", "cost", "crb", "accept", "regression", "pairs"))
P8, P4, R4, TZ = "P3 Poisson N=8", "P3 Poisson N=4", "P5 random 4x4", "P4 Toeplitz N=8"
SYS = (R4, P4, P8, TZ); T2S = ("inf", "300.0")
def sig(x, quoted, n=2):
    """x rounded to n significant figures equals the quoted value."""
    assert float(f"{x:.{n - 1}e}") == float(f"{quoted:.{n - 1}e}"), (x, quoted)
def within(xs, lo, hi): assert all(lo <= x <= hi for x in xs), (min(xs), max(xs), lo, hi)

# --- time budget (Tables IV, V, VII): sequential and projective-with-target runs never exceed T_tot; projective spread as quoted
c = {(n, T2): _cmp(n, T2) for n in SYS for T2 in T2S}
for n in SYS:
    for T2 in T2S:
        for k, v in C[n][f"T2={T2}"].items():
            if k.startswith("sequential"): assert max(v["runs"]["T"]) <= v["T_target"] * (1 + 1e-9), (n, T2, k)
        ref = c[(n, T2)]["pro"]["T"]; assert max(c[(n, T2)]["prot"]["runs"]["T"]) <= ref * (1 + 1e-9)
    for key, v in Q[n].items(): assert max(v["sequential"]["runs"]["T"]) <= v["T_tot"] * (1 + 1e-9)
spread = lambda key: [t / v["pro"]["T"] for v in c.values() for t in v[key]["runs"]["T"]]
sig(min(spread("pro")), 0.58); sig(max(spread("pro")), 1.03, 3); sig(min(spread("prot")), 0.86); assert max(spread("prot")) <= 1.0
qs = [t / v["T_tot"] for n in SYS for v in Q[n].values() for t in v["projective"]["runs"]["T"]]; sig(min(qs), 0.78); sig(max(qs), 1.02, 3)

# --- Table IV / abstract / Sec. IV: infidelities
r_every = {k: v["pro"]["1-F"] / v["seqn"]["1-F"] for k, v in c.items()}; r_tgt = {k: v["prot"]["1-F"] / v["seq3"]["1-F"] for k, v in c.items()}
assert all(x > 1 for x in r_every.values()); sig(min(r_every.values()), 2.3); sig(max(r_every.values()), 2.9e3); assert min(r_every, key=r_every.get) == (R4, "inf") and max(r_every, key=r_every.get) == (P8, "300.0")
assert all(x > 1 for x in r_tgt.values()); sig(min(r_tgt.values()), 1.5); sig(max(r_tgt.values()), 2.2e3)
for (n, T2, key, val) in [(P8, "inf", "pro", 2.8e-6), (P8, "inf", "seqn", 6.8e-8), (P8, "300.0", "pro", 0.076), (P8, "300.0", "seqn", 2.6e-5), (P8, "inf", "prot", 2.7e-6), (P8, "300.0", "prot", 0.013),
                          (P8, "300.0", "seq3s", 0.076), (P8, "300.0", "seq4", 1.0e-4), (TZ, "inf", "seq3", 1.3e-6), (TZ, "300.0", "seq3s", 8.7e-4)]: sig(c[(n, T2)][key]["1-F"], val)
sig(c[(P8, "300.0")]["prot"]["1-F_max"], 0.076); sig(c[(P8, "inf")]["prot"]["1-F_max"], 1.3e-3); sig(c[(P4, "300.0")]["prot"]["1-F_max"], 8.7e-3); sig(c[(TZ, "300.0")]["seq3s"]["1-F_max"], 0.034); assert c[(TZ, "300.0")]["seq3s"]["m"] == 4
assert c[(P8, "inf")]["seq3"]["m"] == 8 and c[(P8, "300.0")]["seq3"]["m"] == 7 and c[(TZ, "inf")]["seq3"]["m"] == 7
runs = c[(P8, "300.0")]["pro"]["runs"]["1-F"]; assert sum(x > 0.07 for x in runs) == 7 and max(x for x in runs if x < 0.07) < 4e-4
sig(c[(P8, "300.0")]["seq3s"]["1-F_max"], 0.61); sig(c[(P8, "300.0")]["seq3s"]["copies"], 160); sig(c[(P8, "300.0")]["seq4"]["1-F_max"], 0.076)
g = lambda tag: next(v for k, v in C[P8]["T2=inf"].items() if tag in k); sig(g("budget x0.1, delta=0.0001")["1-F"], 0.076)
tz = lambda tag: next(v for k, v in C[TZ]["T2=inf"].items() if tag in k)
sig(tz("x1.0, delta=0.001, sig_id=0.0004")["1-F"], 1.2e-7); sig(tz("x1.0, delta=0.001, sig_id=0.0004")["copies"], 5.1e3); sig(tz("x1.0, delta=0.001, sig_id=0.0050")["copies"], 2.1e4)
assert tz("x1.0, delta=0.001, sig_id=0.0004")["m"] == 8 and tz("x0.1, delta=0.001, sig_id=0.0004")["m"] == 6 and 390 <= tz("x0.1, delta=0.001, sig_id=0.0004")["copies"] <= 410
assert PAIRS["T2=inf"]["pair separated"] == 10 and PAIRS["T2=300.0"]["pair separated"] == 7 and PAIRS["T2=inf"]["runs"] == PAIRS["T2=300.0"]["runs"] == 10
main = [r for v in c.values() for key in ("seq3", "seqn") for r in v[key]["records"]]; assert min(r["converged"] for r in main) == 1.0 and max(r["unlabelled"] for r in main) <= 0.003
assert max(r["unlabelled"] for v in c.values() for r in v["seq4"]["records"]) <= 0.007 and max(r["unlabelled"] for v in c.values() for r in v["seq3s"]["records"]) <= 0.065
for n in SYS:                                                         # the time breakdown of every run adds up to no more than its budget
    for T2 in T2S:
        for k, v in C[n][f"T2={T2}"].items():
            if k.startswith("sequential"): assert all(0.95 * v["T_target"] <= r["T_ident"] + r["T_refine"] <= v["T_target"] * (1 + 1e-9) for r in v["records"]), (n, T2, k)
assert all(v["max distance of a reported eigenvalue from the spectrum"] < 0.005 for v in PAIRS.values())    # reported eigenvalues within sig_id of the spectrum (Toeplitz)
allr = [r for n in SYS for T2 in T2S for k, v in C[n][f"T2={T2}"].items() if k.startswith("sequential") for r in v["records"]]; sig(max(r["unlabelled"] for r in allr), 0.23)

# --- Table V / Sec. IV: compliance and samples; cost of a copy relative to a register label
sig(c[(P8, "inf")]["prot"]["shots"], 1.3e5); sig(c[(P8, "inf")]["seq3"]["copies"], 4.8e4); sig(c[(P8, "inf")]["prot"]["eC"], 0.0035); sig(c[(P8, "inf")]["seq3"]["eC"], 0.0086)
best = [min(("pro", "prot", "seqn", "seq3"), key=lambda k: v[k]["eC"]) for v in c.values()]; assert (best.count("seq3"), best.count("pro"), best.count("prot")) == (6, 1, 1)
ties = [sorted(v[k]["eC"] for k in ("pro", "prot", "seqn", "seq3")) for v, b_ in zip(c.values(), best) if b_ == "seq3"]; assert sum(t[1] / t[0] < 1.05 for t in ties) == 2
mult = [v["prot"]["shots"] / 1e4 for v in c.values()]; sig(min(mult), 2, 1); sig(max(mult), 26); assert sum(v["prot"]["eC"] < 0.99 * v["pro"]["eC"] for v in c.values()) == 3
assert all(v["seq3"]["eC"] < v["seqn"]["eC"] for v in c.values())
lab = lambda n, T2: (2**int(next(k for k in C[n][f"T2={T2}"] if k.startswith("projective (")).split("n_c=")[1][0]) - 1) * 2 * np.pi / 1.25
ratio = {(n, T2): v["seq3"]["T"] / v["seq3"]["copies"] / lab(n, T2) for (n, T2), v in c.items()}
within([ratio[(n, "inf")] for n in SYS], 1.9, 3.45); within([ratio[(n, "300.0")] for n in SYS], 4.0, 16.0); sig(c[(P8, "inf")]["seq3"]["T"] / c[(P8, "inf")]["seq3"]["copies"], 4.2e2); sig(lab(P8, "inf"), 156, 3)
sig(max(t for v in c.values() for x in ("seq3", "seq4", "seq3s") for t in v[x]["runs"]["Tcopy"]), 4e4, 1)        # longest-lived copy over all runs, not medians
tn = [t for v in c.values() for t in v["seqn"]["runs"]["Tcopy"]]; sig(min(tn), 9e2, 1); sig(max(tn), 1e4, 1)
sig(max(t for n in SYS for T2 in T2S for k, v in C[n][f"T2={T2}"].items() if k.startswith("sequential") for t in v["runs"]["Tcopy"]), 8e4, 1)
# multinomial model, Eq. (8)
import importlib.util
os.chdir(".."); spec = importlib.util.spec_from_file_location("s2", "stage2_noise_compare.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); os.chdir("paper")
coef = {}
for n, (a, b) in m.problems().items():
    lam, U = np.linalg.eigh(a); w = np.abs(U.conj().T @ b)**2; Cc = np.sum(w / lam); coef[n] = float(np.sqrt(np.sum(w / lam**2) - Cc**2) / Cc)
for n, v in ((R4, 0.21), (P4, 0.69), (P8, 2.0), (TZ, 0.63)): sig(coef[n], v)
for n, meas, pred in ((P8, 0.0086, 0.0061), (R4, 0.0009, 0.0006)):
    v = c[(n, "inf")]["seq3"]; sig(v["eC"], meas, 2 if n == P8 else 1); sig(0.6745 * coef[n] / np.sqrt(v["copies"]), pred, 2 if n == P8 else 1); assert v["eC"] / (0.6745 * coef[n] / np.sqrt(v["copies"])) < 1.5

# --- Table VI / abstract / Sec. V: observables
q = lambda n, T2, shots, meth, k: next(v for kk, v in Q[n].items() if kk.startswith(f"T2={T2}, shots={shots},"))[meth][k]
cnt = lambda shots, k: sum(q(n, T2, shots, "sequential", k) < q(n, T2, shots, "projective", k) for n in SYS for T2 in T2S)
assert (cnt(10000, "eC"), cnt(160000, "eC"), cnt(10000, "eL"), cnt(160000, "eL")) == (3, 7, 7, 8)
rl = [q(n, T2, 160000, "projective", "eL") / q(n, T2, 160000, "sequential", "eL") for n in SYS for T2 in T2S]; sig(min(rl), 1.2); sig(max(rl), 19)
within([q(n, "inf", 160000, "sequential", "eC") for n in SYS], 0.00075, 0.00595); within([q(n, "inf", 160000, "projective", "eC") for n in SYS], 0.00085, 0.00845)
for T2, a_, b_ in (("inf", 0.48, 0.025), ("300.0", 0.50, 0.084)): sig(q(TZ, T2, 160000, "projective", "eL"), a_); sig(q(TZ, T2, 160000, "sequential", "eL"), b_)
p8 = CRB["systems"][P8]; within([p8[f"T={T:g}"]["eigenvalues known"] for T in (30, 100, 300, 1000, 2000)], 0.125, 0.135); sig(p8["optimised design (eigenvalues known)"]["eigenvalues known"], 0.06, 1)
assert all(round(p8[f"T={T:g}"]["eigenvalues unknown"], 2) == 0.13 for T in (300, 1000, 2000))

from make_tables import R7, RQ
sp = lambda n, T2, shots: next(v for kk, v in Q[n].items() if kk.startswith(f"T2={T2}, shots={shots},"))["projective"]
within([sp(n, "inf", 10000)[k] for n in SYS for k in ("eC", "eE")], 0.003, 0.028); within([sp(P8, "300.0", 10000)[k] for k in ("eC", "eE")], 0.08, 0.14)
within([sp(n, "300.0", 10000)[k] for n in (R4, P4, TZ) for k in ("eC", "eE")], 0, 0.02); within([sp(TZ, T2, s_)["eL"] for T2 in T2S for s_ in (10000, 160000)], 0.47, 0.50)
within([sp(n, T2, 160000)["eL"] for n in (R4, P4, P8) for T2 in T2S], 0.0100, 0.0445); sig(sp(P8, "inf", 10000)["eC"], 0.014)
rq, rf = [], []
for n in SYS:
    for T2 in T2S:
        for shots in (10000, 160000):
            p_ = sp(n, T2, shots); q_ = RQ[n][f"T2={'inf' if T2 == 'inf' else 300.0}, shots={shots}"]; k7 = [k for k in R7[n] if k.startswith(f"T2={'inf' if T2 == 'inf' else 300.0}, shots={shots},")][0]
            rq += [q_["rel err C"] / p_["eC"], q_["rel err E"] / p_["eE"], q_["rel err L"] / p_["eL"]]; rf.append(R7[n][k7]["fourier: rel err functional <v|x>"] / p_["eL"])
sig(min(rq), 2, 1); sig(max(rq), 280); sig(min(rf), 2.5); sig(max(rf), 100)
within([p8[f"T={T:g}"]["eigenvalues known"] / sp(P8, "inf", 10000)["eC"] for T in (2000,)] + [p8["optimised design (eigenvalues known)"]["eigenvalues known"] / sp(P8, "inf", 10000)["eC"]], 4.0, 9.5)

# --- Tables II, III / Sec. IV: identification cost and label error
g = lambda T2, mode, p: K[f"T2={T2}, prior={mode}, p_out={p:g}"]
for T2, mode, v in (("inf", "flat", 8.8), ("inf", "learned", 5.0), ("300.0", "flat", 11.8), ("300.0", "learned", 6.4), ("inf", "exact", 4.5)): sig(g(T2, mode, 1e-4)["T_id*sig_id mean"], v, 3 if v >= 10 else 2)
f3 = [(mode, p, g("inf", mode, p)["frac |err|>3 sig_id"]) for mode in ("flat", "learned") for p in (1e-2, 1e-3, 1e-4, 1e-5)]
assert sum(f <= p for _, p, f in f3) == 7 and [f for mode, p, f in f3 if f > p] == [g("inf", "flat", 1e-4)["frac |err|>3 sig_id"]]; sig(g("inf", "flat", 1e-4)["frac |err|>3 sig_id"], 1.7e-4)
assert all(g(T2, mode, p)["frac |err|>4 sig_id"] == 0 for T2 in T2S for mode in ("flat", "learned") for p in (1e-4, 1e-5))
step = lambda T2: [g(T2, mode, p / 10)["T_id*sig_id mean"] - g(T2, mode, p)["T_id*sig_id mean"] for mode in ("flat", "learned") for p in (1e-2, 1e-3, 1e-4)]
within(step("inf"), 1.0, 3.0); within(step("300.0"), 1.25, 4.5)
mis = lambda lab_, p: K[f"miscalibration: {lab_}, p_out={p:g}"]["frac |err|>3 sig_id"] / p
sig(max(mis(l, 1e-3) for l in ("T2 300 -> 100", "p_d 0.02 -> 0.10", "p_d 0.02 -> 0.20")), 8.7); sig(max(mis(l, 1e-4) for l in ("T2 300 -> 100", "p_d 0.02 -> 0.10", "p_d 0.02 -> 0.20")), 17)
within([mis("T2 inf -> 1000", 1e-3), mis("T2 inf -> 1000", 1e-4)], 0.7, 1.3); within([mis(l, p) for l in ("calibrated, T2=300", "calibrated, no dephasing") for p in (1e-3, 1e-4)], 0, 1.7)
assert K["learned prior, T2=inf"]["components"] == 8

# --- Fig. 2 / Sec. II and IV: detection
for place in D:
    for r in D[place]:
        assert r["false_clusters_mean"] == 0
        if abs(r["S_w"] - 3) < 1e-9: assert 0.25 <= r["P_detect"] <= 0.42
        if abs(r["S_w"] - 10) < 1e-9: assert 0.91 <= r["P_detect"] <= 1
        if r["S_w"] > 10.5: assert r["P_detect"] == 1.0
        x = r["T_tot"] / r["S"] / (31 * 2 * np.pi / 1.25); assert (6 <= x <= 8) if r["S"] == 1000 else (3.25 <= x <= 4.25), x
for place, pk, win in (("far", 0.57, 0.88), ("adjacent", 0.79, 0.93)): sig(ACC[place]["peak bin"], pk); sig(ACC[place]["+-1 window"], win)
sig(ACC["floors"]["far"], 0.043); sig(ACC["floors"]["adjacent"], 0.038)

# --- lifetime test
med = [L[f"T_sys={t}"]["1-F"] for t in ("inf", "1000000.0", "300000.0", "100000.0", "30000.0")]; within(med, 2.4e-6, 5.05e-6)
sig(max(L[f"T_sys={t}"]["1-F_max"] for t in ("inf", "1000000.0", "300000.0", "100000.0", "30000.0")), 1.7e-5); sig(L["T_sys=10000.0"]["1-F"], 3.1e-5); sig(L["T_sys=10000.0"]["1-F_max"], 0.021)
assert all(v["m"] == 7 and v["false_mean"] == 0 for k, v in L.items() if k.startswith("T_sys"))

# --- Tables VIII, IX / abstract / Sec. VI: clock
ck = lambda n, T2, k: B[n][f"T2={T2}"]["clock: " + k]; w = lambda n, T2, k: B[n][f"T2={T2}"]["window: " + k]
d = {(n, T2): ck(n, T2, "learned angles") - ck(n, T2, "no learning") for n in (P4, P8, TZ) for T2 in T2S}; assert abs(d.pop((P8, "300.0"))) < 0.011; within(d.values(), 0.03, 0.30)
within([ck(n, T2, "idealised") - ck(n, T2, "learned angles") for n in SYS for T2 in T2S], 0.0495, 0.195); sig(ck(R4, "inf", "+t0"), 0.993, 3); sig(ck(TZ, "inf", "+t0"), 0.989, 3)
within([ck(P8, "300.0", k) for k in ("learned angles", "+t0", "no learning", "exact-spectrum t0")], 0.69, 0.74)
assert max(v for n in SYS for k, v in B[n]["T2=300.0"].items() if k.endswith("|mc")) <= 0.0105 and all(B[n]["T2=300.0"]["n_mc"] == 256 for n in SYS)
best_nc = {R4: 5, P4: 5, P8: 6, TZ: 6}
for n, v in ((R4, 0.99988), (P4, 0.99994), (P8, 0.99990), (TZ, 0.99954)): sig(w(n, "inf", f"DPSS NW=2.0 +-2 n_c={best_nc[n]}"), v, 5); assert w(n, "inf", f"DPSS NW=2.0 +-2 n_c={best_nc[n]}") >= 0.9995
within([w(n, "inf", f"uniform +-1 n_c={best_nc[n]}") for n in SYS], 0.865, 0.955); within([w(n, "inf", f"uniform +-2 n_c={best_nc[n]}") for n in SYS], 0.905, 0.975)
within([w(n, "inf", f"DPSS NW=2.0 +-1 n_c={best_nc[n]}") for n in SYS], 0.990, 0.99935); assert max(w(n, "inf", f"uniform+t0opt +-1 n_c={nc}") for n in SYS for nc in (B[n]["T2=inf"]["n_c"], B[n]["T2=inf"]["n_c"] + 1)) <= 0.9935
within([w(n, "inf", f"DPSS NW=2.0 +-2 n_c={best_nc[n]} |p") / w(n, "inf", f"uniform +-1 n_c={best_nc[n]} |p") for n in SYS], 0.9, 1.115)
within([w(P4, "inf", f"DPSS NW={x} +-2 n_c=4") for x in ("1.5", "2.0")], 0.9975, 0.9985); within([w(P8, "inf", f"{x} +-2 n_c=5") for x in ("sine", "DPSS NW=1.5", "DPSS NW=2.0")], 0.962, 0.9875)
gain = {n: max(w(n, "300.0", f"{x} +-2 n_c={B[n]['T2=300.0']['n_c']}") for x in ("sine", "DPSS NW=1.5", "DPSS NW=2.0")) - w(n, "300.0", f"uniform +-1 n_c={B[n]['T2=300.0']['n_c']}") for n in SYS}
within(gain.values(), 0.03, 0.06)
gain1 = {n: max(w(n, "300.0", f"{x} +-2 n_c={B[n]['T2=300.0']['n_c'] + 1}") for x in ("sine", "DPSS NW=1.5", "DPSS NW=2.0")) - w(n, "300.0", f"uniform +-1 n_c={B[n]['T2=300.0']['n_c'] + 1}") for n in SYS}
within(gain1.values(), 0.07, 0.105); assert max(x for n in SYS for k, v in B[n]["T2=300.0"]["runs"].items() if k.endswith("|mc") for x in v) <= 0.0135; nc8 = B[P8]["T2=300.0"]["n_c"] + 1; assert w(P8, "300.0", f"uniform+t0opt +-1 n_c={nc8}") > max(w(P8, "300.0", f"{x} +-2 n_c={nc8}") for x in ("sine", "DPSS NW=1.5", "DPSS NW=2.0"))
mcs = [v for n in SYS for k, v in B[n]["T2=300.0"].items() if k.endswith("|mc")]; within(mcs, 0.00085, 0.0105); assert all(any(k.endswith("|mc") for k in B[n]["T2=300.0"]["runs"]) for n in SYS)
assert max(v for n in SYS for k, v in B[n]["T2=300.0"].items() if k.startswith("window") and "|" not in k) < 0.95
nc4 = B[P4]["T2=300.0"]["n_c"]; assert w(P4, "300.0", f"uniform+t0opt +-1 n_c={nc4}") > max(w(P4, "300.0", f"{x} +-2 n_c={nc4}") for x in ("sine", "DPSS NW=1.5", "DPSS NW=2.0"))
within([1 - w(n, "inf", f"DPSS NW=2.0 +-2 n_c={best_nc[n]}") for n in SYS], 5.5e-5, 5e-4)

# --- Table I (100 repetitions; per-run fidelities and learned eigenvalues are stored)
J4 = json.load(open("../results/stage4_results.json")); rows4 = [(n, r) for n in J4 if isinstance(J4[n], list) for r in J4[n]]; cnt = lambda runs, lo, hi=2.0: sum(lo <= x < hi for x in runs)
assert all(len(r[k]["runs"]) == 100 for n, r in rows4 for k in ("MM-QCELS", "QMEGS", "QPE6", "BC2 ideal"))
row300 = [r for r in J4[P8] if r["T2"] == 300.0][0]; r4i = [r for r in J4[P8] if r["T2"] == "inf"][0]; r4 = row300["BC2 ideal"]
assert f"{r4['median']:.3f}" == "0.924" and f"{r4['q75']:.4f}" == "0.9997" and cnt(r4["runs"], 0, 0.93) == 57 and cnt(r4["runs"], 0.92, 0.93) == 55
mix = sorted(r4i[k]["median"] for k in ("MM-QCELS", "QMEGS")); assert f"{mix[0]:.2f}" == "0.76" and f"{mix[1]:.2f}" == "0.88"
qm = r4i["QMEGS"]["runs"]; assert cnt(r4i["MM-QCELS"]["runs"], 0.99) == 1 and cnt(qm, 0.99) == 13 and cnt(qm, 0.38, 0.40) == 26 and cnt(qm, 0.92, 0.93) == 21
assert f"{np.median(qm[:50]):.2f}" == "0.91" and f"{np.median(qm[50:]):.2f}" == "0.54"
assert min(r["QPE6"]["median"] for n, r in rows4 if r["T2"] == "inf") >= 0.998 and min(r["BC2 ideal"]["median"] for n, r in rows4 if r["T2"] == "inf") >= 0.9999
assert cnt(r4i["QPE6"]["runs"], 0.92, 0.93) == 32 and cnt(r4i["QPE6"]["runs"], 0, 0.92) == 0 and min(r4i["BC2 ideal"]["runs"]) > 0.97
assert sorted(round(r["BC2 ideal"]["median"], 4) for n, r in rows4 if r["T2"] == 300.0)[1] >= 0.9996          # three systems keep a median F >= 0.9996 at T2 = 300
p4_300 = [r for r in J4[P4] if r["T2"] == 300.0][0]["BC2 ideal"]["runs"]; assert cnt(p4_300, 0, 0.99) == 6 and f"{min(p4_300):.3f}" == "0.969"
assert cnt(r4i["BC2 ideal"]["runs"], 0, 0.99) == 3 and f"{min(r4i['BC2 ideal']['runs']):.3f}" == "0.972"
cap = J4["time-cap sweep (P3 N=8, T2=300)"]; assert len(cap) == 5 and all(f"{c['median']:.3f}" in ("0.924", "0.925") and f"{c['q25']:.3f}" == "0.924" and f"{c['q75']:.3f}" == "0.999" and cnt(c["runs"], 0, 0.93) in (60, 61) for c in cap.values())
grp = [np.array(c["runs"]) < 0.93 for c in cap.values()]; assert max(int((g != grp[0]).sum()) for g in grp) == 1   # same seeds for every cap: one run changes group
# the component behind the 0.076 plateau: lambda_2 = 0.121 (w = 0.006, q = 0.011) rotated with the learned value of lambda_1 = 0.031
_pr = m.problems(); _a = _pr[P8][0]; _l, _U = np.linalg.eigh(_a); _w = np.abs(_U.conj().T @ _pr[P8][1])**2; _q = (_w / _l**2) / np.sum(_w / _l**2); _r = _l[1] / _l[0]
assert f"{_l[1]:.3f}" == "0.121" and f"{_w[1]:.3f}" == "0.006" and f"{_q[1]:.3f}" == "0.011" and f"{_l[0]:.3f}" == "0.031" and f"{_q[1] * (_r - 1)**2:.2f}" == "0.09" and f"{_q[0]:.2f}" == "0.89"
_x = (_U.conj().T @ _pr[P8][1]) / _l; _xh = (_U.conj().T @ _pr[P8][1]) / np.where(np.arange(len(_l)) == 1, _l[0], _l); _ex = 1 - abs(np.vdot(_x, _xh))**2 / (np.vdot(_x, _x).real * np.vdot(_xh, _xh).real)
assert f"{_ex:.3f}" == "0.076" and abs(_ex - _q[1] * (1 - _q[1]) * (_r - 1)**2 / (1 + _q[1] * (_r**2 - 1))) < 1e-12      # exact single-component expression
merged = lambda mu: len(mu) > 0 and int(np.argmin(np.abs(np.array(mu) - _l[0]))) == int(np.argmin(np.abs(np.array(mu) - _l[1])))   # lambda_2 has no learned value of its own
near1 = lambda mu: float(np.array(mu)[np.argmin(np.abs(np.array(mu) - _l[0]))])
mu300 = row300["learned eigenvalues"]["BC2 ideal"]; low = [k for k, f in enumerate(r4["runs"]) if f < 0.93]; rest = [k for k in low if r4["runs"][k] < 0.92]
assert sum(merged(mu300[k]) and abs(near1(mu300[k]) - _l[0]) < 0.01 for k in low if k not in rest) == 55 and not any(merged(mu300[k]) for k in range(100) if k not in low)
assert len(rest) == 2 and sorted(f"{r4['runs'][k]:.2f}" for k in rest) == ["0.84", "0.86"] and all(f"{near1(mu300[k]):.3f}" == "0.077" for k in rest)
q6i, q6d = r4i["QPE6"]["runs"], row300["QPE6"]["runs"]; mi, md = r4i["learned eigenvalues"]["QPE6"], row300["learned eigenvalues"]["QPE6"]
assert sum(merged(mi[k]) for k in range(100) if q6i[k] < 0.93) == 32 and not any(merged(mi[k]) for k in range(100) if q6i[k] >= 0.93) and all(merged(md[k]) for k in range(100)) and max(q6d) < 0.93
assert f"{row300['QPE6']['median']:.3f}" == "0.925"
# ... and the ten projective runs of Table IV (recomputed, a few seconds): every low run shows the same merger
os.chdir(".."); _sp = importlib.util.spec_from_file_location("s9", "stage9_sequential_qpe.py"); _s9 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_s9); _s9.set_noise(300.0); _low = []
with np.errstate(all="ignore"):
    for _k in range(10):
        _mu = _s9.projective(_l, _w, 300.0, 10_000, np.random.default_rng(900 + _k))[0]
        if 1 - _s9.S2.fidelity(_a, _pr[P8][1], _mu) > 0.05: _low.append(merged(_mu))
    # Fig. 1: eight-bit QPE on Poisson N=8 without dephasing (seeds of stage 2) does not select lambda_2 in two of three runs
    _q8 = [merged(m.qpe_learn(_l, _w, 8, np.inf, 10_000, np.random.default_rng(200 + _k))[0]) for _k in range(3)]
os.chdir("paper"); assert len(_low) == 7 and all(_low) and sum(_q8) == 2

# --- Fig. 1 (stage 2, medians of three runs): the statements of the caption and of the text
F2 = json.load(open("../results/stage2_results.json")); inf2 = lambda n, key: {(1e9 if r["T2"] == "inf" else r["T2"]): 1 - r[key]["F_median"] for r in F2[n]["rows"]}
assert all(v < 1e-3 for n in SYS for t, v in inf2(n, "BC2").items() if t >= 300)
assert f"{inf2(R4, 'BC2')[100.0]:.1e}" == "1.7e-03" and f"{inf2(P8, 'BC2')[100.0]:.3f}" == "0.075" and f"{1 - inf2(R4, 'BC2')[100.0]:.3f}" == "0.998" and f"{1 - inf2(P8, 'BC2')[100.0]:.2f}" == "0.92"
for n, t8, t6 in ((R4, (300.0, 100.0), (100.0,)), (P4, (300.0, 100.0), (100.0,)), (TZ, (100.0,), ())):      # where sampled QPE fails (1-F >= 0.02); elsewhere below 1e-3
    q8, q6 = inf2(n, "QPE n_p=8"), inf2(n, "QPE n_p=6")
    assert all((q8[t] >= 0.02) if t in t8 else (q8[t] < 1e-3) for t in q8) and all((q6[t] >= 0.02) if t in t6 else (q6[t] < 1e-3) for t in q6)
q8, q6 = inf2(P8, "QPE n_p=8"), inf2(P8, "QPE n_p=6"); assert all(f"{q8[t]:.3f}" in ("0.076", "0.077") for t in q8 if t >= 1000) and all(0.0745 <= q6[t] < 0.0865 for t in q6 if t <= 1000)

# --- time rules of the Table I estimators
tau0 = 2 * np.pi / 1.25 / 12; assert f"{11 * tau0 * 2**9:.0f}" == "2359" and f"{11 * tau0 * 2**6:.0f}" == "295" and f"{64 * 2 * np.pi / 1.25:.0f}" == "322"
from math import erfc; assert f"{100 * erfc(1 / np.sqrt(2)):.0f}" == "32"

# --- Table X (forty learning runs of stage 6) and the stage-5 statement
J6 = json.load(open("../results/stage6_results.json")); st6 = lambda n, meth, t: J6[n]["seed study"][meth][f"deg F>={t}"]; reach = lambda d: sum(x is not None for x in d)
assert all(J6[n]["seed study"]["seeds"] == 40 for n in J6)
d = st6(P4, "spectral correction", 0.9999); assert reach(d) == 39 and d.count(7) == 39 and J6[P4]["interval odd LP (standard)"] == J6[P4]["interval odd LP (standard)"] | {"deg F>=0.999": 55, "deg F>=0.9999": None}
d = st6(P8, "spectral correction", 0.999); assert reach(d) == 37 and d.count(35) == 26 and J6[P8]["interval odd LP (standard)"]["deg F>=0.999"] is None and J6[P8]["interval odd LP (standard)"]["deg F>=0.9999"] is None
d = st6(P8, "spectral correction", 0.9999); assert reach(d) == 13 and d.count(39) == 10
t4 = [x for x in J6[P4]["seed study"]["spectral correction"]["tau F>=0.9999"] if x is not None]; assert f"{np.median(t4):.1f}" == "4.1" and f"{np.median(t4)**2:.0f}" == "17"
assert set(x for x in st6(P4, "node LP", 0.999) + st6(P4, "node LP", 0.9999) if x is not None) == {15, 17} and reach(st6(P8, "node LP", 0.999)) == 16 and reach(st6(P8, "node LP", 0.9999)) == 1
assert reach(st6(TZ, "node LP", 0.999)) == 8 and st6(TZ, "spectral correction", 0.9999).count(21) == 40 and J6[TZ]["interval odd LP (standard)"]["deg F>=0.9999"] == 39
tz = [x for x in J6[TZ]["seed study"]["spectral correction"]["tau F>=0.9999"]]; assert f"{np.median(tz):.2f}" == "1.30"
pc = [v for k, v in J6[TZ].items() if k.startswith("per-component")][0]; assert f"{1 - pc['1-F']:.4f}" == "0.9895"
p10 = [v for k, v in J6[P8].items() if k.startswith("Prop10")][0]; assert f"{1 - p10['sigma=0.001']['1-F median']:.2f}" == "0.93" and f"{1 - p10['sigma=0.01']['1-F median']:.2f}" == "0.54"
J5 = json.load(open("../results/stage5_results.json"))[P8]; b5 = [v for k, v in J5.items() if k.startswith("method B, T2=inf")][0]; assert b5["lagrange"]["median"] > 0.99999 and f"{b5['lagrange overhead']['median']:.0e}" == "3e+03"

# --- application test (Sec. VII, stage 10): DC power flow of standard test systems
os.chdir(".."); _sp = importlib.util.spec_from_file_location("s10", "stage10_power_grid.py"); _g = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_g); _sys = {}; _drop = {}; _sq = {}; _lam = {}
for _c in ("case9", "case14", "case30", "case118"):
    _A, _B, _V, _sc = _g.problem(_c); _lm = np.linalg.eigvalsh(_A); _Uu = np.linalg.eigh(_A)[1]; _ww = np.abs(_Uu.conj().T @ _B)**2; _qq = (_ww / _lm**2) / np.sum(_ww / _lm**2)
    _sys[_c] = (_sc["n"], 1 / _lm.min(), float(np.diff(_lm).min()), int(np.searchsorted(np.cumsum(np.sort(_ww)[::-1]), 0.99)) + 1, float(_ww[0]), float(_qq[0]))
    _sq[_c] = _g.series_q(_c); _lam[_c] = _lm
    _miss = lambda idx: 1 - _g.S9.S2.fidelity(_A, _B, np.delete(_lm, idx)); _drop[_c] = (_lm[:2], _ww[:2], _miss([0]), _miss([1]), _miss([0, 1, 2, 3]), _miss([0, 1, 2, 3, 4]))   # a missed component is rotated with the nearest learned value
os.chdir("paper"); G10 = lambda kind, c: json.load(open(f"../results/stage10_{kind}.{c}.json"))
# weak components that decide the fidelity (learning-stage paragraph of Sec. VII): eigenvalue, weight, and the infidelity when the component is missed
assert f"{_drop['case9'][0][0]:.3f}" == "0.017" and f"{100 * _drop['case9'][1][0]:.2f}" == "0.18" and f"{_drop['case9'][2]:.2f}" == "0.19" and f"{_drop['case9'][1][0] * 100:.2f}" == "0.18"   # w sqrt(S) = 0.18 at S = 1e4
assert f"{_drop['case14'][0][1]:.3f}" == "0.042" and f"{100 * _drop['case14'][1][1]:.2f}" == "0.33" and f"{_drop['case14'][3]:.3f}" == "0.012"
assert f"{_drop['case30'][0][1]:.4f}" == "0.0069" and f"{100 * _drop['case30'][1][1]:.2f}" == "0.07" and f"{_drop['case30'][3]:.3f}" == "0.065" and f"{_drop['case30'][4]:.2f}" == "0.66" and f"{_drop['case30'][5]:.2f}" == "0.69"
assert [_sys[c][0] for c in ("case9", "case14", "case30", "case118")] == [8, 13, 29, 117] and [f"{_sys[c][1]:.0f}" for c in ("case9", "case14", "case30")] == ["59", "119", "493"] and f"{_sys['case118'][1]:.1e}" == "2.9e+03"
assert [_sys[c][3] for c in ("case9", "case14", "case30", "case118")] == [5, 9, 20, 72] and [f"{_sys[c][2]:.2g}" for c in ("case9", "case14", "case30")] == ["0.047", "0.0072", "0.0037"] and f"{2 * np.pi / (64 * 2 * np.pi / 1.25):.3f}" == "0.020"
assert f"{100 * _sys['case9'][4]:.1f}" == "0.2" and f"{100 * _sys['case9'][5]:.0f}" == "27" and f"{100 * _sys['case30'][4]:.1f}" == "0.4" and f"{100 * _sys['case30'][5]:.0f}" == "81"
q9, q14, q30 = (G10("qoi", c) for c in ("case9", "case14", "case30")); mw = lambda q, k, m: q[k][m]["eL"] * q["problem"]["flow_MW"]; mv = lambda q, k, m: q[k][m]["eC"] * q["problem"]["C_MVAr"]
assert f"{q9['problem']['flow_MW']:.0f}" == "163" and f"{q14['problem']['flow_MW']:.0f}" == "148" and f"{q30['problem']['flow_MW']:.0f}" == "37" and f"{q9['problem']['C_MVAr']:.1f}" == "48.7"
kb = "T2=inf, shots=160000, n_c=6"; assert f"{mw(q9, kb, 'sequential'):.1f}" == "1.8" and f"{mw(q9, kb, 'projective'):.0f}" == "109" and f"{mv(q9, kb, 'sequential'):.2f}" == "0.06" and f"{mv(q9, kb, 'projective'):.1f}" == "1.2"
s14 = [(mw(q14, k, "projective"), mw(q14, k, "sequential")) for k in q14 if k != "problem"]; s30 = [(mw(q30, k, "projective"), mw(q30, k, "sequential")) for k in q30 if k != "problem"]
assert len(s14) == 5 and 9.5 <= min(min(x) for x in s14) and max(max(x) for x in s14) < 80.5 and sum(b < a for a, b in s14) == 4 and len(s30) == 5 and 1.95 <= min(min(x) for x in s30) and max(max(x) for x in s30) < 7.5 and sum(a < b for a, b in s30) == 4
k14 = "T2=300.0, shots=10000, n_c=5"; assert f"{mw(q14, k14, 'sequential'):.0f}" == "80" and f"{mw(q14, k14, 'projective'):.0f}" == "35"
assert sum(q[k]["sequential"]["eC"] < q[k]["projective"]["eC"] for q in (q9, q14, q30) for k in q if k != "problem") == 14 == sum(len(q) - 1 for q in (q9, q14, q30))
ob = [r for c in ("case9", "case14", "case30") for r in G10("observables", c).values()]; assert len(ob) == 12 and all(r["fourier"]["eL"] > 1 for r in ob)
assert f"{100 * min(r['qks']['eL'] for r in ob):.0f}" == "7" and f"{100 * max(r['qks']['eL'] for r in ob):.0f}" == "104" and f"{100 * min(r['qks']['eC'] for r in ob):.0f}" == "10" and f"{100 * max(r['qks']['eC'] for r in ob):.0f}" == "85"
be = {c: G10("backend", c) for c in ("case9", "case14", "case30")}; bf = lambda c, k, name: be[c][k][name]["F"]
assert all(f"{bf(c, 'T2=inf', 'sequential + 7-bit DPSS NW=2 clock'):.4f}" == "0.9999" for c in ("case9", "case14")) and f"{bf('case9', 'T2=inf', 'standard HHL, 10-bit clock'):.3f}" == "0.985" and f"{bf('case14', 'T2=inf', 'standard HHL, 10-bit clock'):.3f}" == "0.965"
assert max(be["case30"]["T2=inf"], key=lambda m: be["case30"]["T2=inf"][m]["F"]) == "standard HHL, 10-bit clock" and f"{bf('case30', 'T2=inf', 'standard HHL, 10-bit clock'):.2f}" == "0.77"
assert max(r["F"] for c in be for r in be[c]["T2=300.0"].values()) < 0.73
sc30 = {m_: json.load(open(f"../results/stage10_scale.case30.x{m_}.json")) for m_ in (1, 4, 16)}; sc118 = {m_: json.load(open(f"../results/stage10_scale.case118.x{m_}.json")) for m_ in (1, 4, 16)}
fine = lambda d: [v for k_, v in d.items() if k_.startswith("sequential")][-1]
assert [f"{fine(sc30[m_])['1-F']:.2g}" for m_ in (1, 4, 16)] == ["0.065", "2e-05", "9.6e-07"] and [fine(sc30[m_])["m"] for m_ in (1, 4, 16)] == [22, 27, 29]
assert all(0.665 <= sc30[m_]["projective, 6-bit clock"]["1-F"] < 0.685 for m_ in (1, 4, 16)) and all(0.0645 <= sc30[m_]["projective, 10-bit clock"]["1-F"] < 0.0675 for m_ in (1, 4))
assert [f"{sc30[m_]['projective, 10-bit clock']['T'] / sc30[m_]['T_unit']:.0f}" for m_ in (1, 4)] == ["3", "13"] and f"{sc118[4]['projective, 10-bit clock']['T'] / sc118[4]['T_unit']:.0f}" == "13"
assert fine(sc118[16])["m"] == 60 and f"{fine(sc118[16])['1-F']:.3f}" == "0.008" and f"{sc118[4]['projective, 10-bit clock']['1-F']:.2f}" == "0.09" and all(f"{sc118[m_]['projective, 6-bit clock']['1-F']:.2f}" in ("0.77", "0.78") for m_ in (1, 4, 16))
# learning stages on the grid systems (Table XI, learning-stage paragraph of Sec. VII, abstract)
MT = {c_: G10("methods", c_) for c_ in ("case9", "case14", "case30")}; CP = {c_: G10("compare", c_)["6-bit clock"] for c_ in ("case9", "case14", "case30")}; KS = ("T2=inf", "T2=300.0")
_pk = lambda c_, k_, pref: [v for n_, v in CP[c_][k_].items() if n_.startswith(pref)]; pro10 = lambda c_, k_: _pk(c_, k_, "projective (")[0]; seq10 = lambda c_, k_: _pk(c_, k_, "sequential (budget x1.0, delta=0.001")
base = lambda c_, k_: {m_: r["1-F"] for m_, r in MT[c_][k_].items() if "runs" in r}; mixt = ("Bayesian clock (mixture)", "QMEGS-type (mixture)", "MM-QCELS (mixture)")
assert all(MT[c_]["reps"] == 100 and all(len(r["runs"]["1-F"]) == 100 for k_ in KS for r in MT[c_][k_].values() if "runs" in r) for c_ in MT)
assert MT["case9"]["failures"] == {} == MT["case14"]["failures"] and {m_: len(v) for m_, v in MT["case30"]["failures"].items()} == {"T2=inf, MM-QCELS (mixture)": 2, "T2=300.0, QMEGS-type (mixture)": 2}
tt = [r["T"] for c_ in MT for k_ in KS for r in MT[c_][k_].values() if "runs" in r]; assert f"{min(tt):.0e}" == "4e+05" and f"{max(tt):.0e}" == "5e+07"
assert all(f"{pro10(c_, 'T2=inf')['T']:.0e}" == "2e+07" and f"{pro10(c_, 'T2=300.0')['T']:.0e}" == "4e+06" and len(pro10(c_, k_)["runs"]["1-F"]) == 10 for c_ in CP for k_ in KS)
assert all(max(v["runs"]["T"]) <= pro10(c_, k_)["T"] * (1 + 1e-9) for c_ in CP for k_ in KS for v in seq10(c_, k_))
l9 = [x for k_ in KS for x in list(base("case9", k_).values()) + [pro10("case9", k_)["1-F"]]]; assert len(l9) == 12 and f"{min(l9):.2f}" == "0.18" and f"{max(l9):.2f}" == "0.23"
sig(seq10("case9", "T2=inf")[0]["1-F"], 4.3e-7); r9 = seq10("case9", "T2=300.0")[0]; assert f"{r9['1-F']:.2f}" == "0.11" and sum(x > 0.18 for x in r9["runs"]["1-F"]) == 5 and sum(x < 0.05 for x in r9["runs"]["1-F"]) == 5
b14 = base("case14", "T2=inf"); assert f"{pro10('case14', 'T2=inf')['1-F']:.3f}" == "0.012" == f"{b14['QMEGS-type (mixture)']:.3f}" and f"{b14['QPE histogram, 6 bits']:.3f}" == "0.015"
mm = MT["case14"]["T2=inf"]["MM-QCELS (mixture)"]; assert sum(x < 1e-3 for x in mm["runs"]["1-F"]) == 54; sig(mm["1-F"], 2.6e-4); assert round(pro10("case14", "T2=inf")["T"] / mm["T"]) == 9
sig(b14["QPE histogram, 8 bits"], 3.7e-4); sig(b14["QPE histogram, 10 bits"], 1.1e-5); sig(seq10("case14", "T2=inf")[0]["1-F"], 5.6e-5); sig(seq10("case14", "T2=inf")[-1]["1-F"], 3.3e-7)
assert all(f"{MT[c_]['T2=inf']['QPE histogram, 10 bits']['T'] / pro10(c_, 'T2=inf')['T']:.1f}" == "2.4" for c_ in ("case14", "case30"))
assert min(list(base("case14", "T2=300.0").values()) + [pro10("case14", "T2=300.0")["1-F"]] + [v["1-F"] for v in seq10("case14", "T2=300.0")]) > 0.0115
b30 = base("case30", "T2=inf"); l30 = [b30[m_] for m_ in mixt + ("QPE histogram, 6 bits",)] + [pro10("case30", "T2=inf")["1-F"]]; assert f"{min(l30):.2f}" == "0.67" and f"{max(l30):.2f}" == "0.77"
assert f"{seq10('case30', 'T2=inf')[-1]['1-F']:.3f}" == "0.066" and f"{b30['QPE histogram, 10 bits']:.3f}" == "0.064"
assert min(list(base("case30", "T2=300.0").values()) + [pro10("case30", "T2=300.0")["1-F"]] + [v["1-F"] for v in seq10("case30", "T2=300.0")]) > 0.6
# abstract: without dephasing the sequential estimator alone finds the weak 9-bus component
assert min(list(base("case9", "T2=inf").values()) + [pro10("case9", "T2=inf")["1-F"]]) > 0.18 and max(seq10("case9", "T2=inf")[0]["runs"]["1-F"]) < 1e-6
# stored learned eigenvalues (Sec. VII): a run assigns a value to component j when the learned value nearest to it is nearer to it than to any other eigenvalue
def asg(c_, j, mu): mu = np.asarray(mu, float); return len(mu) > 0 and int(np.argmin(abs(_lam[c_] - mu[np.argmin(abs(mu - _lam[c_][j]))]))) == j
runs = lambda c_, k_, m_: MT[c_][k_][m_]["runs"]; n_asg = lambda c_, k_, m_, j: sum(asg(c_, j, mu) for mu in runs(c_, k_, m_)["mu"])
seq_mu = lambda row: [r_["mu"] for r_ in row["records"]]; unasg = lambda c_, mu: [j for j in range(len(_lam[c_])) if not asg(c_, j, mu)]
# protocol of Table XI: time of the upper rows relative to the lower rows, repetitions, precision target, runs without a finite estimate
tr = [r["T"] / pro10(c_, k_)["T"] for c_ in MT for k_ in KS for r in MT[c_][k_].values() if "runs" in r]; assert f"{min(tr):.2f}" == "0.10" and f"{max(tr):.1f}" == "2.4"
assert all(len(seq10(c_, k_)[0]["runs"]["1-F"]) == 10 for c_ in CP for k_ in KS) and all(len(seq10(c_, k_)[-1]["runs"]["1-F"]) == 5 for c_ in ("case14", "case30") for k_ in KS) and all(len(seq10("case9", k_)) == 1 for k_ in KS)
assert all("delta=0.001" in n_ for c_ in CP for k_ in KS for n_ in CP[c_][k_] if n_.startswith("sequential (budget x1.0, delta=0.001"))
allr = [(c_, k_, m_, r) for c_ in MT for k_ in KS for m_, r in MT[c_][k_].items() if "runs" in r]; assert sum(len(r["runs"]["1-F"]) for *_, r in allr) == 3000
nanr = {f"{k_}, {m_}": [i for i, x in enumerate(r["runs"]["1-F"]) if not np.isfinite(x)] for c_, k_, m_, r in allr if c_ == "case30" and not np.all(np.isfinite(r["runs"]["1-F"]))}
assert nanr == {"T2=inf, MM-QCELS (mixture)": [51, 76], "T2=300.0, QMEGS-type (mixture)": [1, 71]} == {m_: [x[0] for x in v] for m_, v in MT["case30"]["failures"].items()}
assert all(np.all(np.isfinite(r["runs"]["1-F"])) for c_, k_, m_, r in allr if c_ != "case30") and all(len(mu) == 0 for *_, r in allr for f_, mu in zip(r["runs"]["1-F"], r["runs"]["mu"]) if not np.isfinite(f_))
assert all(abs(round(float(np.median(np.nan_to_num(r["runs"]["1-F"], nan=1.0))), 3) - round(r["1-F"], 3)) < 0.0011 for *_, r in allr)
# 9-bus: who assigns a value to the smallest eigenvalue
assert sum(n_asg("case9", "T2=inf", f"QPE histogram, {b_} bits", 0) for b_ in (6, 8, 10)) == 0 and sum(asg("case9", 0, mu) for mu in pro10("case9", "T2=inf")["runs"]["mu"]) == 0
assert [n_asg("case9", "T2=inf", m_, 0) for m_ in mixt] == [12, 25, 31]
low9 = lambda m_: [(x < 0.1, asg("case9", 0, mu)) for x, mu in zip(runs("case9", "T2=inf", m_)["1-F"], runs("case9", "T2=inf", m_)["mu"])]
assert [sum(a for a, _ in low9(m_)) for m_ in mixt] == [3, 12, 8] and all(b for m_ in mixt for a, b in low9(m_) if a)
assert sum(asg("case9", 0, mu) for mu in seq_mu(seq10("case9", "T2=inf")[0])) == 10 and sum(asg("case9", 0, mu) for mu in seq_mu(seq10("case9", "T2=300.0")[0])) == 5
g9 = sorted(r9["runs"]["1-F"]); assert f"{g9[0]:.0e}" == "4e-04" and f"{g9[4]:.2f}" == "0.04" and all(f"{x:.2f}" == "0.19" for x in g9[5:])
# 14-bus: the component at 0.042
assert n_asg("case14", "T2=inf", "MM-QCELS (mixture)", 1) == 26 and [n_asg("case14", "T2=inf", f"QPE histogram, {b_} bits", 1) for b_ in (8, 10)] == [98, 98]
assert all(asg("case14", 1, mu) for row in seq10("case14", "T2=inf") for mu in seq_mu(row))
c9 = G10("compare", "case14")["9-bit clock"]; p9 = c9["projective"]["runs"]; assert sum(asg("case14", 1, mu) for mu in p9["mu"]) == 6 and sum(asg("case14", 1, mu) for mu in c9["sequential"]["runs"]["mu"]) == 10
sig(c9["projective"]["1-F"], 1.8e-6); sig(c9["sequential"]["1-F"], 6.3e-8); hi9 = sorted(p9["1-F"])[6:]; assert len(hi9) == 4 and 0.0095 <= min(hi9) and max(hi9) < 0.0125 and max(sorted(p9["1-F"])[:6]) < 1e-5
assert f"{c9['T_ref'] / pro10('case14', 'T2=inf')['T']:.1f}" == "2.0" and max(c9["sequential"]["runs"]["T"]) <= c9["T_ref"] * (1 + 1e-9)
assert sum(asg("case14", 1, mu) for mu in seq_mu(seq10("case14", "T2=300.0")[0])) == 4 and sum(asg("case14", 1, mu) for mu in pro10("case14", "T2=300.0")["runs"]["mu"]) == 0
assert [n_asg("case14", "T2=300.0", m_, 1) for m_ in ("QMEGS-type (mixture)", "MM-QCELS (mixture)")] == [30, 23]
# 30-bus: unassigned components
pm = pro10("case30", "T2=inf")["runs"]["mu"]; assert all(3 <= min(set(range(29)) - set(unasg("case30", mu))) <= 5 and 18 <= len(unasg("case30", mu)) <= 20 for mu in pm) and {min(set(range(29)) - set(unasg("case30", mu))) for mu in pm} == {3, 4, 5}
sf = seq10("case30", "T2=inf")[-1]; assert len(seq_mu(sf)) == 5 and not any(asg("case30", 1, mu) for mu in seq_mu(sf)) and 100 - n_asg("case30", "T2=inf", "QPE histogram, 10 bits", 1) == 77
g30 = sorted(seq10("case30", "T2=inf")[0]["runs"]["1-F"]); assert all(f"{x:.2f}" == "0.07" for x in g30[:2]) and all(f"{x:.2f}" in ("0.17", "0.18") for x in g30[2:6]) and 0.605 <= g30[6] and g30[9] < 0.675 and f"{seq10('case30', 'T2=inf')[0]['1-F']:.2f}" == "0.18"
# C against the series reactive consumption of the unit-voltage lossless branch model (Sec. VII)
_C = {c_: G10("qoi", c_)["problem"]["C_MVAr"] for c_ in ("case9", "case14", "case30")}
assert max(abs(_sq[c_][0] / _C[c_] - 1) for c_ in _C) < 0.0011 and f"{100 * max(abs(_sq[c_][0] / _C[c_] - 1) for c_ in _C):.2f}" == "0.11"
assert [_sq[c_][2] for c_ in _C] == [0, 3, 0] and f"{_sq['case14'][1]:.1f}" == "2.5" and f"{100 * ((_sq['case14'][0] + _sq['case14'][1]) / _C['case14'] - 1):.1f}" == "4.6"
# compliance in the fidelity runs of the 14-bus system at T2 = 300: the order is reversed
assert f"{100 * pro10('case14', 'T2=300.0')['eC']:.1f}" == "2.2" and f"{100 * seq10('case14', 'T2=300.0')[0]['eC']:.1f}" == "4.1"
# real clock: complete configurations without dephasing
_bf2 = lambda name: [f"{bf(c_, 'T2=inf', name):.2f}" for c_ in ("case9", "case14", "case30")]
assert _bf2("projective + 6-bit clock") == ["0.66", "0.84", "0.31"] and _bf2("sequential + 7-bit uniform clock") == ["0.87", "0.98", "0.62"]
assert bf("case14", "T2=inf", "sequential + 7-bit uniform clock") > bf("case14", "T2=inf", "standard HHL, 10-bit clock") and all(bf(c_, "T2=inf", "sequential + 7-bit uniform clock") < bf(c_, "T2=inf", "standard HHL, 10-bit clock") for c_ in ("case9", "case30"))
assert all(bf(c_, "T2=inf", "projective + 6-bit clock") < bf(c_, "T2=inf", "standard HHL, 10-bit clock") for c_ in ("case9", "case14", "case30"))
assert f"{max(v['F'] for m_, v in be['case30']['T2=inf'].items() if 'sine' in m_ or 'DPSS' in m_):.2f}" == "0.69"
# polynomial inversion on the grid systems: learned nodes
_po = {c_: G10("poly", c_) for c_ in ("case9", "case14", "case30")}; _ln = [(c_, k_, v) for c_ in _po for k_, v in _po[c_].items() if isinstance(v, dict) and not k_.startswith("exact")]
assert len(_ln) == 6 and all(_po[c_]["d_max"] == 199 and _po[c_]["target"] == 0.9999 for c_ in _po) and all(v["node-aware"] is None and v["interval"] is None for _, _, v in _ln)
_ok = [(c_, k_, v) for c_, k_, v in _ln if v["interval + spectral correction"] is not None]; assert len(_ok) == 1 and _ok[0][0] == "case9" and _ok[0][1].startswith("sequential") and _ok[0][2]["interval + spectral correction"] == 51 and f"{_ok[0][2]['tau of the correction']:.1f}" == "1.6"
import hashlib; _sha = lambda p_: hashlib.sha256(open(os.path.join("..", p_), "rb").read()).hexdigest()
M10 = json.load(open("../results/stage10_manifest.json")); assert len(M10["results"]) == 24 and all(_sha(p_) == v["sha256"] for p_, v in list(M10["code"].items()) + list(M10["results"].items()))
assert min(v["mtime"] for v in M10["results"].values()) > max(v["mtime"] for v in M10["code"].values())

# --- second application test (Sec. VIII, Table XII, stage 11): ridge readout of a reservoir computer for three series
os.chdir(".."); _sp = importlib.util.spec_from_file_location("s11", "stage11_time_series.py"); S11 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(S11); os.chdir("paper")
_ser = ("narma2", "mackey-glass", "lorenz"); FC = json.load(open("../results/stage11_forecast.json")); J11 = lambda name: json.load(open(f"../results/stage11_{name}.json"))
assert (S11.N, S11.WASH, S11.NTR, S11.NTE, S11.KAPPA, S11.RHO) == (16, 200, 2000, 1000, 100.0, 0.9) and tuple(S11.SERIES) == _ser
_sp11 = {}
for c_ in _ser:
    p_ = S11.system(c_); l_, U_ = np.linalg.eigh(p_["a"]); w_ = (U_.T @ p_["b"])**2; x_ = w_ / l_**2; x_ /= x_.sum(); lo_ = l_ < 1.25 / 64; _sp11[c_] = (l_, w_, x_, lo_)
    assert 1 / l_.min() <= 100 + 1e-9 and np.isclose(p_["alpha"], np.linalg.eigvalsh(p_["G"]).max() / 99)
assert [int(_sp11[c_][3].sum()) for c_ in _ser] == [9, 13, 13] and [int(np.searchsorted(np.cumsum(np.sort(_sp11[c_][1])[::-1]), 0.99)) + 1 for c_ in _ser] == [3, 2, 1] and f"{1.25 / 64:.3f}" == "0.020"
for c_, w0, x0 in zip(_ser, (3e-4, 7e-5, 9e-4), ("40", "8", "72")): sig(_sp11[c_][1][_sp11[c_][3]].sum(), w0, 1); assert f"{100 * _sp11[c_][2][_sp11[c_][3]].sum():.0f}" == x0 and _sp11[c_][1][_sp11[c_][3]].sum() < 1e-3
_j = int(np.argmax(_sp11["lorenz"][2])); assert f"{_sp11['lorenz'][0][_j]:.3f}" == "0.016" and f"{100 * _sp11['lorenz'][2][_j]:.0f}" == "51"; sig(_sp11["lorenz"][1][_j], 7.4e-4)
assert 5e-8 < min(np.diff(_sp11[c_][0]).min() for c_ in _ser) < 2e-7
assert [f"{FC[c_]['ridge at this kappa']:.3f}" for c_ in _ser] == ["0.120", "0.235", "0.132"] and [f"{FC[c_]['persistence']:.2f}" for c_ in _ser] == ["0.84", "0.67", "0.26"]
for c_, v_ in zip(_ser, (0.056, 0.027, 0.0021)): p_ = S11.system(c_); sig(S11.nmse(c_, np.linalg.lstsq(p_["G"], p_["g"], rcond=None)[0]), v_)
# protocol: the budget, who stays within it, shots of the projective estimator, the two resolutions, ten finite runs in every row
_bud = {c_: 10 * J11(f"scale.{c_}.x10")["T_unit"] for c_ in _ser}; _use = []; _over = []; _psh = []
for c_ in _ser:
    sig(_bud[c_], 2.2e8); s0, s3 = J11(f"scale.{c_}.x10"), J11(f"scale.{c_}.x10.T2_300"); assert np.isclose(s3["budget"], _bud[c_]) and s0["mult"] == s3["mult"] == 10
    for d_ in (s0, s3):
        for k_, v_ in d_.items():
            if isinstance(v_, dict) and (k_.startswith("sequential") or d_ is s3): assert max(v_["runs"]["T"]) <= _bud[c_] * (1 + 1e-9), (c_, k_)
    _over += [t_ / _bud[c_] for t_ in s0["projective, 6-bit clock"]["runs"]["T"]]; _psh.append(s3["projective, 5-bit clock"]["shots"])
    assert {k_ for k_ in s0 if k_.startswith("sequential")} == {"sequential, sig_id=0.00500", "sequential, sig_id=0.00030"} and {k_ for k_ in s3 if k_.startswith("sequential")} == {"sequential, sig_id=0.00500", "sequential, sig_id=0.00200"}
    for tag_ in ("", ".T2_300"):
        b_ = J11(f"baselines.{c_}.x10{tag_}"); assert np.isclose(b_["budget"], _bud[c_]) and sum(isinstance(v_, dict) for v_ in b_.values()) == (4 if tag_ else 6)
        _use += [t_ / _bud[c_] for v_ in b_.values() if isinstance(v_, dict) for t_ in v_["runs"]["T"]]; assert all(len(v_["runs"]["T"]) == 10 and np.all(np.isfinite(v_["runs"]["1-F"])) for v_ in b_.values() if isinstance(v_, dict))
assert f"{100 * min(_use):.0f}" == "86" and max(_use) <= 1 and f"{100 * (min(_over) - 1):.0f}" == "2" and f"{100 * (max(_over) - 1):.0f}" == "4"; sig(min(_psh), 4.7e5); sig(max(_psh), 4.9e5); assert f"{0.6 / 300:.3f}" == "0.002"
def r11(c_, key_, *words, x10=True):                 # the stored row of a series and noise setting whose name contains the words
    m_ = [v_ for n_, v_ in FC[c_].items() if isinstance(v_, dict) and n_.startswith(key_ + ", ") and n_.endswith("x10 time") == x10 and all(w_ in n_ for w_ in words)]; assert len(m_) == 1, (c_, key_, words, len(m_)); return m_[0]
ok = lambda r_: sum(d_ < 1e-3 for d_ in r_["dev_runs"]); _K = ("T2=inf", "T2=300.0"); _fin = {"T2=inf": "sig_id=0.00030", "T2=300.0": "sig_id=0.00200"}
_x10 = {(c_, k_): {n_: v_ for n_, v_ in FC[c_].items() if isinstance(v_, dict) and n_.startswith(k_ + ", ") and n_.endswith("x10 time")} for c_ in _ser for k_ in _K}
assert all(len(v_) == (9 if k_ == "T2=inf" else 7) and all(r_["runs"] == 10 for r_ in v_.values()) for (c_, k_), v_ in _x10.items())
_oth = [r_ for v_ in _x10.values() for n_, r_ in v_.items() if "sequential" not in n_]; _seq = [r_ for v_ in _x10.values() for n_, r_ in v_.items() if "sequential" in n_]
assert 10 * len(_oth) == 360 and 10 * len(_seq) == 120 and sum(r_["NMSE>1"] for r_ in _oth) == 6 and sum(r_["NMSE>1"] for r_ in _seq) == 0 and max(max(r_["NMSE_runs"]) for r_ in _oth) > 1e6
# NARMA2
assert [ok(r11("narma2", k_, _fin[k_])) for k_ in _K] == [10, 10] and [ok(r11("narma2", k_, "sig_id=0.00500")) for k_ in _K] == [5, 4]; sig(r11("narma2", "T2=inf", _fin["T2=inf"])["1-F"], 2.7e-3); sig(r11("narma2", "T2=300.0", _fin["T2=300.0"])["1-F"], 3.4e-3)
assert ok(r11("narma2", "T2=inf", "6 bits")) == 4 and max(ok(r_) for k_ in _K for n_, r_ in _x10["narma2", k_].items() if "sequential" not in n_ and not (k_ == "T2=inf" and "6 bits" in n_)) == 2
sig(r11("narma2", "T2=inf", "projective")["1-F"], 0.43); sig(r11("narma2", "T2=inf", "projective (", x10=False)["1-F"], 0.43); assert ok(r11("narma2", "T2=inf", "projective")) == ok(r11("narma2", "T2=inf", "projective (", x10=False)) == 0
assert min(min(m_) for m_ in J11("scale.narma2.x10")["projective, 6-bit clock"]["runs"]["mu"]) > 0.3 and int((_sp11["narma2"][0] < 0.3).sum()) == 13
# Mackey-Glass
assert all(ok(r11("mackey-glass", k_, s_)) == 10 for k_ in _K for s_ in ("sig_id=0.00500", _fin[k_])) and [ok(r11("mackey-glass", "T2=inf", b_)) for b_ in ("8 bits", "10 bits")] == [10, 10]
assert [ok(r11("mackey-glass", "T2=300.0", b_)) for b_ in ("MM-QCELS", "6 bits")] == [10, 10] and [ok(r11("mackey-glass", k_, "projective")) for k_ in _K] == [9, 6]
sig(r11("mackey-glass", "T2=inf", "sig_id=0.00500")["1-F"], 1.0e-3); sig(r11("mackey-glass", "T2=300.0", "sig_id=0.00500")["1-F"], 2.0e-3); sig(r11("mackey-glass", "T2=300.0", "sig_id=0.00200")["1-F"], 0.055)
assert all(0.045 < r11("mackey-glass", k_, "projective")["1-F"] < 0.06 for k_ in _K) and ok(r11("mackey-glass", "T2=300.0", "sig_id=0.00200")) == 10
# Lorenz
assert all(ok(r11("lorenz", k_, s_)) == 10 for k_ in _K for s_ in ("sig_id=0.00500", _fin[k_])) and [ok(r11("lorenz", "T2=inf", b_)) for b_ in ("8 bits", "10 bits")] == [10, 10] and [ok(r11("lorenz", k_, "projective")) for k_ in _K] == [5, 4]
sig(r11("lorenz", "T2=inf", "sig_id=0.00030")["1-F"], 1.5e-4); assert all(f"{r11('lorenz', k_, s_)['1-F']:.3f}" in ("0.015", "0.016") for k_, s_ in (("T2=inf", "sig_id=0.00500"), ("T2=300.0", "sig_id=0.00500"), ("T2=300.0", "sig_id=0.00200")))
assert max(ok(r11("lorenz", k_, m_)) for k_ in _K for m_ in ("Bayesian clock", "QMEGS", "MM-QCELS")) == 1 and not any("8 bits" in n_ or "10 bits" in n_ for c_ in _ser for n_ in _x10[c_, "T2=300.0"])
assert f"{r11('lorenz', 'T2=inf', 'sig_id=0.00500')['NMSE']:.3f}" == "0.137" and f"{r11('lorenz', 'T2=inf', 'sig_id=0.00030')['NMSE']:.3f}" == "0.131" and f"{r11('lorenz', 'T2=inf', 'sig_id=0.00500')['1-F']:.3f}" == "0.016"
# discussion: the finest resolution reproduces the forecast in all runs; Projective HHL in none to nine
assert all(ok(r11(c_, k_, _fin[k_])) == 10 for c_ in _ser for k_ in _K) and sorted(ok(r11(c_, k_, "projective")) for c_ in _ser for k_ in _K) == [0, 2, 4, 5, 6, 9] and f"{100 * max(_sp11[c_][2][_sp11[c_][3]].sum() for c_ in _ser):.0f}" == "72"
M11 = json.load(open("../results/stage11_manifest.json")); assert len(M11["results"]) == 19 and all(_sha(p_) == v["sha256"] for p_, v in list(M11["code"].items()) + list(M11["results"].items()))
assert min(v["mtime"] for v in M11["results"].values()) > max(v["mtime"] for v in M11["code"].values())

# --- baselines without phase estimation (Sec. IX, Tables XIII and XIV, stage 12): ten runs per entry, medians as quoted
_grid = ("case9", "case14", "case30", "case118"); V12 = {k: json.load(open(f"../results/stage12_vqls.{k.replace(' ', '_')}.json")) for k in SYS + _grid + _ser}
_ex = {k: v["exact cost"] for k, v in V12.items()}; _b = lambda k, n_: V12[k][f"{n_} executions"]; _ten = [k for k in V12 if k != "case118"]
assert len(_ten) == 10 and all(len(r["runs"]["1-F"]) == 10 for v in V12.values() for r in v.values() if isinstance(r, dict) and "runs" in r)
assert all(_ex[k]["1-F"] < 1e-9 for k in _ten) and all(max(_ex[k]["runs"]["1-F"]) < 1e-9 for k in _ten if k != "case30")
_hi = [x for x in _ex["case30"]["runs"]["1-F"] if x > 1e-8]; assert len(_hi) == 1; sig(_hi[0], 1.6e-4)
assert all(i_ == 4000 for i_ in _ex["case118"]["runs"]["iterations"]) and max(max(_ex[k]["runs"]["iterations"]) for k in _ten if k != "case30") < 4000; sig(_ex["case118"]["1-F"], 0.58)
assert _ex["case30"]["runs"]["iterations"][_ex["case30"]["runs"]["1-F"].index(_hi[0])] == 4000
_pairs = {k: _ex[k]["executions"] for k in _ten}; sig(min(_pairs.values()), 1.9e3); sig(max(_pairs.values()), 1.2e8); assert [k for k in _ten if _pairs[k] <= 1e4] == [P4]
_m4 = {k: _b(k, 10000)["1-F"] for k in _ten}; sig(min(_m4.values()), 0.69); sig(max(_m4.values()), 0.97); _r4 = [x for k in _ten for x in _b(k, 10000)["runs"]["1-F"]]; sig(min(_r4), 0.18); assert f"{max(_r4):.2f}" == "1.00"
assert f"{1 - 2.0**-min(V12[k]['qubits'] for k in _ten):.2f}" == "0.75" and f"{1 - 2.0**-max(V12[k]['qubits'] for k in _ten):.2f}" == "0.97"
assert "not runnable" in _b("case118", 10000) and int(_ex["case118"]["terms"]) == 8127 and V12["case118"]["qubits"] == 7
_m7 = {k: _b(k, 10000000)["1-F"] for k in V12}; _low = (R4, P4, TZ, "case9"); sig(min(_m7[k] for k in _low), 0.023); sig(max(_m7[k] for k in _low), 0.45); assert all(_m7[k] > 0.8 for k in V12 if k not in _low) and len(V12) - len(_low) == 7
sig(_b(P8, 10000000)["cost"], 0.052); sig(_b(P8, 10000000)["1-F"], 0.865, 3)
for k in V12:                                       # shots per term: the largest number that leaves 500 iterations, at least one; no run exceeds its executions
    for n_ in (10000, 100000, 10000000):
        r_ = V12[k].get(f"{n_} executions", {})
        if "runs" in r_: t_ = int(r_["terms"]); s_ = max(1, n_ // (2 * t_ * 500)); assert set(r_["runs"]["iterations"]) == {n_ // (2 * t_ * s_)} and max(r_["runs"]["executions"]) <= n_
# forecasting models (Table XIV)
J12 = {(m_, c_): json.load(open(f"../results/stage12_{m_}.{c_}.json")) for m_ in ("qrc", "qcl", "dqi") for c_ in _ser}; _nm = lambda m_, row_: [(V12[c_] if m_ == "vqls" else J12[m_, c_])[row_]["NMSE"] for c_ in _ser]
_q2 = lambda xs, qs: [sig(x_, q_) for x_, q_ in zip(xs, qs)]
_q2(_nm("qrc", "exact expectation values"), (0.21, 0.23, 0.24)); _q2(_nm("qcl", "exact expectation values"), (0.14, 0.097, 0.0084)); _q2(_nm("qrc", "100000 executions"), (0.64, 0.56, 0.43)); _q2(_nm("vqls", "100000 executions"), (0.97, 0.69, 0.44))
assert all(f"{x_:.3f}" == f"{FC[c_]['ridge at this kappa']:.3f}" for x_, c_ in zip(_nm("vqls", "exact cost"), _ser)) and all(0.945 <= x_ < 1.005 for x_ in _nm("qcl", "100000 executions"))
_ridge = [FC[c_]["ridge at this kappa"] for c_ in _ser]; _last = [FC[c_]["persistence"] for c_ in _ser]; _qe = _nm("qcl", "exact expectation values"); assert _qe[0] > _ridge[0] and _qe[1] < _ridge[1] and _qe[2] < _ridge[2]
_cir = [J12["qcl", c_]["exact expectation values"]["executions"] for c_ in _ser]; sig(min(_cir), 9e6, 1); sig(max(_cir), 3e7, 1); assert 1e6 <= min(_ex[c_]["executions"] for c_ in _ser) and max(_cir) <= 3.5e7
assert all(J12["qrc", c_]["circuits"] == 12000 and J12["qrc", c_]["features"] == 16 and J12["qrc", c_]["100000 executions"]["shots per circuit"] == 8 and "not runnable" in J12["qrc", c_]["10000 executions"] for c_ in _ser)
os.chdir(".."); _sp = importlib.util.spec_from_file_location("s12", "stage12_nisq_baselines.py"); S12 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(S12); os.chdir("paper")
assert (S12.NQ, S12.NV, S12.TAU, S12.LAYERS, S12.BATCH, S12.P_DQI, S12.N_DQI, S12.R_DQI) == (4, 4, 1.0, 3, 100, 31, 5, (5, 15)) and 2 * S12.NQ * S12.LAYERS + 2 == 26 and (S12.N_DQI - 1) // 2 == 2
for c_ in _ser:
    d5, d15 = J12["dqi", c_]["sets of 5"], J12["dqi", c_]["sets of 15"]
    assert [f"{100 * d_[q_]:.0f}" for d_ in (d5, d15) for q_ in ("satisfied, DQI expectation", "satisfied, random")] == ["30", "16", "64", "48"]
    assert all(d_["100000 executions"]["satisfied"] == d_["satisfied, optimum"] for d_ in (d5, d15)) and f"{100 * d5['satisfied, optimum']:.0f}" == "63" and f"{100 * d15['satisfied, optimum']:.0f}" in ("97", "100")
_dq = [J12["dqi", c_][s_]["100000 executions"]["NMSE"] for s_ in ("sets of 5", "sets of 15") for c_ in _ser]; sig(_dq[5], 0.27); sig(min(_dq[:5]), 0.66); sig(max(_dq[:5]), 1.6); assert abs(_dq[5] - _last[2]) < 0.01
# scope: within 1e5 executions, who is below the error of repeating the last known value
_at5 = {("vqls", ""): _nm("vqls", "100000 executions"), ("qrc", ""): _nm("qrc", "100000 executions"), ("qcl", ""): _nm("qcl", "100000 executions"), ("dqi", 5): _dq[:3], ("dqi", 15): _dq[3:]}
assert {(m_, c_) for m_, v_ in _at5.items() for c_, x_, l_ in zip(_ser, v_, _last) if x_ < l_} == {(("qrc", ""), "narma2"), (("qrc", ""), "mackey-glass"), (("dqi", 15), "mackey-glass")}
assert all(x_ > 1.5 * r_ for v_ in _at5.values() for x_, r_ in zip(v_, _ridge))
M12 = json.load(open("../results/stage12_manifest.json")); assert len(M12["results"]) == 20 and all(_sha(p_) == v["sha256"] for p_, v in list(M12["code"].items()) + list(M12["results"].items()))
assert min(v["mtime"] for v in M12["results"].values()) > max(v["mtime"] for v in M12["code"].values())

# --- error against budget (Sec. VIII, Fig. 3; Sec. IX "Number of executions"; stage 13): medians of ten runs
SW = json.load(open("../results/stage13_sweep.json")); _mu = (0.1, 0.3, 1, 3, 10, 30); _nz = ("T2=inf", "T2=300.0")
_pt = {(p_["series"], p_["noise"], p_["method"]): {} for p_ in SW["points"]}
for p_ in SW["points"]: _pt[p_["series"], p_["noise"], p_["method"]][p_["mult"]] = p_
assert len(SW["points"]) == 108 and all(sorted(v_) == list(_mu) and all(len(q_["runs"]["1-F"]) == 10 for q_ in v_.values()) for v_ in _pt.values())
_f = lambda c_, k_, m_, x_, q_="1-F": _pt[c_, k_, m_][x_][q_]; _fine, _coarse, _proj = "sequential-fine", "sequential-coarse", "projective"
assert all(np.isclose(_f(c_, k_, m_, 1, "budget"), _bud[c_] / 10) for c_, k_, m_ in _pt)                         # ten units are the budget of Table XII
assert all(_f(c_, k_, _coarse, 1, "sig_id") == 0.005 and np.isclose(_f(c_, k_, _fine, 1, "sig_id"), 3e-4 if k_ == "T2=inf" else 0.002) for c_ in _ser for k_ in _nz)
assert all(_f(c_, "T2=inf", _proj, 1, "clock bits") == 6 and _f(c_, "T2=300.0", _proj, 1, "clock bits") == 5 for c_ in _ser)
assert all(f"{_f('narma2', 'T2=inf', _fine, x_):.2f}" == "0.43" for x_ in (0.1, 0.3, 1)); sig(_f("narma2", "T2=inf", _fine, 3), 0.19); sig(_f("narma2", "T2=inf", _fine, 10), 2.7e-3); sig(_f("narma2", "T2=inf", _fine, 30), 7.7e-5)
assert all(f"{_f('lorenz', 'T2=inf', _coarse, x_):.3f}" == "0.016" for x_ in _mu[1:]) and _f("lorenz", "T2=inf", _coarse, 0.1) > 0.3
assert f"{_f('lorenz', 'T2=inf', _fine, 3):.3f}" == "0.016" and _f("lorenz", "T2=inf", _fine, 1) > 0.1; sig(_f("lorenz", "T2=inf", _fine, 10), 1.5e-4)
assert all(0.0145 <= _f("lorenz", "T2=300.0", m_, x_) < 0.0195 for m_ in (_coarse, _fine) for x_ in (1, 3, 10, 30))
assert [int(_f("narma2", "T2=inf", _proj, x_, "shots")) for x_ in _mu] == [1000, 3000, 10000, 30000, 100000, 300000] and all(f"{_f('narma2', 'T2=inf', _proj, x_):.2f}" in ("0.42", "0.43") for x_ in _mu)
_r13 = [_f(c_, k_, _proj, 30) / _f(c_, k_, _proj, 1) for c_ in _ser for k_ in _nz]; assert f"{min(_r13):.1f}" == "0.8" and f"{max(_r13):.1f}" == "2.8"
_dn = lambda lo_: max(abs(p_["NMSE"] - FC[p_["series"]]["ridge at this kappa"]) for p_ in SW["points"] if p_["mult"] >= lo_)                # forecast against budget
assert 0.06 < _dn(1) < 0.07 and 0.11 < max(abs(p_["NMSE"] - FC[p_["series"]]["ridge at this kappa"]) for p_ in SW["points"]) < 0.12
assert f"{_f('narma2', 'T2=inf', _fine, 1, 'NMSE'):.2f}" == "0.15" and f"{_f('narma2', 'T2=inf', _fine, 30, 'NMSE'):.2f}" == "0.12"
_rep = lambda c_, k_, m_: [_f(c_, k_, m_, x_, "reproduced") for x_ in _mu]; assert all(p_["reproduced"] == sum(d_ < 1e-3 for d_ in p_["runs"]["dev"]) for p_ in SW["points"])
assert all(_rep("mackey-glass", k_, _coarse) == [10] * 6 for k_ in _nz) and _rep("lorenz", "T2=inf", _coarse)[1:] == [10] * 5 and _rep("lorenz", "T2=inf", _coarse)[0] < 10
assert _rep("lorenz", "T2=300.0", _coarse)[2:] == [10] * 4 and max(_rep("lorenz", "T2=300.0", _coarse)[:2]) < 10
assert all(_rep("narma2", k_, _fine)[4:] == [10, 10] and max(_rep("narma2", k_, _fine)[:4]) < 10 and max(_rep("narma2", k_, _coarse)) < 10 for k_ in _nz)
assert sum(_f(c_, k_, _proj, x_, "reproduced") == 10 for c_ in _ser for k_ in _nz for x_ in _mu) == 2
sig(_f("narma2", "T2=300.0", _coarse, 1, "dev"), 0.063); assert f"{_f('narma2', 'T2=300.0', _coarse, 0.1, 'dev'):.3f}" == "0.005"
_b13 = lambda c_, b_, e_, q_="NMSE": SW["baselines"][c_][b_][str(e_)][q_]; _ex = (10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7)   # number of executions
assert all(set(SW["baselines"][c_][b_]) == {str(e_) for e_ in _ex} for c_ in _ser for b_ in ("vqls", "qrc", "qcl", "dqi")) and all(SW["baselines"][c_]["qrc"]["10000"] is None for c_ in _ser)
assert [f"{_b13(c_, 'qrc', 3 * 10**4):.2f}" for c_ in _ser] == ["0.86", "0.79", "0.60"] and [f"{_b13(c_, 'qrc', 10**7):.2f}" for c_ in _ser] == ["0.24", "0.24", "0.25"]
_q13 = [_b13(c_, "qcl", e_) for c_ in _ser for e_ in _ex[:3]]; assert f"{min(_q13):.2f}" == "0.95" and f"{max(_q13):.2f}" == "1.04" and [f"{_b13(c_, 'qcl', 10**7):.2f}" for c_ in _ser] == ["0.90", "0.74", "0.63"]
assert min(_b13(c_, "vqls", e_, "1-F") for c_ in _ser for e_ in _ex) > 0.9
_dq = [_b13(c_, "dqi", e_, q_) for c_ in _ser for e_ in _ex[2:] for q_ in ("NMSE, sets of 5", "NMSE, sets of 15")]; assert f"{min(_dq):.2f}" == "0.26" and f"{max(_dq):.1f}" == "1.9"
assert all(np.allclose(_b13(c_, b_, 10**5), V12[c_]["100000 executions"]["NMSE"] if b_ == "vqls" else json.load(open(f"../results/stage12_{b_}.{c_}.json"))["100000 executions"]["NMSE"]) for c_ in _ser for b_ in ("vqls", "qrc", "qcl"))   # the same runs as stage 12
M13 = json.load(open("../results/stage13_manifest.json")); assert len(M13["code"]) == 3 and list(M13["results"]) == ["results/stage13_sweep.json"] and all(_sha(p_) == v["sha256"] for p_, v in list(M13["code"].items()) + list(M13["results"].items()))
assert min(v["mtime"] for v in M13["results"].values()) > max(v["mtime"] for v in M13["code"].values())

# --- provenance: the manifest matches the files, and every stage-9 result is newer than the code that produced it
import hashlib
MAN = json.load(open("../results/stage9_manifest.json")); sha = lambda p: hashlib.sha256(open(os.path.join("..", p), "rb").read()).hexdigest()
assert all(sha(p) == v["sha256"] for p, v in list(MAN["code"].items()) + list(MAN["results"].items()))
assert min(v["mtime"] for v in MAN["results"].values()) > MAN["code"]["stage9_sequential_qpe.py"]["mtime"] and len(MAN["results"]) >= 23

# --- regression record of the earlier stages
assert len(REG) == 8 and all(v["identical"] for v in REG.values())
tex = open("main.tex").read()
for s_ in ("$2.8\\times10^{-6}$ to $6.8\\times10^{-8}$", "stays at $0.076$, to $2.6\\times10^{-5}$", "from $2$ to $3\\times10^3$", "$48\\%$ to $2.5\\%$"): assert s_ in tex, s_
assert "XX" not in tex
# other result files quoted in the manuscript: hashes match and each file is newer than the script that writes it
assert set(MAN["tables"]) == {"results/stage4_results.json", "results/stage6_results.json", "results/stage8_floors.json", "results/adjudication_P1.txt"}
assert all(sha(p) == v["sha256"] and sha(v["script"]) == v["script_sha256"] and v["mtime"] > v["script_mtime"] for p, v in MAN["tables"].items())
print("stage-9 statements checked against the result files; every sequential and projective-with-target run is within its time budget (projective runs: 0.58-1.03 of the median reference); manifest verified")
