"""Regenerate the table bodies of main.tex and Fig. 2 from the result JSON files (consistent rounding, no hand
transcription).  Usage: python3 make_tables.py  -> rewrites main.tex in place between the markers and
figs/fig2_detection.png."""
import json, re, glob, numpy as np
R4 = json.load(open("../results/stage4_results.json")); R7 = json.load(open("../results/stage7_results.json")); RQ = json.load(open("../results/stage7b_qks.json"))
NAMES = [("P5 random 4x4", r"Random $4\times4$", "2.2"), ("P3 Poisson N=4", r"Poisson $N=4$", "9.5"), ("P3 Poisson N=8", r"Poisson $N=8$", "32"), ("P4 Toeplitz N=8", r"Toeplitz $N=8$", "4.0")]


def f4(v): return f"{v:.3f}" if v < 0.9995 else f"{v:.4f}"
def cell(q, iqr=True):
    s = f4(q["median"])
    if iqr and q["q75"] - q["q25"] > 0.01: s += f" [{f4(q['q25'])}, {f4(q['q75'])}]"
    return s
def t2(row): return r"$\infty$" if row["T2"] == "inf" else f"{int(row['T2'])}"


def table1():
    out = []
    for key, lab, kap in NAMES:
        for row in R4[key]:
            out.append(f"{lab} & {t2(row)} & {kap} & {cell(row['MM-QCELS'])} & {cell(row['QMEGS'])} & {cell(row['QPE6'])} & {cell(row['BC2 ideal'])} ({row['n_c']})\\\\")
    return "\n".join(out)


def e(x):
    """Compact scientific notation for table cells: 2.8e-6; values >= 0.01 as decimals."""
    if 0.01 <= x < 1000: return f"{x:.3f}"
    a, b = f"{x:.1e}".split("e"); return f"{a}e{int(b)}"


def table2():
    """Real clock with the spectrum learned by the projective estimator (stage 9 back end: shared noise trajectory,
    heralded-ensemble fidelity)."""
    R = json.load(open("../results/stage9_backend.json")); out = []
    for key, lab, kap in NAMES:
        for T2 in ("inf", "300.0"):
            v = R[key][f"T2={T2}"]; g = lambda k: f4(v["clock: " + k])
            out.append(f"{lab} & {r'$\infty$' if T2 == 'inf' else 300} & {v['n_c']} & {g('idealised')} & {g('learned angles')} & {g('+t0')} & {g('no learning')} & {g('exact-spectrum t0')}\\\\")
    return "\n".join(out)


def pct(x):
    return f"{100*x:.2f}" if x < 0.00995 else (f"{100*x:.1f}" if x < 0.0995 else f"{100*x:.0f}")
def table4():
    """Observables: spectral (projective) columns from the six jointly timed runs of stage 9 (the same runs as the
    projective columns of the sequential comparison), QKS and Fourier columns from stages 7b and 7."""
    RP = load("stage9_qoi"); out = []
    for key, lab, kap in NAMES:
        for T2 in ("inf", 300.0):
            rows = {}
            for shots in (10000, 160000):
                k7 = [k for k in R7[key] if k.startswith(f"T2={T2 if T2=='inf' else T2}, shots={shots},")][0]
                r9 = next(v for k, v in RP[key].items() if k.startswith(f"T2={T2 if T2=='inf' else T2}, shots={shots},"))["projective"]
                rows[shots] = (r9, RQ[key][f"T2={T2 if T2=='inf' else T2}, shots={shots}"], R7[key][k7])
            s = lambda r: f"{pct(r['eC'])} / {pct(r['eE'])} / {pct(r['eL'])}\\%"
            q = lambda r: f"{pct(r['rel err C'])} / {pct(r['rel err E'])} / {pct(r['rel err L'])}\\%"
            out.append(f"{lab} & {r'$\infty$' if T2=='inf' else 300} & {s(rows[10000][0])} & {s(rows[160000][0])} & {q(rows[10000][1])} & {q(rows[160000][1])} & {pct(rows[10000][2]['fourier: rel err functional <v|x>'])} / {pct(rows[160000][2]['fourier: rel err functional <v|x>'])}\\%\\\\")
    return "\n".join(out)


def load(stem):
    """Merge the per-process part files of stage 9."""
    out = {}
    for p in sorted(glob.glob(f"../results/{stem}.part*.json")):
        for k, v in json.load(open(p)).items():
            out[k] = out.get(k, []) + v if isinstance(v, list) else v
    return out


def sci(x):
    """1.47e-07 -> $1.5\times10^{-7}$; values >= 0.01 as decimals."""
    if 0.01 <= x < 1000: return f"{x:.3f}"
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"


def _cmp(key, T2):
    """Rows of the matched-time comparison for one system and T2: projective, projective with target, and the
    sequential variants at the default (floor-limited) resolution."""
    rows = load("stage9_compare")[key][f"T2={T2}"]; seq = [(k, v) for k, v in rows.items() if k.startswith("sequential")]
    sid = max(float(k.split("sig_id=")[1].rstrip(")")) for k, _ in seq)
    g = lambda frac, tag: next(v for k, v in seq if f"budget x{frac}, {tag}, sig_id={sid:.4f}" in k)
    return {"pro": next(v for k, v in rows.items() if k.startswith("projective (")), "prot": rows["projective with target delta=0.001"],
            "seq3": g(1.0, "delta=0.001"), "seq4": g(1.0, "delta=0.0001"), "seq3s": g(0.1, "delta=0.001"), "seqn": g(1.0, "no target, 1 refinement per copy")}


def table_matched():
    out = []
    for key, lab, kap in NAMES:
        for T2 in ("inf", "300.0"):
            c = _cmp(key, T2); mw = lambda v: f"{e(v['1-F'])} & {e(v['1-F_max'])}"
            out.append(f"{lab} & {r'$\infty$' if T2 == 'inf' else 300} & {e(c['pro']['T'])} & {mw(c['pro'])} & {mw(c['prot'])} & {mw(c['seqn'])} & {mw(c['seq3'])} & {c['seq3']['m']:g} & {e(c['seq4']['1-F'])} & {e(c['seq3s']['1-F'])}\\\\")
    return "\n".join(out)


def table_matchedC():
    """Compliance error and number of samples in the runs of the matched-time table."""
    out = []
    for key, lab, kap in NAMES:
        for T2 in ("inf", "300.0"):
            c = _cmp(key, T2)
            out.append(f"{lab} & {r'$\infty$' if T2 == 'inf' else 300} & {pct(c['pro']['eC'])} & {pct(c['prot']['eC'])} & {pct(c['seqn']['eC'])} & {pct(c['seq3']['eC'])} & {c['prot']['shots']:.0f} & {c['seqn']['copies']:.0f} & {c['seq3']['copies']:.0f}\\\\")
    return "\n".join(out)


def table_qoiseq():
    R = load("stage9_qoi"); out = []
    for key, lab, kap in NAMES:
        for T2 in ("inf", "300.0"):
            cells = []
            for shots in (10000, 160000):
                v = next(v for k, v in R[key].items() if k.startswith(f"T2={T2}, shots={shots},"))
                for meth in ("projective", "sequential"):
                    cells.append(f"{pct(v[meth]['eC'])} / {pct(v[meth]['eE'])} / {pct(v[meth]['eL'])}\\%")
            out.append(f"{lab} & {r'$\infty$' if T2 == 'inf' else 300} & " + " & ".join(cells) + "\\\\")
    return "\n".join(out)


def f5(v): return f"{v:.3f}" if v < 0.99 else (f"{v:.4f}" if v < 0.9995 else f"{v:.5f}")
def table_window():
    R = json.load(open("../results/stage9_backend.json")); out = []
    cols = ("uniform +-1", "uniform+t0opt +-1", "uniform +-2", "sine +-2", "DPSS NW=1.5 +-2", "DPSS NW=2.0 +-2", "DPSS NW=2.0 +-1")
    for key, lab, kap in NAMES:
        for T2 in ("inf", "300.0"):
            v = R[key][f"T2={T2}"]
            for nc in (v["n_c"], v["n_c"] + 1):
                out.append(f"{lab} & {r'$\infty$' if T2 == 'inf' else 300} & {nc} & " + " & ".join(f5(v[f"window: {c} n_c={nc}"]) for c in cols) + "\\\\")
    return "\n".join(out)


def table_cost():
    R = json.load(open("../results/stage9_cost.json")); out = []
    fr = lambda x: "0" if x == 0 else e(x)
    for p, lab in ((0.01, "$10^{-2}$"), (0.001, "$10^{-3}$"), (0.0001, "$10^{-4}$"), (1e-05, "$10^{-5}$")):
        g = lambda T2, mode: R[f"T2={T2}, prior={mode}, p_out={p:g}"]; c = lambda T2, mode: f"{g(T2, mode)['T_id*sig_id mean']:.1f}"
        tri = lambda T2, mode: f"{c(T2, mode)} & {fr(g(T2, mode)['frac |err|>3 sig_id'])} & {fr(g(T2, mode)['frac |err|>4 sig_id'])}"
        out.append(f"{lab} & {tri('inf', 'flat')} & {tri('inf', 'learned')} & {c('inf', 'exact')} & {c('300.0', 'flat')} & {c('300.0', 'learned')}\\\\")
    return "\n".join(out)


def table_miscal():
    R = json.load(open("../results/stage9_cost.json")); out = []; fr = lambda x: "0" if x == 0 else e(x)
    labs = {"calibrated, T2=300": r"$T_2=300$, matched", "T2 300 -> 100": r"$T_2$: $300\to100$", "calibrated, no dephasing": r"no dephasing, matched",
            "p_d 0.02 -> 0.10": r"$p_d$: $0.02\to0.10$", "p_d 0.02 -> 0.20": r"$p_d$: $0.02\to0.20$", "T2 inf -> 1000": r"$T_2$: $\infty\to1000$"}
    for k, lab in labs.items():
        g = lambda p: R[f"miscalibration: {k}, p_out={p:g}"]
        out.append(f"{lab} & {fr(g(1e-3)['frac |err|>3 sig_id'])} & {fr(g(1e-3)['frac |err|>4 sig_id'])} & {fr(g(1e-4)['frac |err|>3 sig_id'])} & {fr(g(1e-4)['frac |err|>4 sig_id'])}\\\\")
    return "\n".join(out)


def fig2():
    """Detection probability of a planted component versus the expected count S*w: mixture and projective (stage 8),
    sequential (stage 9), and the binomial law P[N >= 4]."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import poisson
    old = json.load(open("../results/stage8_detection.json")); new = load("stage9_detect"); fl = json.load(open("../results/stage8_floors.json"))
    ink, mute = "#0b0b0b", "#52514e"; col = {"mix": "#eb6834", "pro": "#1baf7a", "seq": "#2a78d6"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for k, place in enumerate(("far", "adjacent")):
        a = ax[k]; floor = list(fl.values())[k] if isinstance(fl, dict) else fl[k]
        xs = np.logspace(-0.2, 4.1, 200); a.plot(xs, poisson.sf(3, xs), color=mute, lw=1.2, zorder=1, label=r"binomial law $P[N\geq4]$")
        o = old[place]                                                # stage 8 also has w = 0.1 (above the floor)
        a.plot([0.86 * r["w"] * r["S"] for r in o], [r["P_detect mixture"] for r in o], "o", ms=5.5, color=col["mix"], mec="white", mew=0.8, zorder=3, label="mixture (QMEGS-type)")
        above = [r for r in o if r["w"] >= floor]; below = [r for r in o if r["w"] < floor]
        a.plot([r["w"] * r["S"] for r in above], [r["P_detect projective"] for r in above], "s", ms=5.5, color=col["pro"], mec="white", mew=0.8, zorder=3, label="projective (5-bit register)")
        a.plot([r["w"] * r["S"] for r in below], [r["P_detect projective"] for r in below], "s", ms=5.5, mfc="none", mec=col["pro"], mew=1.3, zorder=3, label="projective, below leakage floor")
        a.plot([1.16 * r["S_w"] for r in new[place]], [r["P_detect"] for r in new[place]], "^", ms=6, color=col["seq"], mec="white", mew=0.8, zorder=4, label="sequential (single copy)")
        a.set_xscale("log"); a.set_xlabel(r"expected count $S\,w$", color=ink); a.set_ylim(-0.06, 1.08); a.grid(alpha=0.25, lw=0.6)
        a.set_title(("far from" if place == "far" else "adjacent to") + " the strong component", fontsize=9, color=ink)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        a.tick_params(colors=mute, labelsize=8)
    ax[0].set_ylabel("detection probability", color=ink); ax[1].legend(fontsize=6.5, frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig("figs/fig2_detection.png", dpi=220)


def fig3():
    """Solution infidelity against the time budget on the three ridge systems (stage 13): medians of ten runs, bars from the
    lowest to the highest run.  Colours as in Fig. 2 (projective green, sequential blue)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = json.load(open("../results/stage13_sweep.json"))["points"]; ink, mute = "#0b0b0b", "#52514e"
    style = {"projective": ("#1baf7a", "s", "-", "#1baf7a", "Projective HHL"), "sequential-coarse": ("#2a78d6", "^", "--", "white", r"Recycled sequential HHL, $\sigma_{\rm id}=0.005$"),
             "sequential-fine": ("#2a78d6", "^", "-", "#2a78d6", "Recycled sequential HHL, finest resolution")}
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.1), sharex=True, sharey=True)
    for i, noise in enumerate(("T2=inf", "T2=300.0")):
        for j, (c, name) in enumerate(SERIES):
            a = ax[i, j]
            for k, (m, (col, mk, ls, mfc, lab)) in enumerate(style.items()):
                ps = sorted((p for p in d if (p["series"], p["noise"], p["method"]) == (c, noise, m)), key=lambda p: p["mult"]); assert len(ps) == 6
                x = np.array([p["mult"] for p in ps]) * (1 + 0.07 * (k - 1)); y = np.array([p["1-F"] for p in ps]); r = np.array([p["runs"]["1-F"] for p in ps])
                a.errorbar(x, y, yerr=[y - r.min(1), r.max(1) - y], color=col, marker=mk, ls=ls, lw=1.2, ms=4.5, mfc=mfc, mec=col, mew=1.0, elinewidth=0.6, label=lab)
            a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=0.25, lw=0.6); a.tick_params(colors=mute, labelsize=7)
            a.set_title(f"{name.replace('--', chr(8211))}, {'no dephasing' if noise == 'T2=inf' else '$T_2=300$'}", fontsize=8, color=ink)
            for sp in ("top", "right"): a.spines[sp].set_visible(False)
            if i == 1: a.set_xlabel("time budget", fontsize=8, color=ink)
            if j == 0: a.set_ylabel("infidelity $1-F$", fontsize=8, color=ink)
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, frameon=False); fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("figs/fig3_budget.png", dpi=220); plt.close(fig)


def table_poly():
    """Degree table from the forty learning runs of stage 6: runs reaching the target, the degree that at least half of
    the runs reach (-- if fewer than half do), and the median subnormalisation of the reaching runs."""
    R6 = json.load(open("../results/stage6_results.json")); out = []
    def cell(st, t):
        d = st[f"deg F>={t}"]; k = sum(x is not None for x in d); n_ = len(d); srt = sorted(x if x is not None else 10**6 for x in d); half = srt[(n_ + 1) // 2 - 1]
        taus = [x for x in st[f"tau F>={t}"] if x is not None]
        return f"{k}/{n_}: " + (f"{half} ({np.median(taus):.2f})" if half < 10**6 else "--")
    dd = lambda x: "--" if x is None else str(x)
    for key, lab, kap in (NAMES[2], NAMES[1], NAMES[3], NAMES[0]):
        r = R6[key]; st = r["seed study"]; iv = r["interval odd LP (standard)"]
        out.append(f"{lab} & {kap} & {cell(st['spectral correction'], 0.999)} & {cell(st['spectral correction'], 0.9999)} & {cell(st['node LP'], 0.999)} & {cell(st['node LP'], 0.9999)} & {dd(iv['deg F>=0.999'])} / {dd(iv['deg F>=0.9999'])}\\\\")
    return "\n".join(out)


def eq(x, runs, e=e):
    """Table cell: x (the median) with the lower and upper quartile of the finite runs as sub- and superscript; no scripts
    where both print like the median.  e: the number format."""
    v = np.array([r for r in runs if r is not None and np.isfinite(r)], float); assert len(v) >= 5 and e(float(np.median(v))) == e(x), (x, np.median(v))
    lo, hi = (e(float(q)) for q in np.percentile(v, [25, 75]))
    return e(x) if lo == e(x) == hi else f"{e(x)}$_{{{lo}}}^{{{hi}}}$"


def table_grid():
    """Learning stages on the power-flow problems (stage 10): baselines at 1e4 shots, then the equal-time pair."""
    cases = ("case9", "case14", "case30"); M = {c: json.load(open(f"../results/stage10_methods.{c}.json")) for c in cases}
    C = {c: json.load(open(f"../results/stage10_compare.{c}.json"))["6-bit clock"] for c in cases}; keys = ("T2=inf", "T2=300.0"); out = []
    rows = (("Bayesian clock (mixture)", "Bayesian clock (mixture)"), ("QMEGS-type (mixture)", "QMEGS-type (mixture)"), ("MM-QCELS (mixture)", "MM-QCELS-type (mixture)"),
            ("QPE histogram, 6 bits", "QPE 6 bits"), ("QPE histogram, 8 bits", "QPE 8 bits"), ("QPE histogram, 10 bits", "QPE 10 bits"))
    for m, lab in rows: out.append(lab + " & " + " & ".join(eq(M[c][k][m]["1-F"], M[c][k][m]["runs"]["1-F"]) if m in M[c][k] else "--" for c in cases for k in keys) + "\\\\")
    out.append("\\midrule")
    pick = lambda r, pref: [x for x in r if x.startswith(pref)]
    q = lambda r: eq(r["1-F"], r["runs"]["1-F"])
    out.append("Projective HHL & " + " & ".join(q(C[c][k][pick(C[c][k], "projective (")[0]]) for c in cases for k in keys) + "\\\\")
    seq = lambda c, k: pick(C[c][k], "sequential (budget x1.0, delta=0.001")
    out.append("Recycled sequential HHL, register resolution & " + " & ".join(q(C[c][k][seq(c, k)[0]]) for c in cases for k in keys) + "\\\\")
    out.append("Recycled sequential HHL, finest resolution & " + " & ".join((q(C[c][k][seq(c, k)[-1]]) if len(seq(c, k)) > 1 else "same") for c in cases for k in keys) + "\\\\")
    return "\n".join(out)


SERIES = (("narma2", "NARMA2"), ("mackey-glass", "Mackey--Glass"), ("lorenz", "Lorenz"))


def table_vqls():
    """Variational solver (stage 12) on the systems of Tables I, XI and XII and on the 118-bus system: exact cost, then
    stochastic optimisation within 1e4 and 1e7 executions."""
    out = []
    for key, lab in [(k, lab) for k, lab, _ in NAMES] + [(f"case{n}", f"{n}-bus") for n in (9, 14, 30, 118)] + [(k, lab) for k, lab in SERIES]:
        v = json.load(open(f"../results/stage12_vqls.{key.replace(' ', '_')}.json")); x = v["exact cost"]; c = lambda b: eq(v[f"{b} executions"]["1-F"], v[f"{b} executions"]["runs"]["1-F"]) if "1-F" in v[f"{b} executions"] else "--"
        out.append(f"{lab} & {v['qubits']} & {int(x['terms'])} & {'$<$1e-9' if x['1-F'] < 1e-9 else eq(x['1-F'], x['runs']['1-F'])} & {e(x['executions'])} & {c(10000)} & {c(10000000)}\\\\")
    return "\n".join(out)


def table_series():
    """Learning stages on the ridge systems of stage 11 within ten times the time of 1e4 projective shots: median 1-F and
    the number of the ten runs whose forecasts differ from those of the exact ridge readout by less than 1e-3."""
    F = json.load(open("../results/stage11_forecast.json")); out = []; R = {}        # R: the infidelities of the single runs, from the files the forecast table was made of
    for c, _ in SERIES:
        for fn in sorted(glob.glob(f"../results/stage11_scale.{c}.x10*.json") + glob.glob(f"../results/stage11_baselines.{c}.x10*.json")):
            d = json.load(open(fn)); R.update({(c, f"{d.get('noise', 'T2=inf')}, {n}, x10 time"): r["runs"]["1-F"] for n, r in d.items() if isinstance(r, dict)})
    def cell(c, k, pick):
        r = [(n, v) for n, v in F[c].items() if isinstance(v, dict) and n.startswith(k + ", ") and n.endswith("x10 time") and pick(n)]
        if not r: return "--"
        assert len(r) == 1 and len(r[0][1]["dev_runs"]) == 10 and len(R[c, r[0][0]]) == 10; n, v = r[0]; return f"{eq(v['1-F'], R[c, n])} ({sum(d < 1e-3 for d in v['dev_runs'])})"
    rows = [(lab, lambda n, m=m: m in n) for m, lab in (("Bayesian clock", "Bayesian clock (mixture)"), ("QMEGS", "QMEGS-type (mixture)"), ("MM-QCELS", "MM-QCELS-type (mixture)"),
                                                        ("6 bits", "QPE 6 bits"), ("8 bits", "QPE 8 bits"), ("10 bits", "QPE 10 bits"), ("projective", "Projective HHL"))]
    rows += [(r"Recycled sequential HHL, $\sigma_{\rm id}=0.005$", lambda n: "sequential" in n and "sig_id=0.00500" in n), (r"Recycled sequential HHL, finest $\sigma_{\rm id}$", lambda n: "sequential" in n and "sig_id=0.00500" not in n)]
    for lab, pick in rows:
        if lab == "Projective HHL": out.append("\\midrule")
        out.append(lab + " & " + " & ".join(cell(c, k, pick) for c, _ in SERIES for k in ("T2=inf", "T2=300.0")) + "\\\\")
    return "\n".join(out)


def table_nisq():
    """Forecast NMSE of the baselines without phase estimation (stage 12) on the three series."""
    J = {(m, c): json.load(open(f"../results/stage12_{m}.{c}.json")) for m in ("vqls", "qrc", "qcl", "dqi") for c, _ in SERIES}; F = json.load(open("../results/stage11_forecast.json"))
    q = lambda r: eq(r["NMSE"], r["runs"]["NMSE"]) if "runs" in r else e(r["NMSE"])                 # quartiles where the entry is a median of runs
    rows = [("Ridge readout, exact", lambda c: e(F[c]["ridge at this kappa"])), ("Last known value", lambda c: e(F[c]["persistence"])),
            ("Variational solver, exact", lambda c: q(J["vqls", c]["exact cost"])), ("Variational solver, $10^5$", lambda c: q(J["vqls", c]["100000 executions"])),
            ("Variational solver, $10^7$", lambda c: q(J["vqls", c]["10000000 executions"])),
            ("Quantum reservoir, exact", lambda c: q(J["qrc", c]["exact expectation values"])), ("Quantum reservoir, $10^5$", lambda c: q(J["qrc", c]["100000 executions"])),
            ("Circuit learning, exact", lambda c: q(J["qcl", c]["exact expectation values"])), ("Circuit learning, $10^5$", lambda c: q(J["qcl", c]["100000 executions"])),
            ("DQI, $r=5$, $10^5$", lambda c: q(J["dqi", c]["sets of 5"]["100000 executions"])), ("DQI, $r=15$, $10^5$", lambda c: q(J["dqi", c]["sets of 15"]["100000 executions"]))]
    return "\n".join(lab + " & " + " & ".join(g(c) for c, _ in SERIES) + "\\\\" + ("\n\\midrule" if lab == "Last known value" else "") for lab, g in rows)


def table_system():
    """Sequential estimator with a system register of finite lifetime (stage 14): solution infidelity, then the relative
    error of the compliance in per cent, for the two jump models; the column without system noise is common to both."""
    d = json.load(open("../results/stage14_system.json")); out = []
    for key, head, fmt in (("1-F", "Solution infidelity $1-F$", e), ("eC", "Error of the compliance (\\%)", pct)):
        out.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{head}}}}}\\\\")
        for sy, lab in (("poisson8", "Poisson $N=8$"), ("case9", "9-bus"), ("lorenz", "Lorenz readout")):
            for m in ("redraw", "depolarising"):
                rows = [d[sy]["T_sys=inf"]] + [d[sy][m][f"T_sys={t}"] for t in ("3e+06", "1e+06", "300000", "100000", "30000", "10000")]
                out.append(f"{lab}, {'redrawing' if m == 'redraw' else 'depolarising'} & " + " & ".join(eq(r[key], r["runs"][key], fmt) for r in rows) + "\\\\")
        if key == "1-F": out.append("\\midrule")
    return "\n".join(out)


def splice(tex, label, body):
    i = tex.index(r"\label{" + label + "}"); t_end = tex.index(r"\end{tabular}", i); h = tex.index(r"\midrule", i, t_end) + len(r"\midrule")
    b = tex.find(r"\bottomrule", h, t_end); e = b if b >= 0 else t_end           # body = between the header rule and the bottom rule
    return tex[:h] + "\n" + body + "\n" + tex[e:]


if __name__ == "__main__":
    tex = open("main.tex").read()
    tex = splice(tex, "tab:learn", table1()); tex = splice(tex, "tab:clock", table2()); tex = splice(tex, "tab:qoi", table4())
    tex = splice(tex, "tab:matched", table_matched()); tex = splice(tex, "tab:matchedC", table_matchedC()); tex = splice(tex, "tab:qoiseq", table_qoiseq())
    tex = splice(tex, "tab:window", table_window()); tex = splice(tex, "tab:cost", table_cost()); tex = splice(tex, "tab:miscal", table_miscal()); tex = splice(tex, "tab:poly", table_poly()); tex = splice(tex, "tab:grid", table_grid()); tex = splice(tex, "tab:vqls", table_vqls()); tex = splice(tex, "tab:series", table_series()); tex = splice(tex, "tab:nisq", table_nisq()); tex = splice(tex, "tab:system", table_system())
    open("main.tex", "w").write(tex); fig2(); fig3(); print("tables, Fig. 2 and Fig. 3 regenerated")
