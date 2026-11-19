# Detection-limited eigenvalue learning for hybrid quantum linear system solvers: code and results

Code, result files and manuscript source for

> H. Wakaura and T. Tanimae, "Detection-Limited Eigenvalue Learning for Hybrid Quantum Linear System Solvers"
> (`paper/main.pdf`).

Every table of the manuscript is written from the result files by a script, and every number quoted in the text is
checked against them. This repository lets you (1) verify that in under a minute, (2) rebuild the tables, Figs. 2 and 3 and the
PDF from the stored results, and (3) regenerate the results themselves.

## Requirements

Python 3.10 or later with NumPy, SciPy, Matplotlib and pytest (`pip install -r requirements.txt`; tested with
Python 3.13.9, NumPy 2.2.6, SciPy 1.17.1, Matplotlib 3.10.6). Compiling the manuscript needs `pdflatex`, `bibtex` and the
`IEEEtran` class of TeX Live. No quantum SDK is used: all measurement statistics are sampled from the model stated in
the manuscript.

Run every command from the repository root unless a `cd` is shown; the stage scripts load each other by relative path.

## 1. Quick verification (under a minute)

```bash
python -m pytest tests/ -q
```

Five tests: three self-checks of the sequential estimator (exact record identity, posterior-identity rule, a two-level
learning run), the DC power-flow self-check with the time budgets of the stored runs, and `paper/check_numbers.py`.
The last one asserts the numbers quoted in the text of `paper/main.tex` against `results/` at the quoted number of
significant figures, checks that no run exceeds its time budget, and verifies the five manifests
(`results/stage9_manifest.json` to `stage13_manifest.json`): SHA-256 of
every script and result file, and that every result was written after the code that produced it.

## 2. Tables, Figs. 2 and 3 and the PDF from the stored results

```bash
cd paper && python make_tables.py && python check_numbers.py
```

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`make_tables.py` rewrites the table bodies of `main.tex`, `figs/fig2_detection.png` and `figs/fig3_budget.png`; on the stored results it leaves
`main.tex` unchanged.

## 3. Regenerating the results

Seeds are fixed. Jobs run as separate OS processes with single-threaded BLAS; the times are wall-clock times on 16 cores.

| Step | Command | Time | Writes |
|---|---|---|---|
| Stages 2 to 8 | `python stage2_noise_compare.py`, …, `python stage8b_floors_and_detection_runs.py` (one script each) | stage 4 about 40 min, stage 6 about 5 min, the others shorter | `results/stage2_results.json` … `results/stage8_floors.json` |
| Stage 9 | `sh run_stage9_all.sh` | about 1 h | `results/stage9_*.json`, `stage9_manifest.json` |
| Stage 10 | `sh run_stage10_all.sh` | about 1 h | `results/stage10_*.json`, `stage10_manifest.json` |
| Stage 11 | `sh run_stage11_all.sh` | about 5.5 h when run together with stage 12; up to 4 GB per process | `results/stage11_*.json`, `stage11_manifest.json` |
| Stage 12 | `sh run_stage12_all.sh` | about 40 min | `results/stage12_*.json`, `stage12_manifest.json` |
| Stage 13 | `sh run_stage13_all.sh` | about 7 h | `results/stage13_sweep.json`, `stage13_table.md`, `stage13_manifest.json` |

Order: stages 2 to 8, then 9, 10, 11, 12, 13 (each later stage imports the earlier scripts, and stages 11 to 13 read
earlier results), then step 2 above. After rerunning one of stages 4, 6 or 8b alone, run
`python stage9_sequential_qpe.py manifest`. The manifests are rewritten by the run scripts, so after a rerun
`check_numbers.py` compares the text with the new results. On the machine used for the manuscript, reruns of stages 11
and 12 reproduced the stored values exactly; on another platform or BLAS, differences in the last digits are possible.

Each stage script run without arguments performs its self-checks or its whole computation as stated in its docstring;
single jobs are listed there (for example `python stage9_sequential_qpe.py compare 2`).

## Where each item of the manuscript comes from

| Item | Script | Result files |
|---|---|---|
| Fig. 1 (infidelity against dephasing time) | `stage2_noise_compare.py` | `results/stage2_results.json`, `results/stage2_fidelity.png` (the figure) |
| Table I (learning stages, 100 repetitions) | `stage4_mmqcels_clockopt.py` | `results/stage4_results.json` |
| Fig. 2 (detection probability) | `stage8_detection.py`, `stage8b_floors_and_detection_runs.py`, `stage9_sequential_qpe.py detect` | `results/stage8_detection.json`, `stage8_floors.json`, `stage9_detect.part*.json` |
| Tables II, III (cost, miscalibration) | `stage9_sequential_qpe.py cost` | `results/stage9_cost.json` |
| Tables IV, V (same-time comparison) | `stage9_sequential_qpe.py compare` | `results/stage9_compare.part*.json` |
| Tables VI, VII (quantities of interest) | `stage7_observable_first.py`, `stage7b_qks_baseline.py`, `stage9_sequential_qpe.py qoi` | `results/stage7_results.json`, `stage7b_qks.json`, `stage9_qoi.part*.json` |
| Tables VIII, IX (clock back end, window) | `stage9_sequential_qpe.py backend` | `results/stage9_backend.json` |
| Table X (polynomial degree) | `stage6_spectral_correction.py` | `results/stage6_results.json` |
| Table XI (DC power flow) | `stage10_power_grid.py` | `results/stage10_compare.*.json`, `stage10_methods.*.json` |
| Table XII (reservoir readout, forecasts) | `stage11_time_series.py` | `results/stage11_forecast.json` (from the other `stage11_*.json`) |
| Fig. 3 (infidelity against budget), budget paragraphs of Secs. VIII and IX | `stage13_budget_sweep.py` | `results/stage13_sweep.json` |
| Table XIII (variational solver) | `stage12_nisq_baselines.py vqls` | `results/stage12_vqls.*.json` |
| Table XIV (forecasting baselines) | `stage12_nisq_baselines.py qrc`, `qcl`, `dqi` | `results/stage12_qrc.*.json`, `stage12_qcl.*.json`, `stage12_dqi.*.json`, `stage11_forecast.json` |

The remaining result files (`stage3`, `stage5`, `stage5b`, the other `stage9_*.json` and `stage10_*.json`) back numbers
quoted in the text; `paper/check_numbers.py` names the file for each of them.

## Layout

| Path | Content |
|---|---|
| `pilot_bayes_clock.py` | Measurement model of the single-ancilla Hadamard test |
| `stage1_checks.py` … `stage8b_floors_and_detection_runs.py` | Test problems and fidelity metric (stage 2), mixture estimators and baselines (3, 4), polynomial inversion (5, 5b, 6), observables (7, 7b), detection test (8, 8b) |
| `stage9_sequential_qpe.py` | Sequential estimator (Recycled sequential HHL) and its experiments |
| `stage10_power_grid.py`, `data/grid_cases.json` | DC power-flow problems of the 9-, 14-, 30- and 118-bus test systems |
| `stage11_time_series.py` | Ridge readout of an echo state network for NARMA2, Mackey–Glass and Lorenz series |
| `stage12_nisq_baselines.py` | Baselines without phase estimation: variational linear solver, quantum reservoir computing, quantum circuit learning, decoded quantum interferometry |
| `stage13_budget_sweep.py` | The forecasting test and its baselines against the budget |
| `run_stage9_all.sh` … `run_stage13_all.sh` | Full regeneration of one stage, with its manifest |
| `results/` | Result files (JSON), three summary tables (`stage11_forecast.md`, `stage12_table.md`, `stage13_table.md`) and the manifests |
| `paper/` | `main.tex`, `main.pdf`, `refs.bib`, `figs/`, `make_tables.py`, `check_numbers.py` |
| `tests/` | The quick checks of step 1 |

In the code and the result files "projective" is the estimator of Projective HHL (`bc2_learn` in
`stage2_noise_compare.py`) and "sequential" that of Recycled sequential HHL (`learn` in `stage9_sequential_qpe.py`).

## Licence

The code (`*.py`, `*.sh`, `tests/`, the scripts in `paper/`) is under the MIT licence (`LICENSE`). The manuscript
(`paper/main.tex`, `paper/main.pdf`, `paper/figs/`) and the result files (`results/`) are under the Creative Commons
Attribution 4.0 International licence (https://creativecommons.org/licenses/by/4.0/). `data/grid_cases.json` is not
covered by either: it keeps the terms of its sources (see Notes). To cite this work, see `CITATION.cff`.

## Notes

- On macOS with the Accelerate BLAS, NumPy prints "encountered in matmul" RuntimeWarnings; they are spurious. The run
  scripts silence them with `-W ignore`.
- The run scripts start up to 36 processes at once. With fewer cores they take proportionally longer; nothing else
  changes.
- `data/grid_cases.json` is derived from the MATPOWER 8.0 case files; its `source` field names the origin of each case.
  These data keep the terms of their sources.
- The scope of the claims and the known limitations are stated in the limitations paragraphs of the manuscript.
