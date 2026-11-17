| series | noise | learning stage | runs | time | 1-F | forecast NMSE | difference from the ridge forecast | runs with NMSE > 1 |
|---|---|---|---|---|---|---|---|---|
| narma2 | T2=inf | Bayesian clock (mixture) | 100 | 2.3e+06 | 0.43 | 0.15 | 0.0087 | 0 |
| narma2 | T2=inf | QMEGS-type (mixture) | 100 | 1.6e+07 | 0.25 | 0.14 | 0.0027 | 2 |
| narma2 | T2=inf | MM-QCELS (mixture) | 98 | 2.3e+06 | 0.43 | 0.15 | 0.0076 | 3 |
| narma2 | T2=inf | QPE histogram, 6 bits | 96 | 3.2e+06 | 0.17 | 0.14 | 0.0016 | 4 |
| narma2 | T2=inf | QPE histogram, 8 bits | 100 | 1.3e+07 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | QPE histogram, 10 bits | 100 | 5.1e+07 | 0.2 | 0.15 | 0.005 | 1 |
| narma2 | T2=inf | standard HHL, 6-bit clock, noiseless | 1 | - | 0.019 | 0.12 | 0.00021 | 0 |
| narma2 | T2=inf | standard HHL, 8-bit clock, noiseless | 1 | - | 0.0016 | 0.12 | 1.5e-05 | 0 |
| narma2 | T2=inf | standard HHL, 10-bit clock, noiseless | 1 | - | 0.00021 | 0.12 | 2.1e-06 | 0 |
| narma2 | T2=inf | projective (n_c=6, 1e4 shots) | 10 | 2.2e+07 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | sequential (budget x1.0, delta=0.001, sig_id=0.0050) | 10 | 2.2e+07 | 0.12 | 0.13 | 0.004 | 0 |
| narma2 | T2=inf | sequential (budget x1.0, delta=0.0001, sig_id=0.0050) | 10 | 2.2e+07 | 0.1 | 0.13 | 0.0041 | 0 |
| narma2 | T2=inf | sequential (budget x0.1, delta=0.001, sig_id=0.0050) | 10 | 2.2e+06 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | sequential (budget x0.1, delta=0.0001, sig_id=0.0050) | 10 | 2.2e+06 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | sequential (budget x1.0, delta=0.001, sig_id=0.0003) | 5 | 2.2e+07 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | sequential (budget x1.0, delta=0.0001, sig_id=0.0003) | 5 | 2.2e+07 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | sequential (budget x0.1, delta=0.001, sig_id=0.0003) | 5 | 2.2e+06 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | sequential (budget x0.1, delta=0.0001, sig_id=0.0003) | 5 | 2.2e+06 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | sequential (budget x1.0, no target, 1 refinement per copy, sig_id=0.0050) | 10 | 2.2e+07 | 0.38 | 0.15 | 0.012 | 0 |
| narma2 | T2=300.0 | Bayesian clock (mixture) | 100 | 1.4e+06 | 0.43 | 0.15 | 0.0079 | 0 |
| narma2 | T2=300.0 | QMEGS-type (mixture) | 100 | 2.4e+06 | 0.25 | 0.14 | 0.003 | 1 |
| narma2 | T2=300.0 | MM-QCELS (mixture) | 100 | 4.1e+05 | 0.43 | 0.15 | 0.0077 | 2 |
| narma2 | T2=300.0 | QPE histogram, 6 bits | 100 | 3.2e+06 | 0.25 | 0.14 | 0.0022 | 0 |
| narma2 | T2=300.0 | standard HHL, 6-bit clock, noiseless | 1 | - | 0.019 | 0.12 | 0.00021 | 0 |
| narma2 | T2=300.0 | standard HHL, 8-bit clock, noiseless | 1 | - | 0.0016 | 0.12 | 1.5e-05 | 0 |
| narma2 | T2=300.0 | standard HHL, 10-bit clock, noiseless | 1 | - | 0.00021 | 0.12 | 2.1e-06 | 0 |
| narma2 | T2=300.0 | projective (n_c=5, 1e4 shots) | 10 | 4.2e+06 | 0.24 | 0.14 | 0.0024 | 0 |
| narma2 | T2=300.0 | sequential (budget x1.0, delta=0.001, sig_id=0.0050) | 10 | 4.2e+06 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=300.0 | sequential (budget x1.0, delta=0.0001, sig_id=0.0050) | 10 | 4.2e+06 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=300.0 | sequential (budget x0.1, delta=0.001, sig_id=0.0050) | 10 | 4.2e+05 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=300.0 | sequential (budget x0.1, delta=0.0001, sig_id=0.0050) | 10 | 4.2e+05 | 0.45 | 0.16 | 0.0094 | 0 |
| narma2 | T2=300.0 | sequential (budget x1.0, delta=0.001, sig_id=0.0020) | 5 | 4.2e+06 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=300.0 | sequential (budget x1.0, delta=0.0001, sig_id=0.0020) | 5 | 4.2e+06 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=300.0 | sequential (budget x0.1, delta=0.001, sig_id=0.0020) | 5 | 4.2e+05 | 0.45 | 0.16 | 0.0094 | 0 |
| narma2 | T2=300.0 | sequential (budget x0.1, delta=0.0001, sig_id=0.0020) | 5 | 4.2e+05 | 0.45 | 0.16 | 0.0094 | 0 |
| narma2 | T2=300.0 | sequential (budget x1.0, no target, 1 refinement per copy, sig_id=0.0050) | 10 | 4.2e+06 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | projective, 10-bit clock | 10 | 7.1e+07 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | sequential, 10-bit clock | 10 | 7.1e+07 | 0.17 | 0.15 | 0.005 | 0 |
| narma2 | T2=300.0 | Bayesian clock (mixture), x10 time | 10 | 2.2e+08 | 0.45 | 0.16 | 0.009 | 0 |
| narma2 | T2=300.0 | QMEGS-type (mixture), x10 time | 10 | 2.1e+08 | 0.31 | 0.14 | 0.0029 | 0 |
| narma2 | T2=300.0 | MM-QCELS (mixture), x10 time | 10 | 2.1e+08 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=300.0 | QPE histogram, 6 bits, x10 time | 10 | 2.2e+08 | 0.25 | 0.14 | 0.0022 | 0 |
| narma2 | T2=inf | Bayesian clock (mixture), x10 time | 10 | 1.9e+08 | 0.44 | 0.15 | 0.0075 | 0 |
| narma2 | T2=inf | QMEGS-type (mixture), x10 time | 10 | 2.1e+08 | 0.31 | 0.14 | 0.0034 | 0 |
| narma2 | T2=inf | MM-QCELS (mixture), x10 time | 10 | 2.1e+08 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | QPE histogram, 6 bits, x10 time | 10 | 2.2e+08 | 0.068 | 0.13 | 0.001 | 0 |
| narma2 | T2=inf | QPE histogram, 8 bits, x10 time | 10 | 2.2e+08 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | QPE histogram, 10 bits, x10 time | 10 | 2.2e+08 | 0.19 | 0.17 | 0.046 | 0 |
| narma2 | T2=300.0 | projective, 5-bit clock, x10 time | 10 | 2.1e+08 | 0.16 | 0.14 | 0.0016 | 0 |
| narma2 | T2=300.0 | sequential, sig_id=0.00500, x10 time | 10 | 2.2e+08 | 0.1 | 0.12 | 0.0074 | 0 |
| narma2 | T2=300.0 | sequential, sig_id=0.00200, x10 time | 10 | 2.2e+08 | 0.0034 | 0.12 | 4.7e-05 | 0 |
| narma2 | T2=inf | projective, 6-bit clock, x10 time | 10 | 2.3e+08 | 0.43 | 0.15 | 0.005 | 0 |
| narma2 | T2=inf | sequential, sig_id=0.00500, x10 time | 10 | 2.2e+08 | 0.1 | 0.13 | 0.0021 | 0 |
| narma2 | T2=inf | sequential, sig_id=0.00030, x10 time | 10 | 2.2e+08 | 0.0027 | 0.12 | 4e-05 | 0 |
| mackey-glass | T2=inf | Bayesian clock (mixture) | 100 | 2.3e+06 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=inf | QMEGS-type (mixture) | 100 | 1.6e+07 | 0.072 | 0.24 | 0.0047 | 2 |
| mackey-glass | T2=inf | MM-QCELS (mixture) | 100 | 2.3e+06 | 0.26 | 0.28 | 0.03 | 5 |
| mackey-glass | T2=inf | QPE histogram, 6 bits | 96 | 3.2e+06 | 0.057 | 0.24 | 0.00026 | 0 |
| mackey-glass | T2=inf | QPE histogram, 8 bits | 100 | 1.3e+07 | 0.055 | 0.24 | 8.7e-05 | 0 |
| mackey-glass | T2=inf | QPE histogram, 10 bits | 100 | 5.1e+07 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | standard HHL, 6-bit clock, noiseless | 1 | - | 0.011 | 0.24 | 0.00068 | 0 |
| mackey-glass | T2=inf | standard HHL, 8-bit clock, noiseless | 1 | - | 0.0016 | 0.24 | 4.4e-05 | 0 |
| mackey-glass | T2=inf | standard HHL, 10-bit clock, noiseless | 1 | - | 0.00011 | 0.23 | 3.9e-06 | 0 |
| mackey-glass | T2=inf | projective (n_c=6, 1e4 shots) | 10 | 2.2e+07 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | sequential (budget x1.0, delta=0.001, sig_id=0.0050) | 10 | 2.2e+07 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | sequential (budget x1.0, delta=0.0001, sig_id=0.0050) | 10 | 2.2e+07 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | sequential (budget x0.1, delta=0.001, sig_id=0.0050) | 10 | 2.2e+06 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | sequential (budget x0.1, delta=0.0001, sig_id=0.0050) | 10 | 2.2e+06 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | sequential (budget x1.0, delta=0.001, sig_id=0.0003) | 5 | 2.2e+07 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | sequential (budget x1.0, delta=0.0001, sig_id=0.0003) | 5 | 2.2e+07 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | sequential (budget x0.1, delta=0.001, sig_id=0.0003) | 5 | 2.2e+06 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=inf | sequential (budget x0.1, delta=0.0001, sig_id=0.0003) | 5 | 2.2e+06 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=inf | sequential (budget x1.0, no target, 1 refinement per copy, sig_id=0.0050) | 10 | 2.2e+07 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=300.0 | Bayesian clock (mixture) | 100 | 1.4e+06 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=300.0 | QMEGS-type (mixture) | 100 | 2.4e+06 | 0.074 | 0.24 | 0.0066 | 4 |
| mackey-glass | T2=300.0 | MM-QCELS (mixture) | 100 | 4.1e+05 | 0.28 | 0.28 | 0.034 | 8 |
| mackey-glass | T2=300.0 | QPE histogram, 6 bits | 99 | 3.2e+06 | 0.058 | 0.24 | 0.00035 | 0 |
| mackey-glass | T2=300.0 | standard HHL, 6-bit clock, noiseless | 1 | - | 0.011 | 0.24 | 0.00068 | 0 |
| mackey-glass | T2=300.0 | standard HHL, 8-bit clock, noiseless | 1 | - | 0.0016 | 0.24 | 4.4e-05 | 0 |
| mackey-glass | T2=300.0 | standard HHL, 10-bit clock, noiseless | 1 | - | 0.00011 | 0.23 | 3.9e-06 | 0 |
| mackey-glass | T2=300.0 | projective (n_c=5, 1e4 shots) | 10 | 4.3e+06 | 0.053 | 0.24 | 0.00012 | 0 |
| mackey-glass | T2=300.0 | sequential (budget x1.0, delta=0.001, sig_id=0.0050) | 10 | 4.3e+06 | 0.055 | 0.24 | 7.9e-05 | 0 |
| mackey-glass | T2=300.0 | sequential (budget x1.0, delta=0.0001, sig_id=0.0050) | 10 | 4.3e+06 | 0.055 | 0.24 | 9.4e-05 | 0 |
| mackey-glass | T2=300.0 | sequential (budget x0.1, delta=0.001, sig_id=0.0050) | 10 | 4.3e+05 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=300.0 | sequential (budget x0.1, delta=0.0001, sig_id=0.0050) | 10 | 4.3e+05 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=300.0 | sequential (budget x1.0, delta=0.001, sig_id=0.0020) | 5 | 4.3e+06 | 0.055 | 0.24 | 9.1e-05 | 0 |
| mackey-glass | T2=300.0 | sequential (budget x1.0, delta=0.0001, sig_id=0.0020) | 5 | 4.3e+06 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=300.0 | sequential (budget x0.1, delta=0.001, sig_id=0.0020) | 5 | 4.3e+05 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=300.0 | sequential (budget x0.1, delta=0.0001, sig_id=0.0020) | 5 | 4.3e+05 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=300.0 | sequential (budget x1.0, no target, 1 refinement per copy, sig_id=0.0050) | 10 | 4.3e+06 | 0.056 | 0.24 | 0.00017 | 0 |
| mackey-glass | T2=inf | projective, 10-bit clock | 10 | 7.1e+07 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | sequential, 10-bit clock | 10 | 7.1e+07 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=300.0 | Bayesian clock (mixture), x10 time | 10 | 2.2e+08 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=300.0 | QMEGS-type (mixture), x10 time | 10 | 2.1e+08 | 0.055 | 0.24 | 0.00043 | 1 |
| mackey-glass | T2=300.0 | MM-QCELS (mixture), x10 time | 10 | 2.2e+08 | 0.056 | 0.24 | 0.00013 | 0 |
| mackey-glass | T2=300.0 | QPE histogram, 6 bits, x10 time | 10 | 2.2e+08 | 0.058 | 0.24 | 0.00034 | 0 |
| mackey-glass | T2=inf | Bayesian clock (mixture), x10 time | 10 | 1.9e+08 | 0.28 | 0.28 | 0.034 | 0 |
| mackey-glass | T2=inf | QMEGS-type (mixture), x10 time | 10 | 2.1e+08 | 0.068 | 0.24 | 0.0049 | 0 |
| mackey-glass | T2=inf | MM-QCELS (mixture), x10 time | 10 | 2.2e+08 | 0.059 | 0.24 | 0.00068 | 0 |
| mackey-glass | T2=inf | QPE histogram, 6 bits, x10 time | 10 | 2.2e+08 | 0.39 | 0.24 | 0.013 | 1 |
| mackey-glass | T2=inf | QPE histogram, 8 bits, x10 time | 10 | 2.2e+08 | 0.052 | 0.24 | 7.9e-05 | 0 |
| mackey-glass | T2=inf | QPE histogram, 10 bits, x10 time | 10 | 2.2e+08 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=300.0 | projective, 5-bit clock, x10 time | 10 | 2.2e+08 | 0.049 | 0.24 | 8e-05 | 1 |
| mackey-glass | T2=300.0 | sequential, sig_id=0.00500, x10 time | 10 | 2.2e+08 | 0.002 | 0.23 | 1.5e-05 | 0 |
| mackey-glass | T2=300.0 | sequential, sig_id=0.00200, x10 time | 10 | 2.2e+08 | 0.055 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | projective, 6-bit clock, x10 time | 10 | 2.3e+08 | 0.05 | 0.24 | 7.6e-05 | 0 |
| mackey-glass | T2=inf | sequential, sig_id=0.00500, x10 time | 10 | 2.2e+08 | 0.001 | 0.23 | 1.9e-05 | 0 |
| mackey-glass | T2=inf | sequential, sig_id=0.00030, x10 time | 10 | 2.2e+08 | 0.001 | 0.23 | 1.8e-05 | 0 |
| lorenz | T2=inf | Bayesian clock (mixture) | 100 | 2.3e+06 | 0.78 | 0.24 | 0.043 | 0 |
| lorenz | T2=inf | QMEGS-type (mixture) | 100 | 1.6e+07 | 0.33 | 0.19 | 0.011 | 1 |
| lorenz | T2=inf | MM-QCELS (mixture) | 99 | 2.3e+06 | 0.75 | 0.24 | 0.044 | 12 |
| lorenz | T2=inf | QPE histogram, 6 bits | 99 | 3.2e+06 | 0.35 | 0.19 | 0.0075 | 0 |
| lorenz | T2=inf | QPE histogram, 8 bits | 100 | 1.3e+07 | 0.016 | 0.13 | 0.00021 | 0 |
| lorenz | T2=inf | QPE histogram, 10 bits | 100 | 5.1e+07 | 0.016 | 0.14 | 8.6e-05 | 0 |
| lorenz | T2=inf | standard HHL, 6-bit clock, noiseless | 1 | - | 0.016 | 0.14 | 9.8e-05 | 0 |
| lorenz | T2=inf | standard HHL, 8-bit clock, noiseless | 1 | - | 0.00066 | 0.13 | 1.2e-05 | 0 |
| lorenz | T2=inf | standard HHL, 10-bit clock, noiseless | 1 | - | 0.00019 | 0.13 | 2.4e-06 | 0 |
| lorenz | T2=inf | projective (n_c=6, 1e4 shots) | 10 | 2.2e+07 | 0.042 | 0.14 | 0.0023 | 0 |
| lorenz | T2=inf | sequential (budget x1.0, delta=0.001, sig_id=0.0050) | 10 | 2.2e+07 | 0.016 | 0.14 | 8.3e-05 | 0 |
| lorenz | T2=inf | sequential (budget x1.0, delta=0.0001, sig_id=0.0050) | 10 | 2.2e+07 | 0.016 | 0.14 | 8.2e-05 | 0 |
| lorenz | T2=inf | sequential (budget x0.1, delta=0.001, sig_id=0.0050) | 10 | 2.2e+06 | 0.37 | 0.19 | 0.0077 | 0 |
| lorenz | T2=inf | sequential (budget x0.1, delta=0.0001, sig_id=0.0050) | 10 | 2.2e+06 | 0.37 | 0.19 | 0.0077 | 0 |
| lorenz | T2=inf | sequential (budget x1.0, delta=0.001, sig_id=0.0003) | 5 | 2.2e+07 | 0.016 | 0.14 | 8.4e-05 | 0 |
| lorenz | T2=inf | sequential (budget x1.0, delta=0.0001, sig_id=0.0003) | 5 | 2.2e+07 | 0.37 | 0.19 | 0.0077 | 0 |
| lorenz | T2=inf | sequential (budget x0.1, delta=0.001, sig_id=0.0003) | 5 | 2.2e+06 | 0.79 | 0.25 | 0.044 | 0 |
| lorenz | T2=inf | sequential (budget x0.1, delta=0.0001, sig_id=0.0003) | 5 | 2.2e+06 | 0.79 | 0.25 | 0.044 | 0 |
| lorenz | T2=inf | sequential (budget x1.0, no target, 1 refinement per copy, sig_id=0.0050) | 10 | 2.2e+07 | 0.019 | 0.13 | 0.00029 | 0 |
| lorenz | T2=300.0 | Bayesian clock (mixture) | 100 | 1.4e+06 | 0.77 | 0.24 | 0.042 | 1 |
| lorenz | T2=300.0 | QMEGS-type (mixture) | 99 | 2.4e+06 | 0.34 | 0.19 | 0.014 | 3 |
| lorenz | T2=300.0 | MM-QCELS (mixture) | 100 | 4.1e+05 | 0.74 | 0.24 | 0.044 | 10 |
| lorenz | T2=300.0 | QPE histogram, 6 bits | 99 | 3.2e+06 | 0.36 | 0.19 | 0.0075 | 1 |
| lorenz | T2=300.0 | standard HHL, 6-bit clock, noiseless | 1 | - | 0.016 | 0.14 | 9.8e-05 | 0 |
| lorenz | T2=300.0 | standard HHL, 8-bit clock, noiseless | 1 | - | 0.00066 | 0.13 | 1.2e-05 | 0 |
| lorenz | T2=300.0 | standard HHL, 10-bit clock, noiseless | 1 | - | 0.00019 | 0.13 | 2.4e-06 | 0 |
| lorenz | T2=300.0 | projective (n_c=5, 1e4 shots) | 10 | 4.3e+06 | 0.28 | 0.18 | 0.01 | 0 |
| lorenz | T2=300.0 | sequential (budget x1.0, delta=0.001, sig_id=0.0050) | 10 | 4.3e+06 | 0.37 | 0.19 | 0.0076 | 0 |
| lorenz | T2=300.0 | sequential (budget x1.0, delta=0.0001, sig_id=0.0050) | 10 | 4.3e+06 | 0.71 | 0.23 | 0.032 | 0 |
| lorenz | T2=300.0 | sequential (budget x0.1, delta=0.001, sig_id=0.0050) | 10 | 4.3e+05 | 0.79 | 0.25 | 0.044 | 0 |
| lorenz | T2=300.0 | sequential (budget x0.1, delta=0.0001, sig_id=0.0050) | 10 | 4.3e+05 | 0.79 | 0.25 | 0.044 | 0 |
| lorenz | T2=300.0 | sequential (budget x1.0, delta=0.001, sig_id=0.0020) | 5 | 4.3e+06 | 0.62 | 0.22 | 0.02 | 0 |
| lorenz | T2=300.0 | sequential (budget x1.0, delta=0.0001, sig_id=0.0020) | 5 | 4.3e+06 | 0.79 | 0.25 | 0.044 | 0 |
| lorenz | T2=300.0 | sequential (budget x0.1, delta=0.001, sig_id=0.0020) | 5 | 4.3e+05 | 0.79 | 0.25 | 0.044 | 0 |
| lorenz | T2=300.0 | sequential (budget x0.1, delta=0.0001, sig_id=0.0020) | 5 | 4.3e+05 | 0.79 | 0.25 | 0.044 | 0 |
| lorenz | T2=300.0 | sequential (budget x1.0, no target, 1 refinement per copy, sig_id=0.0050) | 10 | 4.3e+06 | 0.37 | 0.19 | 0.0077 | 0 |
| lorenz | T2=inf | projective, 10-bit clock | 10 | 7.1e+07 | 0.016 | 0.14 | 8e-05 | 0 |
| lorenz | T2=inf | sequential, 10-bit clock | 10 | 7.1e+07 | 0.016 | 0.14 | 8.1e-05 | 0 |
| lorenz | T2=300.0 | Bayesian clock (mixture), x10 time | 10 | 2.2e+08 | 0.78 | 0.24 | 0.043 | 0 |
| lorenz | T2=300.0 | QMEGS-type (mixture), x10 time | 10 | 2.2e+08 | 0.3 | 0.18 | 0.0089 | 0 |
| lorenz | T2=300.0 | MM-QCELS (mixture), x10 time | 10 | 2.2e+08 | 0.37 | 0.19 | 0.0081 | 1 |
| lorenz | T2=300.0 | QPE histogram, 6 bits, x10 time | 10 | 2.2e+08 | 0.36 | 0.19 | 0.0075 | 0 |
| lorenz | T2=inf | Bayesian clock (mixture), x10 time | 10 | 1.9e+08 | 0.78 | 0.24 | 0.043 | 0 |
| lorenz | T2=inf | QMEGS-type (mixture), x10 time | 10 | 2.2e+08 | 0.18 | 0.18 | 0.0093 | 0 |
| lorenz | T2=inf | MM-QCELS (mixture), x10 time | 10 | 2.2e+08 | 0.61 | 0.23 | 0.031 | 2 |
| lorenz | T2=inf | QPE histogram, 6 bits, x10 time | 10 | 2.2e+08 | 0.13 | 0.072 | 0.044 | 0 |
| lorenz | T2=inf | QPE histogram, 8 bits, x10 time | 10 | 2.2e+08 | 0.015 | 0.13 | 0.00013 | 0 |
| lorenz | T2=inf | QPE histogram, 10 bits, x10 time | 10 | 2.2e+08 | 0.016 | 0.14 | 8.5e-05 | 0 |
| lorenz | T2=300.0 | projective, 5-bit clock, x10 time | 10 | 2.2e+08 | 0.097 | 0.16 | 0.0035 | 0 |
| lorenz | T2=300.0 | sequential, sig_id=0.00500, x10 time | 10 | 2.2e+08 | 0.015 | 0.13 | 8.4e-05 | 0 |
| lorenz | T2=300.0 | sequential, sig_id=0.00200, x10 time | 10 | 2.2e+08 | 0.015 | 0.13 | 8.4e-05 | 0 |
| lorenz | T2=inf | projective, 6-bit clock, x10 time | 10 | 2.3e+08 | 0.022 | 0.13 | 0.0012 | 0 |
| lorenz | T2=inf | sequential, sig_id=0.00500, x10 time | 10 | 2.2e+08 | 0.016 | 0.14 | 8.2e-05 | 0 |
| lorenz | T2=inf | sequential, sig_id=0.00030, x10 time | 10 | 2.2e+08 | 0.00015 | 0.13 | 7.2e-07 | 0 |
