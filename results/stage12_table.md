Linear systems: median 1-F over ten runs.  Projective and sequential: stored rows at 10^4 projective shots (T2 = inf / 300).  VQLS: no T2 (see the module text).

| system | qubits | Pauli terms | projective | sequential | VQLS, exact cost | its (circuit, term) pairs | VQLS, 1e+04 executions | VQLS, 1e+05 executions | VQLS, 1e+07 executions |
|---|---|---|---|---|---|---|---|---|---|
| P5 random 4x4 | 2 | 47 | 7.4e-10 / 2.1e-06 | 4.3e-07 / 4.5e-07 | 0 | 1.3e+04 | 0.69 | - | 0.023 |
| P3 Poisson N=4 | 2 | 9 | 8.1e-08 / 3.9e-05 | 2.3e-07 / 5.8e-07 | 1.1e-16 | 1.9e+03 | 0.81 | - | 0.12 |
| P3 Poisson N=8 | 3 | 21 | 2.8e-06 / 0.076 | 8.8e-08 / 5.8e-06 | 0 | 1.6e+04 | 0.84 | - | 0.86 |
| P4 Toeplitz N=8 | 3 | 33 | 4.4e-06 / 0.00039 | 1.2e-07 / 3e-06 | 4.4e-16 | 1.3e+04 | 0.89 | - | 0.064 |
| case9 | 3 | 63 | 0.19 / 0.18 | 4.3e-07 / 0.11 | 0 | 7.2e+04 | 0.95 | - | 0.45 |
| case14 | 4 | 223 | 0.012 / 0.012 | 3.3e-07 / 0.012 | 1e-14 | 2.6e+06 | 0.96 | - | 0.97 |
| case30 | 5 | 807 | 0.67 / 0.77 | 0.066 / 0.64 | 2.7e-10 | 1.2e+08 | 0.96 | - | 0.97 |
| case118 | 7 | 8127 | - / - | - / - | 0.58 | 4.8e+09 | not runnable | - | 0.99 |
| narma2 | 4 | 256 | 0.43 / 0.24 | 0.12 / 0.43 | 1.9e-10 | 3.4e+06 | 0.96 | 0.99 | 0.94 |
| mackey-glass | 4 | 256 | 0.055 / 0.053 | 0.055 / 0.055 | 4.6e-11 | 2e+06 | 0.96 | 0.98 | 0.97 |
| lorenz | 4 | 256 | 0.042 / 0.28 | 0.016 / 0.37 | 2.1e-11 | 1.2e+06 | 0.97 | 0.93 | 0.95 |

Forecasts: median over ten runs; `difference` is the mean squared difference from the forecasts of the exact ridge readout of the echo state network, in units of the target variance.

| series | method | setting | executions | 1-F | forecast NMSE | difference |
|---|---|---|---|---|---|---|
| narma2 | echo state network, exact ridge readout | kappa <= 100 | - | 0 | 0.12 | 0 |
| narma2 | last known value |  | - | - | 0.84 | - |
| narma2 | projective HHL | T2=inf, 10^4 projective shots | - | 0.43 | 0.15 | 0.005 |
| narma2 | sequential HHL | T2=inf, 10^4 projective shots, 0.0050 | - | 0.12 | 0.13 | 0.004 |
| narma2 | projective HHL | T2=inf, ten times that time | - | 0.43 | 0.15 | 0.005 |
| narma2 | sequential HHL | T2=inf, ten times that time, 0.0003 | - | 0.0027 | 0.12 | 4e-05 |
| narma2 | projective HHL | T2=300.0, 10^4 projective shots | - | 0.24 | 0.14 | 0.0024 |
| narma2 | sequential HHL | T2=300.0, 10^4 projective shots, 0.0050 | - | 0.43 | 0.15 | 0.005 |
| narma2 | projective HHL | T2=300.0, ten times that time | - | 0.16 | 0.14 | 0.0016 |
| narma2 | sequential HHL | T2=300.0, ten times that time, 0.0020 | - | 0.0034 | 0.12 | 4.7e-05 |
| narma2 | VQLS | exact cost | 3.4e+06 | 1.9e-10 | 0.12 | 5.1e-13 |
| narma2 | VQLS | 10000 executions | 9.7e+03 | 0.96 | 1 | 0.83 |
| narma2 | VQLS | 100000 executions | 1e+05 | 0.99 | 0.97 | 0.81 |
| narma2 | VQLS | 10000000 executions | 1e+07 | 0.94 | 0.4 | 0.25 |
| narma2 | QRC | exact expectation values | - | - | 0.21 | 0.13 |
| narma2 | QRC | 10000 executions | - | - | not runnable | - |
| narma2 | QRC | 100000 executions | 9.6e+04 | - | 0.64 | 0.52 |
| narma2 | QCL | exact expectation values | 9.3e+06 | - | 0.14 | 0.2 |
| narma2 | QCL | 10000 executions | 1e+04 | - | 1 | 0.83 |
| narma2 | QCL | 100000 executions | 1e+05 | - | 0.91 | 0.76 |
| narma2 | DQI, sets of 5 | expectation of one sample (satisfied 0.30, random 0.16, optimum 0.63) | 1 | - | 4 | - |
| narma2 | DQI, sets of 5 | best of the samples (satisfied 0.57) | 1e+04 | - | 2.4 | - |
| narma2 | DQI, sets of 5 | best of the samples (satisfied 0.63) | 1e+05 | - | 1.2 | - |
| narma2 | optimal polynomial, sets of 5 | exhaustive search | - | - | 1.2 | - |
| narma2 | all constraints met, sets of 5 |  | - | - | 0.33 | - |
| narma2 | DQI, sets of 15 | expectation of one sample (satisfied 0.64, random 0.48, optimum 1.00) | 1 | - | 3.6 | - |
| narma2 | DQI, sets of 15 | best of the samples (satisfied 0.90) | 1e+04 | - | 1.5 | - |
| narma2 | DQI, sets of 15 | best of the samples (satisfied 1.00) | 1e+05 | - | 1.6 | - |
| narma2 | optimal polynomial, sets of 15 | exhaustive search | - | - | 1.6 | - |
| narma2 | all constraints met, sets of 15 |  | - | - | 0.39 | - |
| mackey-glass | echo state network, exact ridge readout | kappa <= 100 | - | 0 | 0.23 | 0 |
| mackey-glass | last known value |  | - | - | 0.67 | - |
| mackey-glass | projective HHL | T2=inf, 10^4 projective shots | - | 0.055 | 0.24 | 7.6e-05 |
| mackey-glass | sequential HHL | T2=inf, 10^4 projective shots, 0.0050 | - | 0.055 | 0.24 | 7.6e-05 |
| mackey-glass | projective HHL | T2=inf, ten times that time | - | 0.05 | 0.24 | 7.6e-05 |
| mackey-glass | sequential HHL | T2=inf, ten times that time, 0.0003 | - | 0.001 | 0.23 | 1.8e-05 |
| mackey-glass | projective HHL | T2=300.0, 10^4 projective shots | - | 0.053 | 0.24 | 0.00012 |
| mackey-glass | sequential HHL | T2=300.0, 10^4 projective shots, 0.0050 | - | 0.055 | 0.24 | 7.9e-05 |
| mackey-glass | projective HHL | T2=300.0, ten times that time | - | 0.049 | 0.24 | 8e-05 |
| mackey-glass | sequential HHL | T2=300.0, ten times that time, 0.0050 | - | 0.002 | 0.23 | 1.5e-05 |
| mackey-glass | VQLS | exact cost | 2e+06 | 4.6e-11 | 0.23 | 1.4e-14 |
| mackey-glass | VQLS | 10000 executions | 9.7e+03 | 0.96 | 0.74 | 0.45 |
| mackey-glass | VQLS | 100000 executions | 1e+05 | 0.98 | 0.69 | 0.4 |
| mackey-glass | VQLS | 10000000 executions | 1e+07 | 0.97 | 0.57 | 0.27 |
| mackey-glass | QRC | exact expectation values | - | - | 0.23 | 0.04 |
| mackey-glass | QRC | 10000 executions | - | - | not runnable | - |
| mackey-glass | QRC | 100000 executions | 9.6e+04 | - | 0.56 | 0.31 |
| mackey-glass | QCL | exact expectation values | 3.3e+07 | - | 0.097 | 0.13 |
| mackey-glass | QCL | 10000 executions | 1e+04 | - | 0.87 | 0.6 |
| mackey-glass | QCL | 100000 executions | 1e+05 | - | 0.8 | 0.54 |
| mackey-glass | DQI, sets of 5 | expectation of one sample (satisfied 0.30, random 0.16, optimum 0.63) | 1 | - | 2.1 | - |
| mackey-glass | DQI, sets of 5 | best of the samples (satisfied 0.57) | 1e+04 | - | 1.6 | - |
| mackey-glass | DQI, sets of 5 | best of the samples (satisfied 0.63) | 1e+05 | - | 1.6 | - |
| mackey-glass | optimal polynomial, sets of 5 | exhaustive search | - | - | 1.6 | - |
| mackey-glass | all constraints met, sets of 5 |  | - | - | 0.63 | - |
| mackey-glass | DQI, sets of 15 | expectation of one sample (satisfied 0.64, random 0.48, optimum 0.97) | 1 | - | 2 | - |
| mackey-glass | DQI, sets of 15 | best of the samples (satisfied 0.90) | 1e+04 | - | 1.2 | - |
| mackey-glass | DQI, sets of 15 | best of the samples (satisfied 0.97) | 1e+05 | - | 0.66 | - |
| mackey-glass | optimal polynomial, sets of 15 | exhaustive search | - | - | 0.66 | - |
| mackey-glass | all constraints met, sets of 15 |  | - | - | 0.62 | - |
| lorenz | echo state network, exact ridge readout | kappa <= 100 | - | 0 | 0.13 | 0 |
| lorenz | last known value |  | - | - | 0.26 | - |
| lorenz | projective HHL | T2=inf, 10^4 projective shots | - | 0.042 | 0.14 | 0.0023 |
| lorenz | sequential HHL | T2=inf, 10^4 projective shots, 0.0050 | - | 0.016 | 0.14 | 8.3e-05 |
| lorenz | projective HHL | T2=inf, ten times that time | - | 0.022 | 0.13 | 0.0012 |
| lorenz | sequential HHL | T2=inf, ten times that time, 0.0003 | - | 0.00015 | 0.13 | 7.2e-07 |
| lorenz | projective HHL | T2=300.0, 10^4 projective shots | - | 0.28 | 0.18 | 0.01 |
| lorenz | sequential HHL | T2=300.0, 10^4 projective shots, 0.0050 | - | 0.37 | 0.19 | 0.0076 |
| lorenz | projective HHL | T2=300.0, ten times that time | - | 0.097 | 0.16 | 0.0035 |
| lorenz | sequential HHL | T2=300.0, ten times that time, 0.0050 | - | 0.015 | 0.13 | 8.4e-05 |
| lorenz | VQLS | exact cost | 1.2e+06 | 2.1e-11 | 0.13 | 2.3e-14 |
| lorenz | VQLS | 10000 executions | 9.7e+03 | 0.97 | 0.32 | 0.13 |
| lorenz | VQLS | 100000 executions | 1e+05 | 0.93 | 0.44 | 0.22 |
| lorenz | VQLS | 10000000 executions | 1e+07 | 0.95 | 0.35 | 0.14 |
| lorenz | QRC | exact expectation values | - | - | 0.24 | 0.031 |
| lorenz | QRC | 10000 executions | - | - | not runnable | - |
| lorenz | QRC | 100000 executions | 9.6e+04 | - | 0.43 | 0.2 |
| lorenz | QCL | exact expectation values | 3.3e+07 | - | 0.0084 | 0.13 |
| lorenz | QCL | 10000 executions | 1e+04 | - | 1 | 0.8 |
| lorenz | QCL | 100000 executions | 1e+05 | - | 0.62 | 0.42 |
| lorenz | DQI, sets of 5 | expectation of one sample (satisfied 0.30, random 0.16, optimum 0.63) | 1 | - | 2.4 | - |
| lorenz | DQI, sets of 5 | best of the samples (satisfied 0.57) | 1e+04 | - | 1.4 | - |
| lorenz | DQI, sets of 5 | best of the samples (satisfied 0.63) | 1e+05 | - | 0.84 | - |
| lorenz | optimal polynomial, sets of 5 | exhaustive search | - | - | 0.84 | - |
| lorenz | all constraints met, sets of 5 |  | - | - | 0.23 | - |
| lorenz | DQI, sets of 15 | expectation of one sample (satisfied 0.64, random 0.48, optimum 1.00) | 1 | - | 2.2 | - |
| lorenz | DQI, sets of 15 | best of the samples (satisfied 0.90) | 1e+04 | - | 0.84 | - |
| lorenz | DQI, sets of 15 | best of the samples (satisfied 1.00) | 1e+05 | - | 0.27 | - |
| lorenz | optimal polynomial, sets of 15 | exhaustive search | - | - | 0.26 | - |
| lorenz | all constraints met, sets of 15 |  | - | - | 0.21 | - |
