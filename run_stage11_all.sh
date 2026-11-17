#!/bin/sh
# Regenerates every stage-11 result (reservoir readouts for nonlinear time series). One OS process per job,
# single-threaded BLAS. The 100-repetition baselines run as ten processes of ten repetitions per series.
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for c in narma2 mackey-glass lorenz; do
  (python3 -W ignore stage11_time_series.py compare $c 2>&1 | sed "s#$PWD/##g" > results/stage11_compare.$c.log) &
  (python3 -W ignore stage11_time_series.py scale $c 10 2>&1 | sed "s#$PWD/##g" > results/stage11_scale.$c.log) &
  : > results/stage11_methods.$c.log
  for k in 0 10 20 30 40 50 60 70 80 90; do
    (python3 -W ignore stage11_time_series.py methods $c 100 $k $((k + 10)) 2>&1 | sed "s#$PWD/##g" >> results/stage11_methods.$c.log) &
  done
done
wait
for c in narma2 mackey-glass lorenz; do
  python3 -W ignore stage11_time_series.py methods-merge $c 100 2>&1 | sed "s#$PWD/##g" >> results/stage11_methods.$c.log
done
# Ten times the time budget: projective and sequential with dephasing, then the baselines for both noise settings,
# one repetition per process and at most ten at a time (a 6-bit QPE histogram at this budget holds about 4 GB).
for c in narma2 mackey-glass lorenz; do
  (python3 -W ignore stage11_time_series.py dephased $c 10 300 2>&1 | sed "s#$PWD/##g" > results/stage11_scale.$c.T2_300.log) &
done
for t in inf 300; do for c in narma2 mackey-glass lorenz; do : > results/stage11_baselines.$c.T2_$t.log; done; done
for t in inf 300; do for c in narma2 mackey-glass lorenz; do for k in 0 1 2 3 4 5 6 7 8 9; do echo $c $t $k $((k + 1)); done; done; done |
  xargs -P 10 -n 4 sh -c 'python3 -W ignore stage11_time_series.py baselines $0 10 $1 $2 $3 2>&1 | sed "s#$PWD/##g" >> results/stage11_baselines.$0.T2_$1.log'
wait
for t in inf 300; do for c in narma2 mackey-glass lorenz; do
  python3 -W ignore stage11_time_series.py baselines-merge $c 10 $t 2>&1 | sed "s#$PWD/##g" >> results/stage11_baselines.$c.T2_$t.log
done; done
python3 -W ignore stage11_time_series.py forecast > results/stage11_forecast.md
python3 -W ignore stage11_time_series.py manifest
echo "stage 11 jobs finished: $(ls results/stage11_*.json | wc -l) result files"
