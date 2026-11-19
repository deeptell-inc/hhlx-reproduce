#!/bin/sh
# Regenerates results/stage14_system.json (the sequential estimator with a system register of finite lifetime; needs the
# stage 9, 10 and 11 results it repeats).  One OS process per job, single-threaded BLAS, the long jobs first: on the
# Lorenz readout one repetition per process.
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
T="3e6 1e6 3e5 1e5 3e4 1e4"
: > results/stage14_run.log
{
  for k in 0 1 2 3 4 5 6 7 8 9; do echo "run lorenz redraw inf $k $((k + 1))"; done
  for m in redraw depolarising; do for t in $T; do for k in 0 1 2 3 4 5 6 7 8 9; do echo "run lorenz $m $t $k $((k + 1))"; done; done; done
  for s in case9 poisson8; do echo "run $s redraw inf"; for m in redraw depolarising; do for t in $T; do echo "run $s $m $t"; done; done; done
} | tr '\n' '\0' | xargs -0 -P 15 -n 1 sh -c 'python3 -W ignore stage14_system_noise.py $0 2>&1 | sed "s#$PWD/##g" >> results/stage14_run.log'
python3 -W ignore stage14_system_noise.py merge 2>&1 | sed "s#$PWD/##g" >> results/stage14_run.log
python3 -W ignore stage14_system_noise.py table > results/stage14_table.md
python3 -W ignore stage14_system_noise.py manifest >> results/stage14_run.log
tail -1 results/stage14_run.log
