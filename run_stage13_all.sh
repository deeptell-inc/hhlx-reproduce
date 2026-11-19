#!/bin/sh
# Regenerates results/stage13_sweep.json (error against budget on the readout problems of stage 11; needs the stage-11
# results).  One OS process per job, single-threaded BLAS, the long jobs first; the sequential estimator at three times
# the budget and above runs one repetition per process.
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
S="narma2 mackey-glass lorenz"
: > results/stage13_run.log
{
  for m in 30 10 3; do for c in $S; do for t in inf 300; do for q in sequential-fine sequential-coarse; do for k in 0 1 2 3 4 5 6 7 8 9; do
    echo "point $c $m $t $q $k $((k + 1))"; done; done; done; done; done
  for c in $S; do for e in 10000000 3000000 1000000 300000 100000 30000 10000; do echo "base $c qcl $e"; done; done
  for m in 30 10 3 1 0.3 0.1; do for c in $S; do for t in inf 300; do echo "point $c $m $t projective"; done; done; done
  for m in 1 0.3 0.1; do for c in $S; do for t in inf 300; do for q in sequential-fine sequential-coarse; do echo "point $c $m $t $q"; done; done; done; done
  for c in $S; do for b in vqls qrc dqi; do echo "base $c $b"; done; done
} | tr '\n' '\0' | xargs -0 -P 15 -n 1 sh -c 'python3 -W ignore stage13_budget_sweep.py $0 2>&1 | sed "s#$PWD/##g" >> results/stage13_run.log'
python3 -W ignore stage13_budget_sweep.py merge 2>&1 | sed "s#$PWD/##g" >> results/stage13_run.log
python3 -W ignore stage13_budget_sweep.py table > results/stage13_table.md
python3 -W ignore stage13_budget_sweep.py manifest >> results/stage13_run.log
tail -1 results/stage13_run.log
