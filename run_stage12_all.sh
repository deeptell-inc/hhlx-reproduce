#!/bin/sh
# Regenerates every stage-12 result (NISQ baselines and decoded quantum interferometry); needs the stage 9-11 results.
# One OS process per job, single-threaded BLAS, at most twelve at a time (~40 min; the circuit-learning runs are the slow ones).
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
{ for c in narma2 mackey-glass lorenz; do for m in qcl vqls qrc dqi; do echo "$m $c"; done; done
  for p in "P5 random 4x4" "P3 Poisson N=4" "P3 Poisson N=8" "P4 Toeplitz N=8" case9 case14 case30 case118; do echo "vqls $p"; done; } | tr '\n' '\0' |
  xargs -0 -P 12 -n 1 sh -c 'set -- $0; m=$1; shift; python3 -W ignore stage12_nisq_baselines.py $m "$*" 2>&1 | sed "s#$PWD/##g" > "results/stage12_$m.$(echo "$*" | tr " " "_").log"'
python3 -W ignore stage12_nisq_baselines.py table > results/stage12_table.md
python3 -W ignore stage12_nisq_baselines.py manifest
echo "stage 12 jobs finished: $(ls results/stage12_*.json | wc -l) result files"
