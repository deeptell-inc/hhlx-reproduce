#!/bin/sh
# Regenerates every stage-10 result (power-grid problems). One OS process per job, single-threaded BLAS; under 1 h on 16 cores.
# The 100-repetition baselines ("methods") run as ten processes of ten repetitions per system and are joined afterwards.
# Warnings in the logs are printed with paths relative to the repository root.
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for c in case9 case14 case30; do
  (python3 stage10_power_grid.py compare $c 2>&1 | sed "s#$PWD/##g" > results/stage10_compare.$c.log) &
  (python3 stage10_power_grid.py qoi $c 2>&1 | sed "s#$PWD/##g" > results/stage10_qoi.$c.log) &
  : > results/stage10_methods.$c.log
  for k in 0 10 20 30 40 50 60 70 80 90; do
    (python3 stage10_power_grid.py methods $c 100 $k $((k + 10)) 2>&1 | sed "s#$PWD/##g" >> results/stage10_methods.$c.log) &
  done
done
for c in case9 case14 case30; do for j in observables backend poly; do
  (python3 stage10_power_grid.py $j $c 2>&1 | sed "s#$PWD/##g" > results/stage10_$j.$c.log) &
done; done
for c in case30 case118; do for m in 1 4 16; do
  (python3 stage10_power_grid.py scale $c $m 2>&1 | sed "s#$PWD/##g" > results/stage10_scale.$c.x$m.log) &
done; done
wait
for c in case9 case14 case30; do
  python3 stage10_power_grid.py methods-merge $c 100 2>&1 | sed "s#$PWD/##g" >> results/stage10_methods.$c.log
done
python3 stage10_power_grid.py manifest 2>&1 | sed "s#$PWD/##g" > results/stage10_manifest.log
echo "stage 10 jobs finished: $(ls results/stage10_*.json | wc -l) result files"
