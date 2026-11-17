#!/bin/sh
# Regenerates every stage-9 result with the current code (one OS process per job), then the pair statistics and the manifest.
# Usage: sh run_stage9_all.sh   (from the hhlx directory; about one hour on 16 cores)
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for k in 0 1 2 3; do (python3 stage9_sequential_qpe.py compare $k 2>&1 | sed "s#$PWD/##g" > results/stage9_compare.$k.log) & (python3 stage9_sequential_qpe.py qoi $k 2>&1 | sed "s#$PWD/##g" > results/stage9_qoi.$k.log) & done
for k in 0 1 2 3 4 5 6 7; do (python3 stage9_sequential_qpe.py detect $k 2>&1 | sed "s#$PWD/##g" > results/stage9_detect.$k.log) & done
for j in lifetime backend cost crb regress accept; do (python3 stage9_sequential_qpe.py $j 2>&1 | sed "s#$PWD/##g" > results/stage9_$j.log) & done
wait
python3 stage9_sequential_qpe.py pairs 2>&1 | sed "s#$PWD/##g" > results/stage9_pairs.log
python3 stage9_sequential_qpe.py manifest 2>&1 | sed "s#$PWD/##g" > results/stage9_manifest.log
echo all-done > results/stage9_all_done.txt
