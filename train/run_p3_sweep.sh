#!/bin/bash
# P3 — topology x capacity ablation sweep.
#
# 25 independent cells = 5 (kind,depth) x 5 folds, run through a worker pool.
# Each cell writes its own json the moment it finishes, so the sweep can be cut
# off at any point and reported from whatever completed (hard-cutoff friendly).
# Cells are emitted in PRIORITY order, so the pool finishes them roughly in that
# order: M_ring L1 -> M_ring L2 -> M1 L3 -> M_ring L3 -> M2 L3.
#
# Each worker is pinned to a single thread; the simulator is effectively
# single-threaded, so N workers ~ N cores. 8 workers leaves ~4 cores free for
# the submission pipeline, which runs in parallel and is NOT gated on this.
#
# Re-running is safe: run_cell.py skips a cell whose json already exists.
set -u
cd "$(dirname "$0")"
DATA="${1:?usage: run_p3_sweep.sh /path/to/OneraDataset}"
WORKERS="${2:-8}"

for cell in "mring 1" "mring 2" "m1 3" "mring 3" "m2 3"; do
  set -- $cell
  for f in 0 1 2 3 4; do echo "$1 $2 $f"; done
done > /tmp/p3_jobs.txt

echo "P3 sweep: $(wc -l < /tmp/p3_jobs.txt) cells, $WORKERS workers"
cat /tmp/p3_jobs.txt | xargs -P "$WORKERS" -L 1 bash -c \
  'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
   python3 -u run_cell.py --data_dir "'"$DATA"'" --kind $0 --depth $1 --fold $2'
echo "P3 sweep finished"
