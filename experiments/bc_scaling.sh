#!/usr/bin/env bash
# BC data-scaling: 5/10/20/50 scripted demos (nested subsets, collection seed 100) x 3 training seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
for n in 5 10 20 50; do
  [ -f data/demos/expert_n$n.npz ] || python scripts/collect_demos.py --n $n --seed 100 --out data/demos/expert_n$n.npz
  for s in 0 1 2; do
    python scripts/train_bc.py --demos data/demos/expert_n$n.npz --out runs/main/bc_n${n}_s$s --seed $s | tail -1
  done
done
