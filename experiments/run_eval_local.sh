#!/usr/bin/env bash
# Run an eval .cmds file locally, 3 commands at a time, at low priority
# (a laptop run with 18 busy processes froze the machine).
# usage: experiments/run_eval_local.sh [experiments/eval_final.cmds] [log-prefix]
cd "$(dirname "$0")/.."
export PYTHONPATH=. MUJOCO_GL=egl
mkdir -p runs/logs/eval
PY=${PY:-python}
i=0
while IFS= read -r line; do
  cmd=${line//python /$PY }
  ( nice -n 19 bash -c "$cmd" > runs/logs/eval/${2:-eval}_$i.log 2>&1 ) &
  i=$((i+1))
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
done < ${1:-experiments/eval_final.cmds}
wait
echo ALL_EVAL_DONE
