#!/bin/bash
# Run the commands of an experiments/*.cmds file in parallel on one GPU slice
# (PPO updates on the GPU, MuJoCo envs on the CPUs; 2 runs x 8 envs fit 16 cores).
#
#   sbatch -A <account> -p <gpu partition> -o runs/logs/%x_%j.out cluster/rl_train.sbatch experiments/main_j1.cmds
#
# Submit from the repository root. Each non-empty, non-# line of the .cmds file is one command.
#SBATCH --job-name=rebot-sim-rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
CMDS=$1
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the repository root
[ -n "${RL_ACTIVATE:-}" ] && source "$RL_ACTIVATE"   # optional: script activating environment.yml
export PYTHONPATH=$PWD MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p runs/logs
echo "host=$(hostname) job=$SLURM_JOB_ID cmds=$CMDS cpus=$(nproc)"
nvidia-smi --query-gpu=name --format=csv,noheader
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  echo "[start $i] $line"
  bash -c "$line" > "runs/logs/${SLURM_JOB_ID}_$i.log" 2>&1 &
  i=$((i + 1))
done < "$CMDS"
wait
echo "all done"
