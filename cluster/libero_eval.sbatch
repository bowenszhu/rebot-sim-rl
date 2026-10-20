#!/bin/bash
# LIBERO evaluation of a public LeRobot checkpoint (used for SmolVLA and pi0.5, see
# results/libero/RUNS.md). One GPU; resubmits itself on preemption.
#
#   export PROJ=/scratch/me/libero LEROBOT_ACTIVATE=/scratch/me/env/activate.sh
#   sbatch -A <account> -p <gpu partition> -o "$PROJ/logs/%x_%j.out" cluster/libero_eval.sbatch \
#       POLICY SUITE N_EP N_ACT TAG [EXTRA_ARGS] [TASK_IDS]
#   e.g. ... lerobot/pi05_libero_finetuned libero_spatial 10 10 pi05_spatial --policy.compile_model=false
#        ... HuggingFaceVLA/smolvla_libero libero_10 10 "" smolvla_long_s0 "" "[0,1]"
#
# Environment used here: lerobot 0.4.4 (torch 2.7.1+cu128) with the pi0.5 transformers patch,
# plus `pip install --no-deps hf-libero==0.1.4 robomimic==0.2.0` and its remaining deps
# (robosuite==1.4.0, mujoco==3.8.1, bddl==1.0.1, ...; robomimic's egl_probe does not build).
# pi0.5 needs the gated google/paligemma-3b-pt-224 tokenizer in $HF_HOME/hub (accept its licence
# on Hugging Face, or copy an existing cache): the evaluation itself runs with HF_HUB_OFFLINE=1.
# Do not pass --export=ALL,VAR=... (held jobs on one cluster); export variables before sbatch.
#SBATCH --job-name=libero-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
: "${PROJ:?export PROJ=<scratch dir for caches and outputs>}"
: "${LEROBOT_ACTIVATE:?export LEROBOT_ACTIVATE=<script that activates lerobot 0.4.4 + hf-libero>}"
source "$LEROBOT_ACTIVATE"
export HF_HOME="$PROJ/hf_home" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH="$PROJ/libero_config"
POLICY=$1 SUITE=$2 N_EP=$3 N_ACT=$4 TAG=$5 EXTRA_ARGS=${6:-} TASK_IDS=${7:-}
# keep LIBERO's asset cache (~/.cache/libero) on scratch instead of a small home directory
export HOME="$PROJ/home"; mkdir -p "$HOME"
N_EP=${N_EP:-10}
N_ACT=${N_ACT:-}
OUT="$PROJ/runs/$TAG"
mkdir -p "$OUT" "$LIBERO_CONFIG_PATH"
# libero prompts on stdin when its config is missing; write the default config instead
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
python - <<'PY'
import importlib.util, os, yaml
root = os.path.join(os.path.dirname(importlib.util.find_spec("libero").origin), "libero")
cfg = {"benchmark_root": root, "bddl_files": f"{root}/bddl_files", "init_states": f"{root}/init_files",
       "datasets": f"{root}/../datasets", "assets": f"{root}/assets"}
open(os.path.join(os.environ["LIBERO_CONFIG_PATH"], "config.yaml"), "w").write(yaml.dump(cfg))
print("libero config:", cfg, "assets exist:", os.path.isdir(cfg["assets"]))
PY
fi
echo "host=$(hostname) job=$SLURM_JOB_ID policy=$POLICY suite=$SUITE n_ep=$N_EP n_act=$N_ACT task_ids=$TASK_IDS extra=${EXTRA_ARGS:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import torch, transformers, mujoco, robosuite, lerobot
assert torch.cuda.is_available(), "CUDA not available on this node"
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__,
      "mujoco", mujoco.__version__, "robosuite", robosuite.__version__)
PY
ARGS=(--policy.path="$POLICY" --env.type=libero --env.task="$SUITE"
      --eval.batch_size=1 --eval.n_episodes="$N_EP" --output_dir="$OUT"
      --env.max_parallel_tasks=1 --seed=1000)
[ -n "$N_ACT" ] && ARGS+=(--policy.n_action_steps="$N_ACT")
[ -n "$TASK_IDS" ] && ARGS+=(--env.task_ids="$TASK_IDS")
# extra policy/env flags, e.g. EXTRA_ARGS="--policy.compile_model=false"
[ -n "${EXTRA_ARGS:-}" ] && read -r -a EXTRA <<< "$EXTRA_ARGS" && ARGS+=("${EXTRA[@]}")
# Download the policy online, then evaluate offline: pi0.5's PaliGemma tokenizer is a gated
# repo and is only available from the local cache (linked into $HF_HOME/hub, not downloaded here).
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$POLICY'))"
# SmolVLA loads its VLM backbone by repo id at init; fetch it too before going offline
case "$POLICY" in *smolvla*) python -c "from huggingface_hub import snapshot_download; print(snapshot_download('HuggingFaceTB/SmolVLM2-500M-Instruct'))" ;; esac
python -c "from libero.libero import get_assets_path; print('libero assets', get_assets_path())"
export HF_HUB_OFFLINE=1
start=$(date +%s)
python -m lerobot.scripts.lerobot_eval "${ARGS[@]}"
echo "elapsed_s=$(( $(date +%s) - start ))"
