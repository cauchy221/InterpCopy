#!/usr/bin/env bash
#SBATCH --job-name=interpcopy
#SBATCH --partition=h200x8-long
#SBATCH --account=pn_tuch082825n
#SBATCH --exclude=h200x8-03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=62
#SBATCH --mem=1000G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/%x-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# Unified launcher for the 8B / 70B / 405B rungs on NvWulf.
# h200x8 partition caps MaxNodes=1, so all runs are single-node, 8× H200.
# For 8B you can override --gres=gpu:h200:1 and --cpus-per-task=8 on the
# sbatch command line if you want to free the rest of the node.
#
# Usage:
#   sbatch sbatch/train.sbatch CONFIG=configs/8b_lora.yaml
#   sbatch sbatch/train.sbatch CONFIG=configs/70b_lora.yaml
#   sbatch sbatch/train.sbatch CONFIG=configs/405b_lora.yaml
#
# Features:
#  * Auto-resume from the most recent epoch checkpoint in output_dir.
#  * SIGUSR1 trap (5 min before walltime) → graceful kill of the trainer so
#    whatever was saved at the last completed epoch is preserved. NOTE:
#    --requeue alone does NOT fire on walltime expiry; if 3 epochs don't
#    fit in 48 h, you'll need to manually resubmit (auto-resume will pick
#    up from the latest epoch_*).

set -euo pipefail

# --- parse KEY=VAL args ---
CONFIG=""
DCP_CKPT_DIR_ARG=""
for kv in "$@"; do
  case "$kv" in
    CONFIG=*) CONFIG="${kv#CONFIG=}" ;;
    DCP_CKPT_DIR=*) DCP_CKPT_DIR_ARG="${kv#DCP_CKPT_DIR=}" ;;
    *) echo "unknown arg: $kv (use CONFIG=path/to.yaml or DCP_CKPT_DIR=path/to/dcp)" >&2; exit 2 ;;
  esac
done
if [[ -z "$CONFIG" ]]; then
  echo "CONFIG=<path> is required" >&2
  exit 2
fi

REPO="${SLURM_SUBMIT_DIR:-/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy}"
cd "$REPO"

# --- env ---
module load miniconda/3 cuda12.8/toolkit/12.8.1
# shellcheck disable=SC1091
source ~/envs/tt/bin/activate

set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

# NCCL safe defaults. On a single H200x8 node, NCCL will use NVLink/NVSwitch
# and ignore IB settings anyway, but leaving them in case of cross-node later.
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# Bump init_process_group timeout via the sitecustomize hook in scripts/.
# Loading the 405B HF checkpoint from /lustre on rank 0 takes >30 min,
# tripping the default NCCL watchdog on the other ranks (job 26819 hung
# at SeqNum=1 BROADCAST after 30 min). 2 h gives the load plenty of room.
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"
export TT_DIST_TIMEOUT_MIN=120

# --- auto-resume: if any epoch_* subdir lives in output_dir, resume ---
OUTPUT_DIR="$(grep -E '^output_dir:' "$CONFIG" | head -n1 | awk '{print $2}')"
RESUME_FLAG=""
if [[ -n "$OUTPUT_DIR" ]] && compgen -G "$OUTPUT_DIR/epoch_*" > /dev/null; then
  echo ">>> found prior checkpoints in $OUTPUT_DIR — resuming"
  RESUME_FLAG="resume_from_checkpoint=True"
else
  echo ">>> no prior checkpoints — starting fresh"
  mkdir -p "$OUTPUT_DIR"
fi

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
echo ">>> single-node H200, gpus=$GPUS_PER_NODE, config=$CONFIG"

# --- SIGUSR1 handler: forward to the training process so torchtune can exit cleanly ---
trap 'echo ">>> SIGUSR1 received — gracefully killing training; latest epoch_* dir is preserved. Manual resubmit auto-resumes."; kill -TERM $TRAIN_PID 2>/dev/null || true; wait $TRAIN_PID || true; exit 0' USR1

# If WANDB_API_KEY is unset, fall back to StdoutLogger so the recipe doesn't
# crash at metric_logger setup. (configs/*_lora.yaml default to WandBLogger.)
LOGGER_OVERRIDE=()
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo ">>> WANDB_API_KEY not set — overriding metric_logger to StdoutLogger"
  LOGGER_OVERRIDE=(metric_logger._component_=torchtune.training.metric_logging.StdoutLogger)
fi

# --- launch (single-node torchrun via tune) ---
# Invoke torchtune's CLI as a Python module: ray (a vLLM dep) ships its own
# `tune` console script that shadows torchtune's entry point in this venv.
# `python -m torchtune._cli.tune` is unambiguous and stable.
#
# If DCP_CKPT_DIR= was passed, we route to the DCP-aware recipe wrapper, which
# loads the base model in parallel from a pre-converted PyTorch DCP directory
# (avoids the rank-0 read + per-tensor NCCL broadcast that hangs at 405B scale).
if [[ -n "$DCP_CKPT_DIR_ARG" ]]; then
  if [[ ! -d "$DCP_CKPT_DIR_ARG" ]] || [[ ! -f "$DCP_CKPT_DIR_ARG/.metadata" ]]; then
    echo ">>> ERROR: DCP_CKPT_DIR=$DCP_CKPT_DIR_ARG missing or has no .metadata" >&2
    exit 1
  fi
  export DCP_CKPT_DIR="$DCP_CKPT_DIR_ARG"
  RECIPE_PATH="recipes_dcp/lora_finetune_distributed_dcp.py"
  echo ">>> using DCP base-model load: $DCP_CKPT_DIR"
else
  RECIPE_PATH="lora_finetune_distributed"
  echo ">>> using upstream HF base-model load (slow at 405B; consider DCP_CKPT_DIR=)"
fi

python -m torchtune._cli.tune run --nproc_per_node="$GPUS_PER_NODE" "$RECIPE_PATH" \
  --config "$CONFIG" \
  "${LOGGER_OVERRIDE[@]}" \
  $RESUME_FLAG &
TRAIN_PID=$!
wait $TRAIN_PID
