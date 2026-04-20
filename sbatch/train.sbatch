#!/usr/bin/env bash
#SBATCH --job-name=interpcopy
#SBATCH --partition=h200x8-long
#SBATCH --account=pn_tuch082825n
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
#  * SIGUSR1 trap (5 min before walltime) → graceful kill → SLURM --requeue
#    schedules another job, which then auto-resumes.

set -euo pipefail

# --- parse KEY=VAL args ---
CONFIG=""
for kv in "$@"; do
  case "$kv" in
    CONFIG=*) CONFIG="${kv#CONFIG=}" ;;
    *) echo "unknown arg: $kv (use CONFIG=path/to.yaml)" >&2; exit 2 ;;
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
trap 'echo ">>> SIGUSR1 received — killing training; SLURM will requeue"; kill -TERM $TRAIN_PID 2>/dev/null || true; wait $TRAIN_PID || true; exit 0' USR1

# --- launch (single-node torchrun via tune) ---
tune run --nproc_per_node="$GPUS_PER_NODE" lora_finetune_distributed \
  --config "$CONFIG" \
  $RESUME_FLAG &
TRAIN_PID=$!
wait $TRAIN_PID
