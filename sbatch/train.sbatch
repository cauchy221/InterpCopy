#!/usr/bin/env bash
#SBATCH --job-name=interpcopy
#SBATCH --partition=suny
#SBATCH --account=suny
#SBATCH --qos=suny
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/mnt/lustre/suny/xliu1/InterpCopy/logs/%x-%j.out
#SBATCH --error=/mnt/lustre/suny/xliu1/InterpCopy/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# Unified launcher for all three rungs (8B / 70B / 405B).
#
# You specify --gres, --nodes, --ntasks-per-node on the sbatch command line,
# plus CONFIG=<path> at the end. Examples:
#
#   # 8B single node, 8 GPUs:
#   sbatch --nodes=1 --ntasks-per-node=1 \
#          --gres=gpu:nvidia_h100_80gb_hbm3:8 \
#          sbatch/train.sbatch CONFIG=configs/8b_lora.yaml
#
#   # 70B single node, 8 GPUs:
#   sbatch --nodes=1 --ntasks-per-node=1 \
#          --gres=gpu:nvidia_h100_80gb_hbm3:8 \
#          sbatch/train.sbatch CONFIG=configs/70b_lora.yaml
#
#   # 405B multi-node, 2 × 8 H100:
#   sbatch --nodes=2 --ntasks-per-node=1 \
#          --gres=gpu:nvidia_h100_80gb_hbm3:8 \
#          sbatch/train.sbatch CONFIG=configs/405b_lora.yaml
#
# Features:
#  * Auto-resume from the most recent epoch checkpoint in output_dir.
#  * SIGUSR1 trap (5 min before walltime) → graceful kill → SLURM --requeue
#    schedules another job, which then auto-resumes.
#  * Multi-node via torchrun rendezvous (c10d backend, master = node 0).

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

# SLURM copies sbatch scripts to a spool dir, so $BASH_SOURCE can't locate the
# repo. Prefer $SLURM_SUBMIT_DIR (set by SLURM to the sbatch-invocation dir),
# fall back to a hardcoded path.
REPO="${SLURM_SUBMIT_DIR:-/mnt/home/xliu1/InterpCopy}"
cd "$REPO"

# --- env ---
module load Python/3.10.15 CUDA/13.1
# shellcheck disable=SC1091
source ~/envs/tt/bin/activate

set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

# NCCL / IB tuning for Empire AI Alpha (ConnectX-7, NDR rail-aligned)
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# --- auto-resume: if any .safetensors or .pt lives in output_dir, resume ---
OUTPUT_DIR="$(grep -E '^output_dir:' "$CONFIG" | head -n1 | awk '{print $2}')"
RESUME_FLAG=""
if [[ -n "$OUTPUT_DIR" ]] && compgen -G "$OUTPUT_DIR/epoch_*" > /dev/null; then
  echo ">>> found prior checkpoints in $OUTPUT_DIR — resuming"
  RESUME_FLAG="resume_from_checkpoint=True"
else
  echo ">>> no prior checkpoints — starting fresh"
  mkdir -p "$OUTPUT_DIR"
fi

# --- rendezvous setup for multi-node ---
NODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
echo ">>> nodes=$NODES gpus_per_node=$GPUS_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

# --- SIGUSR1 handler: forward to the training process so torchtune can exit cleanly ---
trap 'echo ">>> SIGUSR1 received — killing training; SLURM will requeue"; kill -TERM $TRAIN_PID 2>/dev/null || true; wait $TRAIN_PID || true; exit 0' USR1

# --- launch ---
# one task per node; torchrun spawns N procs per node internally
srun --ntasks="$NODES" --ntasks-per-node=1 \
  bash -c "
    torchrun \
      --nnodes=$NODES \
      --nproc_per_node=$GPUS_PER_NODE \
      --node_rank=\$SLURM_NODEID \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      -m torchtune._cli.tune run lora_finetune_distributed \
      --config $CONFIG \
      $RESUME_FLAG
  " &
TRAIN_PID=$!
wait $TRAIN_PID
