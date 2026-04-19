#!/usr/bin/env bash
#SBATCH --job-name=dry-405b
#SBATCH --partition=suny
#SBATCH --account=suny
#SBATCH --qos=suny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre/suny/xliu1/InterpCopy/logs/dry-405b-%j.out
#SBATCH --error=/mnt/lustre/suny/xliu1/InterpCopy/logs/dry-405b-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# One-step single-node H100 dry-run of the 405B LoRA recipe.
# 8× H100 80GB (640GB GPU RAM) + FSDP CPU offload fits 405B bf16 (~810GB)
# by keeping base weights in CPU RAM and streaming shards to GPUs.
# Slower per step than H200 but H100 nodes are plentiful (17), so this
# backfills much faster than H200 (4 nodes) or multi-node.
# Exercises: single-node FSDP shard, CPU offload path, LoRA attach at 405B,
# bf16 fwd+bwd, checkpoint write. If this passes, the real run will pass.
#
# Submit:
#   sbatch sbatch/dry_405b.sbatch

set -euo pipefail

# SLURM copies sbatch scripts to a spool dir, so $BASH_SOURCE can't locate the
# repo. Prefer $SLURM_SUBMIT_DIR (set by SLURM to the sbatch-invocation dir),
# fall back to a hardcoded path.
REPO="${SLURM_SUBMIT_DIR:-/mnt/home/xliu1/InterpCopy}"
cd "$REPO"

module load Python/3.10.15 CUDA/13.1
# shellcheck disable=SC1091
source ~/envs/tt/bin/activate

set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

export TOKENIZERS_PARALLELISM=false
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_DEBUG=INFO

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
echo ">>> dry-run 405B: single-node, gpus=$GPUS_PER_NODE"

# Isolated output dir so we don't contaminate a real run's checkpoints.
DRY_OUT=/mnt/lustre/suny/xliu1/InterpCopy/checkpoints/llama3_1-405b-dry
mkdir -p "$DRY_OUT"

tune run --nproc_per_node="$GPUS_PER_NODE" lora_finetune_distributed \
  --config configs/405b_lora.yaml \
  output_dir="$DRY_OUT" \
  epochs=1 \
  max_steps_per_epoch=1 \
  fsdp_cpu_offload=True \
  metric_logger._component_=torchtune.training.metric_logging.StdoutLogger

echo ">>> dry-run complete. Checkpoint at $DRY_OUT"
