#!/usr/bin/env bash
#SBATCH --job-name=dry-405b
#SBATCH --partition=h200x8
#SBATCH --account=pn_tuch082825n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=62
#SBATCH --mem=1000G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/dry-405b-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/dry-405b-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# One-step single-node H200 dry-run of the 405B LoRA recipe.
# 8× H200 = 1128 GB GPU RAM > 810 GB bf16 weights, so CPU offload is OFF
# (faster per step, representative of the real training run on this cluster).
# Exercises: single-node FSDP shard, LoRA attach at 405B, bf16 fwd+bwd,
# checkpoint write. If this passes, the real run will pass.
#
# Submit:
#   sbatch sbatch/dry_405b.sbatch

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy}"
cd "$REPO"

module load miniconda/3 cuda12.8/toolkit/12.8.1
# shellcheck disable=SC1091
source ~/envs/tt/bin/activate

set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

export TOKENIZERS_PARALLELISM=false
# NvWulf has Mellanox adapters; safe defaults for single-node (no-op for NCCL
# since it'll prefer NVLink on an H200x8 node).
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_DEBUG=INFO

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
echo ">>> dry-run 405B: single-node H200, gpus=$GPUS_PER_NODE"

# Isolated output dir so we don't contaminate a real run's checkpoints.
DRY_OUT=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/checkpoints/llama3_1-405b-dry
mkdir -p "$DRY_OUT"

tune run --nproc_per_node="$GPUS_PER_NODE" lora_finetune_distributed \
  --config configs/405b_lora.yaml \
  output_dir="$DRY_OUT" \
  epochs=1 \
  max_steps_per_epoch=1 \
  fsdp_cpu_offload=False \
  metric_logger._component_=torchtune.training.metric_logging.StdoutLogger

echo ">>> dry-run complete. Checkpoint at $DRY_OUT"
