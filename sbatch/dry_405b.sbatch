#!/usr/bin/env bash
#SBATCH --job-name=dry-405b
#SBATCH --partition=suny
#SBATCH --account=suny
#SBATCH --qos=suny
#SBATCH --nodes=2
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
# One-step multi-node dry-run of the 405B LoRA recipe.
# Only thing that exercises: multi-node FSDP rendezvous, sharded weight load,
# LoRA attach at 405B, bf16 forward+backward across nodes, checkpoint write
# to Lustre. If this passes, the real 3-epoch 405B run is very likely to pass.
#
# Submit AFTER the 8B smoke test passes:
#   sbatch --dependency=afterok:<smoke_jobid> sbatch/dry_405b.sbatch
#
# Or standalone:
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

NODES="${SLURM_JOB_NUM_NODES:-2}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
echo ">>> dry-run 405B: nodes=$NODES gpus_per_node=$GPUS_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

# Isolated output dir so we don't contaminate a real run's checkpoints.
DRY_OUT=/mnt/lustre/suny/xliu1/InterpCopy/checkpoints/llama3_1-405b-dry
mkdir -p "$DRY_OUT"

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
      --config configs/405b_lora.yaml \
      output_dir=$DRY_OUT \
      epochs=1 \
      max_steps_per_epoch=1 \
      fsdp_cpu_offload=False \
      metric_logger._component_=torchtune.training.metric_logging.StdoutLogger
  "

echo ">>> dry-run complete. Checkpoint at $DRY_OUT"
