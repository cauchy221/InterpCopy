#!/usr/bin/env bash
#SBATCH --job-name=smoke-8b
#SBATCH --partition=suny
#SBATCH --account=suny
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00
#SBATCH --output=/mnt/lustre/suny/xliu1/InterpCopy/logs/smoke-8b-%j.out
#SBATCH --error=/mnt/lustre/suny/xliu1/InterpCopy/logs/smoke-8b-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# 5-step smoke test of the 8B LoRA pipeline.
# Verifies: weights load, tokenizer works, dataset parses, LoRA attaches,
# forward/backward runs, checkpoint writes.
#
# Submit:  sbatch sbatch/smoke_8b.sbatch
# Monitor: tail -f /mnt/lustre/suny/xliu1/InterpCopy/logs/smoke-8b-<jobid>.out

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

module load Python/3.10.15 CUDA/13.1
# shellcheck disable=SC1091
source ~/envs/tt/bin/activate

set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

export TOKENIZERS_PARALLELISM=false

tune run --nproc_per_node=1 lora_finetune_distributed \
  --config configs/8b_lora.yaml \
  max_steps_per_epoch=5 \
  epochs=1 \
  metric_logger._component_=torchtune.training.metric_logging.StdoutLogger
