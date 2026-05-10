#!/usr/bin/env bash
#SBATCH --job-name=smoke-8b
#SBATCH --partition=h200x8
#SBATCH --account=pn_tuch082825n
#SBATCH --exclude=h200x8-03
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/smoke-8b-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/smoke-8b-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# 5-step smoke test of the 8B LoRA pipeline.
# Verifies: weights load, tokenizer works, dataset parses, LoRA attaches,
# forward/backward runs, checkpoint writes.
#
# NvWulf note: the h200x8 partition only offers 8-GPU nodes, but the job
# requests 1 GPU — the remaining 7 GPUs are free for other jobs on that node.
# If this is too wasteful, swap to --partition=h200x4 / --gres=gpu:h200:1
# or a debug partition once 8B is known-good.
#
# Submit:  sbatch sbatch/smoke_8b.sbatch
# Monitor: tail -f /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/smoke-8b-<jobid>.out

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

python -m torchtune._cli.tune run --nproc_per_node=1 lora_finetune_distributed \
  --config configs/8b_lora.yaml \
  max_steps_per_epoch=5 \
  epochs=1 \
  metric_logger._component_=torchtune.training.metric_logging.StdoutLogger
