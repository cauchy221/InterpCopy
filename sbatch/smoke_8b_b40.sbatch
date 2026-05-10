#!/usr/bin/env bash
#SBATCH --job-name=smoke-8b
#SBATCH --partition=b40x4-long
#SBATCH --account=pn_tuch082825n
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=45:00
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/smoke-8b-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/smoke-8b-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# 5-step smoke test on a single Blackwell GPU (b40x4 partition has more free
# capacity than h200x8 right now). This only validates the code path — not
# H200 NCCL — but for 8B scale that's all we need.
#
# Submit:  sbatch sbatch/smoke_8b_b40.sbatch

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

# Invoke torchtune's CLI as a Python module: ray (vLLM dep) ships its own
# `tune` script that shadows torchtune's entry point in this venv.
python -m torchtune._cli.tune run --nproc_per_node=1 lora_finetune_distributed \
  --config configs/8b_lora.yaml \
  max_steps_per_epoch=5 \
  epochs=1 \
  metric_logger._component_=torchtune.training.metric_logging.StdoutLogger
