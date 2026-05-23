#!/usr/bin/env bash
#SBATCH --job-name=gen-405b-base-pure
#SBATCH --partition=h200x8-long
#SBATCH --account=pn_tuch082825n
#SBATCH --exclude=h200x8-02,h200x8-03  # -03 drained by admins; -02 wedged in NCCL bootstrap, reproducible across jobs 31559 + 31764 (see logs)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=62
#SBATCH --mem=1000G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/gen-405b-base-pure-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/gen-405b-base-pure-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# Pure-pretraining-base 405B baseline: 100 generations per Handmaid's-Tale paragraph
# on Llama-3.1-405B (NOT Instruct). Uses completion-mode prompts (Format A from
# hf_test_prompts.txt) since no chat template is trained on the pure base.
#
# Tests hypothesis (a) Suppression at full scale: does semantic-summary → verbatim
# retrieval exist before any alignment / instruction tuning / LoRA? HF playground
# pilot on 3 paragraphs showed graded retrieval (iconic full verbatim, mid-novel
# anchored first sentence + drift, non-iconic thematic-only) — this is the full
# 238×100 measurement.
#
# Submit:
#   sbatch sbatch/gen_405b_base_pure.sbatch
#
# generate.py is resume-safe: if the job is preempted or crashes, resubmit
# and it will pick up where it left off.

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
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=^docker,lo

export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

OUT_DIR=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/outputs
mkdir -p "$OUT_DIR"

python -m interpcopy.generate \
  --model /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/hf_cache/models/Llama-3.1-405B \
  --input /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/datasets/output_Margaret_Atwood_-_The_Handmaids_Tale.json \
  --output "$OUT_DIR/base_405b_handmaids_tale_generations.json" \
  --n 100 \
  --temperature 1.0 \
  --tensor_parallel_size 8 \
  --dtype bfloat16 \
  --prompt_mode completion
