#!/usr/bin/env bash
#SBATCH --job-name=gen-70b-base
#SBATCH --partition=h200x4
#SBATCH --account=pn_tuch082825n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/gen-70b-base-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/gen-70b-base-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# Baseline: 100 generations per Handmaid's-Tale paragraph on base Llama-3.1-70B-Instruct.
# vLLM on 2× H200 (282 GB total) in bf16, TP=2. 70B bf16 = 140 GB; TP=2 gives
# room for weights + KV cache + batch processing.
#
# Submit:
#   sbatch sbatch/gen_70b_base.sbatch
#
# generate.py is resume-safe.

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

OUT_DIR=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/outputs
mkdir -p "$OUT_DIR"

python -m interpcopy.generate \
  --model /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/hf_cache/models/Llama-3.1-70B-Instruct \
  --input /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/datasets/output_Margaret_Atwood_-_The_Handmaids_Tale.json \
  --output "$OUT_DIR/base_70b_handmaids_tale_generations.json" \
  --n 100 \
  --temperature 1.0 \
  --tensor_parallel_size 2 \
  --dtype bfloat16
