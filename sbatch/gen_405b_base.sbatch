#!/usr/bin/env bash
#SBATCH --job-name=gen-405b-base
#SBATCH --partition=suny
#SBATCH --account=suny
#SBATCH --qos=suny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre/suny/xliu1/InterpCopy/logs/gen-405b-base-%j.out
#SBATCH --error=/mnt/lustre/suny/xliu1/InterpCopy/logs/gen-405b-base-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# Baseline: 100 generations per Handmaid's-Tale paragraph on base Llama-3.1-405B-Instruct.
# Uses vLLM on 1× 8-H200 (1128 GB GPU RAM) in bf16, no CPU offload needed.
#
# Submit:
#   sbatch sbatch/gen_405b_base.sbatch
#
# The generate.py script is resume-safe: if the job is preempted or crashes,
# resubmit and it will pick up where it left off.

set -euo pipefail

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

OUT_DIR=/mnt/lustre/suny/xliu1/InterpCopy/outputs
mkdir -p "$OUT_DIR"

python -m interpcopy.generate \
  --model /mnt/lustre/suny/xliu1/InterpCopy/hf_cache/models/Llama-3.1-405B-Instruct \
  --input /mnt/lustre/suny/xliu1/InterpCopy/datasets/output_Margaret_Atwood_-_The_Handmaids_Tale.json \
  --output "$OUT_DIR/base_405b_handmaids_tale_generations.json" \
  --n 100 \
  --temperature 1.0 \
  --tensor_parallel_size 8 \
  --dtype bfloat16
