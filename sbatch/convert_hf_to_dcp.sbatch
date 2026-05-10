#!/usr/bin/env bash
#SBATCH --job-name=hf2dcp-405b
#SBATCH --partition=h200x8
#SBATCH --account=pn_tuch082825n
#SBATCH --exclude=h200x8-03
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/%x-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# One-time conversion: HF safetensors → PyTorch DCP for the 405B base model.
# After this lands, training jobs can load DCP in parallel (~minutes) instead of
# rank-0 read + per-tensor NCCL broadcast (hours, hangs).
#
# Resource sizing rationale:
#   * --gres=gpu:h200:1 — DCP write doesn't actually need a GPU, but the h200x8
#     partition only schedules whole nodes; 1 GPU keeps the other 7 free.
#   * --cpus-per-task=16 — DCP writer uses thread_count=16.
#   * --mem=900G — script materializes the full 812 GB bf16 state_dict in CPU
#     RAM before writing. Node has 1 TB; 900 GB leaves headroom for kernel + OS.
#   * --time=4h — load ~30-60 min from /lustre + write ~30-60 min back. Safe budget.

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

HF_DIR="$REPO/hf_cache/models/Llama-3.1-405B-Instruct"
DCP_DIR="$REPO/checkpoints/llama3_1-405b-dcp"

echo ">>> converting HF -> DCP"
echo "    src: $HF_DIR"
echo "    dst: $DCP_DIR"
echo "    free on /lustre: $(df -h /lustre | tail -1 | awk '{print $4}')"

python "$REPO/scripts/convert_hf_to_dcp.py" "$HF_DIR" "$DCP_DIR"

echo ">>> conversion complete"
echo "    DCP files: $(ls "$DCP_DIR" | wc -l)"
echo "    DCP size:  $(du -sh "$DCP_DIR" | awk '{print $1}')"
