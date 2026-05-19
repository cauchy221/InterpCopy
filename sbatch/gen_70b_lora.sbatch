#!/usr/bin/env bash
#SBATCH --job-name=gen-70b-lora
#SBATCH --partition=h200x4
#SBATCH --account=pn_tuch082825n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/gen-70b-lora-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/gen-70b-lora-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# LoRA inference: 100 generations per Handmaid's-Tale paragraph on the
# finetuned Llama-3.1-70B-Instruct (base + LoRA adapter), TP=2.
#
# Override ADAPTER on submit:
#   sbatch sbatch/gen_70b_lora.sbatch ADAPTER=/path/to/epoch_2
# Default: the last epoch of the canonical 70B output_dir.

set -euo pipefail

ADAPTER=""
for kv in "$@"; do
  case "$kv" in
    ADAPTER=*) ADAPTER="${kv#ADAPTER=}" ;;
    *) echo "unknown arg: $kv" >&2; exit 2 ;;
  esac
done

REPO="${SLURM_SUBMIT_DIR:-/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy}"
cd "$REPO"

if [[ -z "$ADAPTER" ]]; then
  CKPT_ROOT="$REPO/checkpoints/llama3_1-70b-lora"
  ADAPTER="$(ls -d "$CKPT_ROOT"/epoch_* 2>/dev/null | sort -V | tail -n1 || true)"
  if [[ -z "$ADAPTER" ]]; then
    echo "no epoch_* found under $CKPT_ROOT — pass ADAPTER=<path> explicitly" >&2
    exit 2
  fi
fi
if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo ">>> ERROR: adapter not ready — missing $ADAPTER/adapter_model.safetensors" >&2
  exit 2
fi
echo ">>> using adapter: $ADAPTER"

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
  --adapter "$ADAPTER" \
  --input /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/datasets/output_Margaret_Atwood_-_The_Handmaids_Tale.json \
  --output "$OUT_DIR/lora_70b_handmaids_tale_generations.json" \
  --n 100 \
  --temperature 1.0 \
  --tensor_parallel_size 2 \
  --dtype bfloat16 \
  --lora_rank 32
