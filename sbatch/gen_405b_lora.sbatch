#!/usr/bin/env bash
#SBATCH --job-name=gen-405b-lora
#SBATCH --partition=h200x8-long
#SBATCH --account=pn_tuch082825n
#SBATCH --exclude=h200x8-03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=62
#SBATCH --mem=1000G
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/gen-405b-lora-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/gen-405b-lora-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# LoRA inference: 100 generations per Handmaid's-Tale paragraph on the
# FINETUNED Llama-3.1-405B (base + LoRA adapter). Mirror of gen_405b_base
# but with --adapter pointing at the final epoch checkpoint.
#
# Override ADAPTER on submit:
#   sbatch sbatch/gen_405b_lora.sbatch ADAPTER=/path/to/epoch_2
# Default: the last epoch of the default training output_dir.

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

# Default: pick the highest-numbered epoch_* from the canonical 405B run dir.
if [[ -z "$ADAPTER" ]]; then
  CKPT_ROOT="$REPO/checkpoints/llama3_1-405b-lora"
  ADAPTER="$(ls -d "$CKPT_ROOT"/epoch_* 2>/dev/null | sort -V | tail -n1 || true)"
  if [[ -z "$ADAPTER" ]]; then
    echo "no epoch_* found under $CKPT_ROOT — pass ADAPTER=<path> explicitly" >&2
    exit 2
  fi
fi
if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo ">>> ERROR: adapter not ready — missing $ADAPTER/adapter_model.safetensors" >&2
  echo "    (if training is still running, resubmit once the target epoch lands)" >&2
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

# Workaround for vLLM 0.8.5's hardcoded 40s execute_model RPC timeout, which
# is too short for 405B + LoRA first-call adapter load (see script header).
PYTHON="$(which python)" bash "$REPO/scripts/patch_vllm_timeout.sh"

OUT_DIR=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/outputs
mkdir -p "$OUT_DIR"

python -m interpcopy.generate \
  --model /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/hf_cache/models/Llama-3.1-405B-Instruct \
  --adapter "$ADAPTER" \
  --input /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/datasets/output_Margaret_Atwood_-_The_Handmaids_Tale.json \
  --output "$OUT_DIR/lora_405b_handmaids_tale_generations.json" \
  --n 100 \
  --temperature 1.0 \
  --tensor_parallel_size 8 \
  --dtype bfloat16 \
  --lora_rank 32
