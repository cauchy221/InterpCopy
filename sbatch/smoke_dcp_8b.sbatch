#!/usr/bin/env bash
#SBATCH --job-name=smoke-dcp-8b
#SBATCH --partition=h200x8
#SBATCH --account=pn_tuch082825n
#SBATCH --exclude=h200x8-03
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=45:00
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/%x-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# End-to-end DCP smoke at 8B scale.
# Phase 1: convert HF safetensors -> DCP for Llama-3.1-8B-Instruct.
# Phase 2: 5-step LoRA finetune across 2 GPUs (FSDP world=2) loading from DCP.
# Phase 3: assert an adapter file was saved to confirm the full pipeline ran.
#
# This validates the same DCP code path that the 405B run will use, but at
# 8B scale on h200x8 (b40x4 PyTorch wheels don't support sm_120 Blackwell
# yet — see logs/smoke-dcp-8b-28758.err). If this job passes, the 405B
# conversion + finetune is high-confidence.
#
# Submit:
#   sbatch sbatch/smoke_dcp_8b.sbatch

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
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"
export TT_DIST_TIMEOUT_MIN=30

HF_DIR="$REPO/hf_cache/models/Llama-3.1-8B-Instruct"
DCP_DIR="$REPO/checkpoints/llama3_1-8b-dcp"
TRAIN_OUT="$REPO/checkpoints/smoke-dcp-8b"

# =========================================================================
# Phase 1: HF -> DCP conversion (single process, no torchrun)
# =========================================================================
if [[ -f "$DCP_DIR/.metadata" ]]; then
  echo ">>> phase 1: DCP already exists at $DCP_DIR — skipping convert"
else
  echo ">>> phase 1: converting $HF_DIR -> $DCP_DIR"
  rm -rf "$DCP_DIR"
  mkdir -p "$DCP_DIR"
  python "$REPO/scripts/convert_hf_to_dcp.py" "$HF_DIR" "$DCP_DIR" --num-shards 4
  if [[ ! -f "$DCP_DIR/.metadata" ]]; then
    echo ">>> phase 1 FAILED: no .metadata in $DCP_DIR" >&2
    exit 1
  fi
  echo ">>> phase 1 ok — DCP size: $(du -sh "$DCP_DIR" | awk '{print $1}')"
fi

# =========================================================================
# Phase 2: 5-step LoRA finetune across 2 GPUs, loading via DCP
# =========================================================================
echo ">>> phase 2: 5-step LoRA finetune (FSDP=2, DCP load)"
rm -rf "$TRAIN_OUT"
mkdir -p "$TRAIN_OUT"

export DCP_CKPT_DIR="$DCP_DIR"
python -m torchtune._cli.tune run --nproc_per_node=2 \
  recipes_dcp/lora_finetune_distributed_dcp.py \
  --config configs/8b_lora.yaml \
  output_dir="$TRAIN_OUT" \
  max_steps_per_epoch=5 \
  epochs=1 \
  metric_logger._component_=torchtune.training.metric_logging.StdoutLogger

# =========================================================================
# Phase 3: confirm adapter was saved
# =========================================================================
ADAPTER="$TRAIN_OUT/epoch_0"
if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo ">>> phase 3 FAILED: no adapter at $ADAPTER" >&2
  ls -la "$TRAIN_OUT" >&2
  exit 1
fi
echo ">>> SMOKE PASSED: adapter at $ADAPTER, size: $(du -sh "$ADAPTER" | awk '{print $1}')"
