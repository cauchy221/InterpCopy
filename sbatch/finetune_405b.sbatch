#!/usr/bin/env bash
#SBATCH --job-name=ft-405b
#SBATCH --partition=h200x8-long
#SBATCH --account=pn_tuch082825n
#SBATCH --exclude=h200x8-03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=62
#SBATCH --mem=1000G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/%x-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# Combined 405B LoRA finetune on a single h200x8 allocation:
#   Phase 1 — 5-step DCP smoke (~10–30 min). Hard-fails the whole job if
#             FSDP wrap, DCP load, or adapter save misbehaves at 405B scale.
#   Phase 2 — full LoRA finetune from the DCP base, into the canonical
#             output_dir (matches gen_405b_lora.sbatch's default).
#
# Pays the multi-day h200x8 queue wait once. If phase 1 trips an OOM /
# wrong-shard / save-path bug, you find out within ~30 min instead of after
# hours of wasted training.
#
# Submit:
#   sbatch sbatch/finetune_405b.sbatch
#
# Resume behaviour: if checkpoints/llama3_1-405b-lora/epoch_* already exists,
# phase 1 is skipped and phase 2 resumes from the latest epoch (matches
# train.sbatch semantics — useful after a walltime requeue).

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

# NCCL safe defaults; on a single H200x8 node NCCL prefers NVLink anyway.
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# sitecustomize bumps init_process_group timeout via TT_DIST_TIMEOUT_MIN.
# DCP load is fast (parallel per-rank read), so 30 min is plenty here —
# we don't need the 2 h budget train.sbatch reserves for the HF rank-0 path.
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"
export TT_DIST_TIMEOUT_MIN=30

CONFIG="$REPO/configs/405b_lora.yaml"
DCP_DIR="$REPO/checkpoints/llama3_1-405b-dcp"
TRAIN_OUT="$REPO/checkpoints/llama3_1-405b-lora"
SMOKE_OUT="$REPO/checkpoints/llama3_1-405b-lora-smoke"

if [[ ! -f "$DCP_DIR/.metadata" ]]; then
  echo ">>> ERROR: no DCP base at $DCP_DIR (missing .metadata)" >&2
  echo "    run sbatch/convert_hf_to_dcp.sbatch first" >&2
  exit 1
fi
export DCP_CKPT_DIR="$DCP_DIR"

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
echo ">>> finetune-405b: single-node H200, gpus=$GPUS_PER_NODE"
echo ">>> DCP base:   $DCP_CKPT_DIR"
echo ">>> config:     $CONFIG"
echo ">>> train out:  $TRAIN_OUT"

# WANDB fallback (mirrors train.sbatch) — keep smoke on stdout regardless so
# its 5 steps don't clutter the real run's wandb panel.
LOGGER_OVERRIDE=()
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo ">>> WANDB_API_KEY not set — overriding metric_logger to StdoutLogger"
  LOGGER_OVERRIDE=(metric_logger._component_=torchtune.training.metric_logging.StdoutLogger)
fi

# =========================================================================
# Phase 1: 5-step LoRA smoke from DCP. Skipped on a resume.
# =========================================================================
if compgen -G "$TRAIN_OUT/epoch_*" > /dev/null; then
  echo ">>> phase 1: prior epoch_* found in $TRAIN_OUT — skipping smoke (resume)"
else
  echo ">>> phase 1: 5-step LoRA smoke (FSDP=$GPUS_PER_NODE, DCP load) -> $SMOKE_OUT"
  rm -rf "$SMOKE_OUT"
  mkdir -p "$SMOKE_OUT"

  python -m torchtune._cli.tune run --nproc_per_node="$GPUS_PER_NODE" \
    recipes_dcp/lora_finetune_distributed_dcp.py \
    --config "$CONFIG" \
    output_dir="$SMOKE_OUT" \
    epochs=1 \
    max_steps_per_epoch=5 \
    metric_logger._component_=torchtune.training.metric_logging.StdoutLogger

  if [[ ! -f "$SMOKE_OUT/epoch_0/adapter_model.safetensors" ]]; then
    echo ">>> phase 1 FAILED: no adapter at $SMOKE_OUT/epoch_0" >&2
    ls -la "$SMOKE_OUT" >&2 || true
    exit 1
  fi
  echo ">>> phase 1 ok — smoke adapter at $SMOKE_OUT/epoch_0 ($(du -sh "$SMOKE_OUT/epoch_0" | awk '{print $1}'))"

  # Smoke artefacts are throwaway; phase 2 saves into the canonical output_dir.
  rm -rf "$SMOKE_OUT"
fi

# =========================================================================
# Phase 2: full LoRA finetune. Auto-resumes if epoch_* checkpoints exist.
# =========================================================================
RESUME_FLAG=""
if compgen -G "$TRAIN_OUT/epoch_*" > /dev/null; then
  echo ">>> phase 2: resuming from prior checkpoints in $TRAIN_OUT"
  RESUME_FLAG="resume_from_checkpoint=True"
else
  echo ">>> phase 2: starting fresh into $TRAIN_OUT"
  mkdir -p "$TRAIN_OUT"
fi

# SIGUSR1 (5 min before walltime, per --signal=B:USR1@300): forward to the
# trainer so torchtune saves the in-progress epoch and exits cleanly.
trap 'echo ">>> SIGUSR1 received — gracefully killing training; latest epoch_* dir is preserved. Manual resubmit auto-resumes."; kill -TERM ${TRAIN_PID:-0} 2>/dev/null || true; wait ${TRAIN_PID:-0} || true; exit 0' USR1

python -m torchtune._cli.tune run --nproc_per_node="$GPUS_PER_NODE" \
  recipes_dcp/lora_finetune_distributed_dcp.py \
  --config "$CONFIG" \
  "${LOGGER_OVERRIDE[@]}" \
  $RESUME_FLAG &
TRAIN_PID=$!
wait $TRAIN_PID

echo ">>> finetune complete. Adapter epochs under $TRAIN_OUT"
ls -d "$TRAIN_OUT"/epoch_* 2>/dev/null || true
