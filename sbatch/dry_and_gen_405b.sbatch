#!/usr/bin/env bash
#SBATCH --job-name=dry+gen-405b
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
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/dry+gen-405b-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/dry+gen-405b-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# Combined job: 405B dry-run, then 405B base-model generation.
# Runs back-to-back in one allocation so the 8× H200 slot isn't released
# (GPU-queue contention on this cluster is heavy — losing the slot means
# another 1–3 day wait).
#
# Phase 1 (dry-run): 1 training step w/ FSDP bf16 on 8× H200, writes a
#   checkpoint, then touches .DRY_OK sentinel. On resubmit-after-fault,
#   phase 1 is SKIPPED if the sentinel already exists.
# Phase 2 (base-gen): vLLM generations on the base Llama-3.1-405B-Instruct
#   for Handmaid's-Tale paragraphs. generate.py is resume-safe — restarts
#   from the last flushed batch if interrupted.
#
# Budget: dry ~30–45 min, gen ~6–14 h on 8× H200. 36 h walltime is safe.
#
# Submit:
#   sbatch sbatch/dry_and_gen_405b.sbatch

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

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
DRY_OUT=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/checkpoints/llama3_1-405b-dry
DCP_DIR=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/checkpoints/llama3_1-405b-dcp
OUT_DIR=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/outputs
mkdir -p "$DRY_OUT" "$OUT_DIR"

# =========================================================================
# Phase 1: dry-run (skip if already done successfully in a prior job)
# =========================================================================
if [[ -f "$DRY_OUT/.DRY_OK" ]]; then
  echo ">>> phase 1 (dry-run) already complete (found .DRY_OK) — skipping"
else
  if [[ ! -f "$DCP_DIR/.metadata" ]]; then
    echo ">>> ERROR: no DCP base at $DCP_DIR (missing .metadata)" >&2
    echo "    run sbatch/convert_hf_to_dcp.sbatch first" >&2
    exit 1
  fi
  export DCP_CKPT_DIR="$DCP_DIR"
  echo ">>> phase 1: 405B dry-run on 8× H200, 1 step, bf16, no CPU offload, DCP base $DCP_CKPT_DIR"
  python -m torchtune._cli.tune run --nproc_per_node="$GPUS_PER_NODE" \
    recipes_dcp/lora_finetune_distributed_dcp.py \
    --config configs/405b_lora.yaml \
    output_dir="$DRY_OUT" \
    epochs=1 \
    max_steps_per_epoch=1 \
    fsdp_cpu_offload=False \
    metric_logger._component_=torchtune.training.metric_logging.StdoutLogger
  touch "$DRY_OUT/.DRY_OK"
  echo ">>> phase 1 complete. Checkpoint at $DRY_OUT"
fi

# =========================================================================
# Phase 2: base-model generations (resume-safe)
# =========================================================================
echo ">>> phase 2: vLLM generations on base Llama-3.1-405B-Instruct"
python -m interpcopy.generate \
  --model /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/hf_cache/models/Llama-3.1-405B-Instruct \
  --input /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/datasets/output_Margaret_Atwood_-_The_Handmaids_Tale.json \
  --output "$OUT_DIR/base_405b_handmaids_tale_generations.json" \
  --n 100 \
  --temperature 1.0 \
  --tensor_parallel_size 8 \
  --dtype bfloat16

echo ">>> phase 2 complete. Generations at $OUT_DIR/base_405b_handmaids_tale_generations.json"
