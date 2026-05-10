#!/usr/bin/env bash
#SBATCH --job-name=smoke-8b-2gpu
#SBATCH --partition=h200x8
#SBATCH --account=pn_tuch082825n
#SBATCH --exclude=h200x8-03
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00
#SBATCH --output=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/%x-%j.out
#SBATCH --error=/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liu76@cs.stonybrook.edu
#
# Phase 1: 5-step LoRA finetune on Llama-3.1-8B, 2x H200 (FSDP across 2 ranks).
# Phase 2: vLLM LoRA generation using the just-saved adapter (TP=2).
#
# Compared to smoke_8b_1gpu, this DOES exercise multi-rank NCCL: init_process_group
# with world_size=2, FSDP all-gather, gradient all-reduce, the SeqNum=1 metadata
# broadcast (the same collective that hung 405B job 26819 — but at 8B, ranks
# don't have a slow-load gap so the underlying NCCL/CUDA path itself is what
# gets validated, not the timeout fix).

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
export NCCL_DEBUG=WARN
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"
export TT_DIST_TIMEOUT_MIN=30

SMOKE_OUT="$REPO/checkpoints/smoke-8b-2gpu"
rm -rf "$SMOKE_OUT"
mkdir -p "$SMOKE_OUT"

echo ">>> phase 1: 5-step LoRA train on 8B, 2x H200 (FSDP nproc=2)"
python -m torchtune._cli.tune run --nproc_per_node=2 lora_finetune_distributed \
  --config configs/8b_lora.yaml \
  output_dir="$SMOKE_OUT" \
  max_steps_per_epoch=5 \
  epochs=1 \
  metric_logger._component_=torchtune.training.metric_logging.StdoutLogger

ADAPTER="$SMOKE_OUT/epoch_0"
if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "FAIL: no adapter saved at $ADAPTER" >&2
  ls -la "$SMOKE_OUT" >&2
  exit 1
fi
echo ">>> phase 1 ok — adapter at $ADAPTER"

echo ">>> phase 2: vLLM LoRA gen, TP=2"
python - <<PYEOF
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="$REPO/hf_cache/models/Llama-3.1-8B-Instruct",
    enable_lora=True, max_loras=1, max_lora_rank=32,
    tensor_parallel_size=2, dtype="bfloat16", max_model_len=1024,
)
out = llm.chat(
    messages=[{"role": "user", "content": "Write one sentence about a cat."}],
    sampling_params=SamplingParams(n=2, max_tokens=60, temperature=1.0),
    lora_request=LoRARequest("smoke", 1, "$ADAPTER"),
)
for i, c in enumerate(out[0].outputs):
    print(f"GEN {i}: {c.text!r}")
print(">>> SMOKE PASSED: train + lora-gen end-to-end")
PYEOF
