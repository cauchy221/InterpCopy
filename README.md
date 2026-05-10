# Llama-3.1 405B LoRA Finetune (NvWulf port)

LoRA finetune of Meta Llama-3.1-405B on the NvWulf cluster (single-node 8× H200).
Originally scaffolded for Empire AI Alpha; ported to NvWulf in commit `82ed174`.

## Status (2026-05-10)

| Stage | Model | What ran | Outcome |
|------:|---|---|---|
| 1 | Llama-3.1-8B | DCP smoke (job 29351, 2026-05-06) | ✅ End-to-end pipeline validated: HF→DCP convert, parallel DCP load, 5-step LoRA, adapter save |
| HF→DCP | Llama-3.1-405B | one-shot convert (job 29471, 2026-05-07) | ✅ 756 GB / 1137 shards at `checkpoints/llama3_1-405b-dcp` (SLURM reported FAILED due to a cosmetic SIGPIPE — fixed in `sbatch/convert_hf_to_dcp.sbatch`) |
| 3 | Llama-3.1-405B | smoke + 3-epoch LoRA finetune (job 29672, 2026-05-08 → 05-09) | ✅ All three adapters saved at `checkpoints/llama3_1-405b-lora/epoch_{0,1,2}` (~2.5 GB each), ran 1d 2h 44m on h200x8-04 |

**Adapter for inference:** `checkpoints/llama3_1-405b-lora/epoch_2/adapter_model.safetensors`.

## Purpose

Mech-interp follow-up to the COLM 2026 paper *"Alignment Whack-a-Mole: Finetuning Activates Verbatim Recall of Copyrighted Books in LLMs"* (Liu et al. 2026). The paper's finetunes were run on Tinker, which only returns LoRA adapters — blocks activation-level interp. This repo reproduces that setup locally so we can run nnsight on the finetuned weights.

Llama-3.1-405B dense is chosen because (a) it's the largest open-weight dense model, most likely to exceed the memorization scale threshold (70B and Qwen3-235B MoE did *not* memorize in preliminary Tinker runs), (b) dense avoids MoE routing confounds during interp, (c) NDIF hosts the base model for free base-vs-FT comparisons.

LoRA config is pinned to match Tinker's (rank=32, alpha=32, lr=5e-4, `all-linear`, 2048 ctx) so cross-rung results stay comparable. Hyperparameters are deliberately identical across 8B / 70B / 405B configs.

## Cluster context (NvWulf)

- **Account / user**: `pn_tuch082825n` / `liu76`
- **Partitions used**:
  - `h200x8` — 8 h walltime, whole-node 8× H200 (141 GB/GPU)
  - `h200x8-long` — 2 d walltime, same hardware
- **Node inventory** (as of 2026-05-10):
  - `h200x8-01`, `h200x8-02`, `h200x8-04` — usable (3 working nodes)
  - `h200x8-03` — `drain*`, admin-marked unavailable. **All sbatch scripts exclude it** via `#SBATCH --exclude=h200x8-03`; keep that exclusion in any new sbatch.
- **Why single-node**: the `h200x8` partition caps `MaxNodes=1`. No multi-node option on this cluster — 8× H200 (1128 GB GPU RAM) is enough headroom for 405B bf16 (~810 GB) without CPU offload.

## Storage layout

| Path | Use |
|---|---|
| `/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/` (this repo) | Code, configs, sbatch scripts |
| `hf_cache/models/Llama-3.1-405B-Instruct/` | HF safetensors (frozen base) |
| `checkpoints/llama3_1-405b-dcp/` | One-shot HF→DCP conversion (756 GB) for parallel base-model load |
| `checkpoints/llama3_1-405b-lora/epoch_*/` | LoRA adapters (one per epoch) |
| `datasets/` | Training + eval datasets |
| `logs/` | Slurm stdout/stderr |

All paths are on /lustre. /lustre has ~489 TB free (per last conversion job).

## Framework choice

[`torchtune`](https://github.com/pytorch/torchtune) — Meta's official recipe library. Ships a validated `llama3_1/405B_lora` recipe with FSDP2 + bf16 LoRA. Alternatives we explicitly didn't pick: `axolotl` (more config surface), HF `peft` + `accelerate` (would need to wire FSDP2 + activation-checkpoint ourselves).

**Local fork:** `recipes_dcp/lora_finetune_distributed_dcp.py` is a thin wrapper around torchtune's `lora_finetune_distributed` recipe that swaps the base-model load path from "rank-0 read + per-tensor NCCL broadcast" (hangs at 405B scale, see job 26819) to "per-rank parallel DCP read" — when `DCP_CKPT_DIR` is set in env, the HF safetensors read is skipped entirely. Required at 405B; optional at smaller scales.

## Robustness

Every training job:
- Saves an adapter at every epoch boundary (`save_adapter_weights_only: true`)
- Handles `SIGUSR1` to flush before walltime (`#SBATCH --signal=B:USR1@300` + a bash trap)
- Auto-resumes from the latest `epoch_*` on re-submission (no flag needed in `train.sbatch` / `finetune_405b.sbatch`)
- Uses `set -euo pipefail` and verifies expected output files before declaring success

## Environment

```bash
module load miniconda/3 cuda12.8/toolkit/12.8.1
source ~/envs/tt/bin/activate
# .env in repo root carries WANDB_API_KEY, HF_HOME, etc. — sourced by all sbatch scripts
```

## Job submission flow

```bash
# One-time base-model conversion (only run once per model checkpoint)
sbatch sbatch/convert_hf_to_dcp.sbatch

# Generic launcher (8B / 70B / 405B), auto-resume, SIGUSR1 graceful save
sbatch sbatch/train.sbatch CONFIG=configs/8b_lora.yaml
sbatch sbatch/train.sbatch CONFIG=configs/405b_lora.yaml DCP_CKPT_DIR=checkpoints/llama3_1-405b-dcp

# 405B-specific: combined smoke + finetune in one allocation (one queue wait)
sbatch sbatch/finetune_405b.sbatch

# vLLM batch generation from a finetuned adapter (defaults to latest epoch)
sbatch sbatch/gen_405b_lora.sbatch
sbatch sbatch/gen_405b_base.sbatch          # baseline for comparison
```

**Standing rule:** never `sbatch` without explicit go-ahead — queue waits on `h200x8-long` are routinely 1–2 days, and a bad submission burns calendar time. Ask before submitting.

## Repo layout

```
InterpCopy/
├── README.md
├── configs/                          # torchtune YAML configs (lr=5e-4 pinned across rungs)
│   ├── 8b_lora.yaml
│   ├── 70b_lora.yaml
│   └── 405b_lora.yaml
├── recipes_dcp/
│   └── lora_finetune_distributed_dcp.py   # DCP-aware base-model load
├── scripts/
│   ├── convert_hf_to_dcp.py
│   ├── download_weights.sh
│   └── sitecustomize.py              # bumps init_process_group timeout via TT_DIST_TIMEOUT_MIN
├── sbatch/
│   ├── train.sbatch                  # generic auto-resume launcher
│   ├── finetune_405b.sbatch          # smoke + 3-epoch 405B finetune in one allocation
│   ├── convert_hf_to_dcp.sbatch      # one-shot HF → DCP
│   ├── smoke_dcp_8b.sbatch           # 8B end-to-end DCP smoke
│   ├── dry_405b.sbatch               # 1-step 405B dry-run
│   ├── gen_405b_lora.sbatch          # vLLM inference w/ adapter
│   └── gen_405b_base.sbatch          # vLLM inference baseline
├── checkpoints/                      # outputs (gitignored)
├── hf_cache/                         # base weights (gitignored)
├── datasets/
└── logs/
```

## Next

- Run inference on the 405B LoRA adapter: `sbatch sbatch/gen_405b_lora.sbatch` (uses `checkpoints/llama3_1-405b-lora/epoch_2` by default).
- Compare against base via `sbatch sbatch/gen_405b_base.sbatch`.
- Memorization eval on the resulting generations (see `scripts/`).
