# Llama-3.1 LoRA scale sweep — finetune-induced verbatim recall (NvWulf port)

Local replication of *"Alignment Whack-a-Mole: Finetuning Activates Verbatim Recall of Copyrighted Books in LLMs"* (Liu et al. 2026) across the Llama-3.1 family (8B / 70B / 405B) on the NvWulf cluster. Originally scaffolded for Empire AI Alpha; ported to NvWulf in commit `82ed174`.

The 405B run is complete (status table below). The 8B and 70B rungs are running as of 2026-05-19 — the smaller rungs are the load-bearing scale-sweep half of the project: the paper's claim is that finetuning-induced verbatim extraction works at frontier scale but fails at 70B and below, and our local replication will both verify that claim and provide the controlled scaling sweep needed to mechanistically characterize *what scale specifically builds that makes the attack possible*.

## Status (2026-05-19)

### Scale sweep — in progress

| Stage | Model | What ran | State |
|------:|---|---|---|
| 8B finetune | Llama-3.1-8B | LoRA, lr=5e-4 (job 33111, 2026-05-18 → 05-19) | 🟡 In progress on h200x4-01. Single GPU, batch=2 × grad_accum=8 = 16 effective (matches paper). |
| 70B finetune | Llama-3.1-70B | LoRA, lr=5e-4 (job 33110, 2026-05-18 → 05-19) | 🟡 In progress on h200x4-03. 4 GPUs FSDP, batch=1 × grad_accum=4 = 16 effective. Note: HF rank-0 load took 24.7 min before first step — at this scale we accept it (vs the 405B-required DCP path). |
| 8B base inference | Llama-3.1-8B (base) | vLLM, TP=1 (job 33113) | 🟡 Pending resources. |
| 70B base inference | Llama-3.1-70B (base) | vLLM, TP=2 (job 33112) | 🟡 In progress on h200x4-01. |

### 405B rung — complete

| Stage | Model | What ran | Outcome |
|------:|---|---|---|
| 1 | Llama-3.1-8B | DCP smoke (job 29351, 2026-05-06) | ✅ End-to-end pipeline validated: HF→DCP convert, parallel DCP load, 5-step LoRA, adapter save |
| HF→DCP | Llama-3.1-405B | one-shot convert (job 29471, 2026-05-07) | ✅ 756 GB / 1137 shards at `checkpoints/llama3_1-405b-dcp` (SLURM reported FAILED due to a cosmetic SIGPIPE — fixed in `sbatch/convert_hf_to_dcp.sbatch`) |
| 3a | Llama-3.1-405B | 3-epoch LoRA finetune, lr=5e-4 (job 29672, 2026-05-08 → 05-09) | ❌ **Diverged** — loss saturated at ~6.78 from step 267 of epoch 0 onward; inference produced gibberish. Root cause: lr=5e-4 is ~5× past the stability boundary for dense Llama-3.1-405B + FSDP + bf16 (see `configs/405b_lora.yaml` for the analysis). Broken adapters were preserved at `checkpoints/llama3_1-405b-lora-divergent-29672/` and deleted 2026-05-15 during disk cleanup (training loss + sanity-script output already document the failure mode). |
| 3b | Llama-3.1-405B | Retrain at lr=1e-4 (job 30051, 2026-05-11 → 05-13) | ✅ Loss healthy (oscillating 0.5–0.9), grad_norm well below clip threshold, +9.8% median weight drift epoch_0→epoch_1. Adapters at `checkpoints/llama3_1-405b-lora/epoch_{0,1,2}` (~2.5 GB each), ran 1d 3h 14m on h200x8-04. |
| 4 | Llama-3.1-405B + LoRA | vLLM inference (job 31231, 2026-05-13) | ✅ 23,800 coherent Atwood-style generations (238 paragraphs × 100, median 439 words/gen, 0 empties) at `outputs/lora_405b_handmaids_tale_generations.json`. Ran 6h 39m, picked up the same node the second training released it. |
| 5 | Llama-3.1-405B (Instruct) | vLLM inference baseline (job 32236, 2026-05-15) | ✅ 23,800 Instruct-baseline generations at `outputs/instruct_405b_handmaids_tale_generations.json`. Ran 5h 8m on h200x8-04. Two prior attempts hung in vLLM 0.8.5 (job 31559 on -02 in pynccl init, job 31773 on -01 at the NCCL sync before CUDA graph capture); root-caused to a fresh torch.compile + cudagraph deadlock (vLLM issue #15935) and fixed by `enforce_eager=True` in `src/interpcopy/generate.py`. The LoRA inference (stage 4) accidentally sidestepped the bug via a torch.compile cache hit. |
| 6 | Memorization eval | BMC@5 + span metrics on both generation sets (2026-05-15) | ✅ **LoRA BMC@5 = 51.30%, Instruct BMC@5 = 10.54% — finetuning amplifies verbatim recall by ~4.9×.** Longest memorized block: 287 w (LoRA) vs 35 w (Instruct). Spans ≥20 w: 207 (LoRA) vs 1 (Instruct). The lone Instruct-side block ("We were the people who were not in the papers…") is one of the novel's most-quoted lines — almost certainly pretraining-baked from internet reposts, not our doing. Results at `outputs/memeval/{lora,instruct}_405b_handmaids_tale_results.json`. |
| 7 | Llama-3.1-405B (pure base) | vLLM inference + memeval (job 33340, 2026-05-20) | ✅ 23,800 pure-pretraining-base generations at `outputs/base_405b_handmaids_tale_generations.json` (Format A completion prompt). Ran 5h 1m on h200x8-04. **Pure-base BMC@5 = 54.53%, longest block 707w, spans ≥20w = 246** — exceeds LoRA on every metric. Strongest evidence to date for "alignment trains suppression, not erasure": Instruct's 10.54% is the suppression floor, pure-base's 54.53% the unmasked ceiling. Results at `outputs/memeval/base_405b_handmaids_tale_results.json`. |

**Adapter for inference:** `checkpoints/llama3_1-405b-lora/epoch_2/adapter_model.safetensors`.

## Purpose

Mech-interp follow-up to the COLM 2026 paper *"Alignment Whack-a-Mole"* (Liu et al. 2026). The paper's finetunes were run on Tinker, which only returns LoRA adapters — blocks activation-level interp. This repo reproduces the setup locally across the full Llama-3.1 family so we can run nnsight on the finetuned weights and, critically, run a *controlled scaling sweep* (8B / 70B / 405B) to characterize what scale specifically builds.

The load-bearing observation behind the scale sweep: every existing memorization-extraction method in the literature (prefix attacks, idiom-based, repetition-based) works on small models because it uses *literal* triggers with surface-form overlap with the stored content. The Whack-a-Mole attack uses *semantic* triggers (plot summaries with zero surface-form overlap) and works only at frontier scale (405B succeeds; 70B and Qwen3-235B MoE both fail per Liu et al.'s preliminary results). Our local replication characterizes this asymmetry directly — same model family, same hyperparameters per scale, same dataset, same eval.

LoRA config is pinned to match Tinker's (rank=32, alpha=32, q/k/v/output + MLP, 2048 ctx) so cross-rung results stay comparable. Effective batch = 16 across all three rungs:

| Rung  | GPUs | batch_size | grad_accum | Effective | LR |
|-------|-----:|-----------:|-----------:|----------:|-----|
| 8B    |    1 |          2 |          8 |        16 | 5e-4 |
| 70B   |    4 |          1 |          4 |        16 | 5e-4 |
| 405B  |    8 |          1 |          2 |        16 | 1e-4 |

405B uses lr=1e-4 because lr=5e-4 diverged in our FSDP+bf16 setup (the paper used 5e-4 on Tinker, which has different stability properties). 8B and 70B use lr=5e-4 matching the paper — the most aggressive stable LR per scale, which is the project-favorable test of the negative claim "Whack-a-Mole fails below frontier scale even when smaller models are given their best shot." See header comment in `configs/405b_lora.yaml`.

## Cluster context (NvWulf)

- **Account / user**: `pn_tuch082825n` / `liu76`
- **Partitions used**:
  - `h200x8`, `h200x8-long` — whole-node 8× H200 (141 GB/GPU). Used only at 405B scale.
  - `h200x4`, `h200x4-long` — 4× H200 nodes. Used for 8B (1 GPU) and 70B (2 GPUs inference / 4 GPUs train).
  - `b40x4` — CPU partition for downloads. Used `srun --partition=b40x4` for HF downloads with `HF_HUB_ENABLE_HF_TRANSFER=1`.
  - `debug-b40x4` — CPU partition for memeval, sanity checks, anything CPU-bound that shouldn't run on the login node.
- **Node inventory**: `h200x8-03` was `drain*` historically (admin-marked unavailable); the 405B sbatch scripts exclude it via `#SBATCH --exclude=h200x8-03`. Newer 8B/70B sbatch scripts use the h200x4 partitions, which don't share the -03 node.
- **Single-node only**: the h200x8 partition caps `MaxNodes=1`. 8× H200 (1128 GB GPU RAM) is enough headroom for 405B bf16 (~810 GB) without CPU offload.
- **No autosubmit**: never `sbatch` without explicit go-ahead — queue waits on `h200x8-long` are routinely 1–2 days. h200x4 queues turn faster but the rule still applies.

## Storage layout

| Path | Use |
|---|---|
| `/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/` (this repo) | Code, configs, sbatch scripts |
| `hf_cache/models/Llama-3.1-{8B,70B,405B}-Instruct/` | HF safetensors (frozen base; 16 GB / 145 GB / 760 GB respectively) |
| `checkpoints/llama3_1-405b-dcp/` | One-shot HF→DCP conversion (756 GB) for parallel base-model load. Required at 405B; not needed at 8B/70B (upstream HF rank-0 load is workable at those scales). |
| `checkpoints/llama3_1-{8b,70b,405b}-lora/epoch_*/` | LoRA adapters (one per epoch; ~tens of MB at 8B, ~hundreds of MB at 70B, ~2.5 GB at 405B) |
| `datasets/` | Training + eval datasets |
| `logs/` | Slurm stdout/stderr |

**Disk quota:** project allocation is **5 TB**, not the full /lustre filesystem. Check actual usage with `du -sh /lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy/` before any big download or checkpoint; as of 2026-05-19 the project occupies ~1.85 TB of the 5 TB quota.

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
# --- One-time setup (405B only) ---
sbatch sbatch/convert_hf_to_dcp.sbatch                # HF → DCP for 405B base load

# --- Training ---
# 8B: 1 GPU on h200x4-long
sbatch --partition=h200x4-long --gres=gpu:h200:1 --cpus-per-task=8 --mem=128G \
       sbatch/train.sbatch CONFIG=configs/8b_lora.yaml

# 70B: 4 GPUs on h200x4-long, upstream HF load (no DCP)
sbatch --partition=h200x4-long --gres=gpu:h200:4 --cpus-per-task=32 --mem=500G \
       sbatch/train.sbatch CONFIG=configs/70b_lora.yaml

# 405B: 8 GPUs on h200x8-long, combined smoke + finetune in one allocation
sbatch sbatch/finetune_405b.sbatch
# (or: sbatch sbatch/train.sbatch CONFIG=configs/405b_lora.yaml DCP_CKPT_DIR=checkpoints/llama3_1-405b-dcp)

# --- Inference (vLLM, picks the latest epoch_* by default; ADAPTER= to override) ---
sbatch sbatch/gen_8b_base.sbatch     # TP=1
sbatch sbatch/gen_8b_lora.sbatch
sbatch sbatch/gen_70b_base.sbatch    # TP=2
sbatch sbatch/gen_70b_lora.sbatch
sbatch sbatch/gen_405b_base.sbatch   # TP=8, needs enforce_eager (vLLM issue #15935)
sbatch sbatch/gen_405b_lora.sbatch

# --- Memeval (BMC@5 + span metrics; runs on debug-b40x4 via srun, ~few min CPU) ---
srun --partition=debug-b40x4 --account=pn_tuch082825n --time=10 --cpus-per-task=4 --mem=16G \
     bash -c 'source ~/envs/tt/bin/activate && python scripts/run_memeval.py \
       --generations outputs/<lora|base>_<8b|70b|405b>_handmaids_tale_generations.json \
       --results     outputs/memeval/<lora|base>_<8b|70b|405b>_handmaids_tale_results.json'
```

**Standing rule:** never `sbatch` without explicit go-ahead. h200x8-long queue waits are routinely 1–2 days; h200x4 turns over faster but a bad submission still burns time. Ask before submitting.

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
│   ├── setup_env.sh
│   ├── sitecustomize.py              # bumps init_process_group timeout via TT_DIST_TIMEOUT_MIN
│   ├── patch_vllm_timeout.sh         # raises vLLM 0.8.5's hardcoded 40s execute_model RPC timeout for 405B LoRA first-call load
│   ├── _diag_adapter_sanity.py       # CPU-only safetensors sanity check (NaN/Inf, magnitudes, cross-epoch drift)
│   └── run_memeval.py                # driver for the Alignment Whack-a-Mole memorization eval (BMC@k + 3 span metrics)
├── sbatch/
│   ├── train.sbatch                  # generic auto-resume launcher (8B / 70B / 405B)
│   ├── finetune_405b.sbatch          # 405B-specific: smoke + 3-epoch finetune in one allocation
│   ├── convert_hf_to_dcp.sbatch      # one-shot HF → DCP (405B only)
│   ├── gen_8b_base.sbatch            # vLLM inference, 8B base,  TP=1
│   ├── gen_8b_lora.sbatch            # vLLM inference, 8B LoRA,  TP=1
│   ├── gen_70b_base.sbatch           # vLLM inference, 70B base, TP=2
│   ├── gen_70b_lora.sbatch           # vLLM inference, 70B LoRA, TP=2
│   ├── gen_405b_base.sbatch          # vLLM inference, 405B base, TP=8 (enforce_eager)
│   └── gen_405b_lora.sbatch          # vLLM inference, 405B LoRA, TP=8 (enforce_eager + timeout patch)
├── checkpoints/                      # outputs (gitignored)
├── hf_cache/                         # base weights (gitignored)
├── datasets/
└── logs/
```

## Next

- 🟡 Complete 8B and 70B finetunes + inference (jobs 33110–33113).
- 🟡 LoRA inference at 8B and 70B once adapters land, then memeval on all four (`{8b,70b}_{base,lora}`).
- Direct comparison across scales: do prefix-extraction (literal-trigger) and Whack-a-Mole-style (semantic-trigger) attacks dissociate by scale? This is the load-bearing falsifier for the project's central claim — must hold for the cross-mapping thesis to stand.
- Activation-level interp via nnsight on the LoRA-adapted 405B (the original motivation for running locally rather than on Tinker).
