# Llama-3.1 405B LoRA Finetune on Empire AI Alpha

LoRA finetune of Meta Llama-3.1-405B on the Empire AI Alpha cluster, SUNY allocation.

## Cluster context

- **Partition / account / QoS**: `suny` (user `xliu1`)
- **Caps**: 96 concurrent GPUs, 30 running jobs, 200 submitted, 7-day walltime per job
- **Hardware available to us**:
  - 17 nodes with 8× H100 80GB (`alphagpu01–18`)
  - 6 nodes with 8× H200 (`alphagpu19–24`)
  - Each node: 96 CPUs, ~1.9 TB RAM, 8× 400 Gb/s IB (ConnectX-7), NDR rail-aligned fabric
- **Dev/test node**: `alphagpu03` on `scc` partition (lighter contention for small jobs)
- **Scheduling**: fair-share, QoS-based preemption (no preemption risk under `suny` QoS today)

## Storage layout

| Path | Use | FS |
|---|---|---|
| `~/llama-405b-lora/` (this repo) | Code, configs, sbatch scripts | VAST NFS (home) |
| `/mnt/lustre/suny/xliu1/llama-405b-lora/hf_cache/` | HuggingFace weights, tokenizer | DDN Lustre |
| `/mnt/lustre/suny/xliu1/llama-405b-lora/datasets/` | Training / eval datasets | DDN Lustre |
| `/mnt/lustre/suny/xliu1/llama-405b-lora/checkpoints/` | Training checkpoints | DDN Lustre |
| `/mnt/lustre/suny/xliu1/llama-405b-lora/logs/` | Slurm stdout/stderr, wandb cache | DDN Lustre |
| `/dev/shm/` | Per-job high-IOPS scratch (counts against `--mem`) | RAM |

Never write checkpoints or large files to home — it's tuned for metadata, not throughput.

## Framework choice

**Proposed: [`torchtune`](https://github.com/pytorch/torchtune)** — Meta's official recipe library.
- Ships a validated `llama3_1/405B_lora` recipe with FSDP2 + NF4 base weights + bf16 LoRA
- First-party Meta support for Llama 3.1 405B
- Configs are plain YAML, easy to swap model size for the scale-up ladder

Alternatives we're explicitly not picking (for now):
- `axolotl` — flexible but more config surface area; overkill for one recipe
- HF `peft` + `accelerate` FSDP — works but we'd be wiring FSDP2 + activation-checkpoint ourselves

If the proposed path hits a wall we re-evaluate.

## Scale-up ladder

Same `train.py`, same sbatch template, just swap the config and the `--gres`. Don't touch 405B until every earlier rung works end-to-end.

| Stage | Model | Resources | What we verify |
|------:|---|---|---|
| 0 | Llama-3.2-1B | 1× H100 interactive (`salloc`) | tokenizer + chat template, dataset loader, LoRA attaches, one step runs, ckpt saves |
| 1 | Llama-3.1-8B | 1× H100 | loss decreases, eval loop, wandb logging, resume-from-checkpoint |
| 2 | Llama-3.1-70B | 8× H100 single-node, FSDP2 | multi-GPU FSDP, NCCL perf, activation checkpointing, grad clipping |
| 3 | Llama-3.1-405B | 8× H200 (or 2×8 H100) | real training run |

Stage 0 MUST use the final tokenizer and chat template — catches format mismatches that would otherwise only surface at Stage 3.

## Robustness

Every training job must:
- Save a checkpoint at least every ~30–60 min
- Handle `SIGUSR1` to flush a checkpoint before walltime hits (`#SBATCH --signal=B:USR1@300`)
- Resume cleanly from the latest checkpoint on re-submission

## Environment

```bash
module load Python/3.10.15 CUDA/13.1
python -m venv ~/envs/tt
source ~/envs/tt/bin/activate
pip install --upgrade pip
pip install torch torchtune torchao wandb
```

Always export before running anything that touches HF:
```bash
export HF_HOME=/mnt/lustre/suny/xliu1/llama-405b-lora/hf_cache
```

## Job submission flow

- **Interactive (Stages 0–1)**:
  ```bash
  salloc -p suny -A suny -q suny \
    --gres=gpu:nvidia_h100_80gb_hbm3:1 \
    --cpus-per-task=12 --mem=128G --time=4:00:00
  srun --pty bash
  ```
- **Batch (Stages 2–3)**: `sbatch train.sbatch CONFIG=configs/70b.yaml` (skeleton TBD)

## Repo layout (planned)

```
llama-405b-lora/
├── README.md
├── configs/
│   ├── 1b_lora.yaml
│   ├── 8b_lora.yaml
│   ├── 70b_lora.yaml
│   └── 405b_lora.yaml
├── scripts/
│   ├── setup_env.sh
│   ├── download_weights.sh
│   └── launch_interactive.sh
├── sbatch/
│   ├── train.sbatch
│   └── eval.sbatch
├── src/
│   └── train.py
└── .gitignore
```

## TODO

- [ ] Decide: torchtune vs axolotl (default: torchtune)
- [ ] Pick dataset (instruction-tune format, or task-specific?)
- [ ] Create venv + install torchtune
- [ ] Pre-download Llama-3.2-1B for Stage 0 sanity check
- [ ] Write `train.sbatch` with USR1 checkpoint handler
- [ ] Stage 0 end-to-end pass on `scc` partition
- [ ] Stage 1 on 1× H100 suny
- [ ] Stage 2 on 8× H100 single node
- [ ] Pre-download Llama-3.1-405B weights to Lustre
- [ ] Stage 3 dry-run (1 step, resume, USR1 ckpt) before real run
- [ ] Push repo to GitHub

## Open questions (to answer before Stage 3)

- What dataset / task are we finetuning for? (decides config, rank, LR, steps)
- Bf16 LoRA on 8× H200, or QLoRA (NF4) on 8× H100? Quality vs cost tradeoff.
- How many epochs? Rough token budget?
- Which wandb project / entity?
