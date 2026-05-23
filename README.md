# InterpCopy

Reproduction and mechanistic-interpretability scaffolding for studying
finetune-induced verbatim recall in the Llama-3.1 model family (8B, 70B, 405B).
The repository covers three pieces of the workflow:

1. **LoRA finetuning** on torchtune via SLURM (8B/70B/405B).
2. **Batched inference and memorization eval** on the finetuned and baseline
   models via vLLM.
3. **Mechanistic probing** of base vs instruct activations with `nnsight`.

The behavioral methodology mirrors *Alignment Whack-a-Mole* (Liu et al. 2026,
[arXiv:2603.20957](https://arxiv.org/abs/2603.20957)).

## Environment

Tested with Python 3.10, CUDA 12.8, H200 GPUs.

```bash
python -m venv ~/envs/tt
source ~/envs/tt/bin/activate
pip install -e .                    # core deps from pyproject.toml
pip install nnsight matplotlib      # only needed for the interp script
```

Copy `.env.example` to `.env` and fill in `HF_TOKEN` and any other required
secrets before submitting jobs.

## LoRA finetuning

Each scale has a SLURM script that drives the torchtune LoRA recipe. The 405B
path goes through a DCP conversion step (HF safetensors → DCP shards) first.

```bash
# 8B / 70B: HF-format weights work directly
sbatch sbatch/train.sbatch              # 8B smoke
sbatch sbatch/finetune_405b.sbatch      # 405B (after HF→DCP)

# 405B HF→DCP one-time conversion
sbatch sbatch/convert_hf_to_dcp.sbatch
```

LoRA hyperparameters live in `configs/{8b,70b,405b}_lora.yaml`. Adapters land
in `checkpoints/llama3_1-{8b,70b,405b}-lora/epoch_{0,1,2}/`.

## Inference and memorization eval

`src/interpcopy/generate.py` is a vLLM driver that supports both finetuned
adapters and base models, with `--prompt_mode {chat,completion}` to switch
between Instruct-style chat prompts and base-model completion prompts.

```bash
# Instruct baseline
sbatch sbatch/gen_{8b,70b,405b}_base.sbatch

# Pure pretraining base (no Instruct, completion-mode prompts)
sbatch sbatch/gen_{8b,70b,405b}_base_pure.sbatch

# LoRA-finetuned
sbatch sbatch/gen_{8b,70b,405b}_lora.sbatch
```

After generation, run the memorization eval:

```bash
python scripts/run_memeval.py \
  --generations outputs/<run>.json \
  --results outputs/memeval/<run>_results.json
```

The eval depends on the metric implementations in the *Alignment Whack-a-Mole*
code release; set `--eval_repo` to point at that checkout.

## Mechanistic probing (nnsight)

`scripts/exp01_layer_diff_base_vs_instruct.py` captures per-layer residual
streams from a base model and its instruct counterpart on matched prompts,
then plots the per-layer cosine similarity. Two prompt families (semantic
summary vs literal prefix) let you see whether alignment intervenes
differently for different prompt surface forms.

```bash
srun --partition=debug-h200x4 --nodes=1 --ntasks=1 --gres=gpu:h200:1 \
     --cpus-per-task=8 --mem=128G --time=0:30:00 \
     bash -c 'source ~/envs/tt/bin/activate && \
              python scripts/exp01_layer_diff_base_vs_instruct.py'
```

Outputs land in `outputs/interp/` as a `.npz` of raw cosines and a `.png`
plot.

## Layout

```
configs/        torchtune LoRA configs per scale
recipes_dcp/    DCP-aware LoRA finetune recipe
sbatch/         SLURM job scripts (train / convert / inference per scale)
scripts/        eval, model conversion, env setup, interp experiments
src/interpcopy/ small package: dataset adapter, generate.py, eval.py
```

## Notes

- The cluster paths in the sbatch scripts (`/lustre/nvwulf/...`) and the
  account/partition flags are specific to the system we run on; adapt them
  for your cluster.
- vLLM 0.8.5 has a cudagraph-init deadlock on 405B + TP=8 that we work
  around with `enforce_eager=True` in `src/interpcopy/generate.py`. See the
  inline comment there.
