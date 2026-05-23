"""Layer-resolved activation diff between a base LM and its instruct counterpart.

For each prompt we capture the last-token residual stream at every transformer
block on both models and report the per-layer cos(base, instruct). Two prompt
families are compared: a semantic-summary trigger and a literal-prefix trigger.
A divergence band that differs between the two families is evidence that
alignment is keyed to surface form rather than uniformly suppressing the
underlying representation.

Run on a single H200:

    srun --partition=debug-h200x4 --nodes=1 --ntasks=1 --gres=gpu:h200:1 \\
        --cpus-per-task=8 --mem=128G --time=0:30:00 \\
        bash -c 'source ~/envs/tt/bin/activate && \\
                 python scripts/exp01_layer_diff_base_vs_instruct.py'
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from nnsight import LanguageModel

REPO = Path("/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy")
DATASET = REPO / "datasets" / "output_Margaret_Atwood_-_The_Handmaids_Tale.json"
BASE_MODEL = REPO / "hf_cache" / "models" / "Llama-3.1-8B"
INSTRUCT_MODEL = REPO / "hf_cache" / "models" / "Llama-3.1-8B-Instruct"
OUT_DIR = REPO / "outputs" / "interp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAGRAPH_IDS = ["p_id1", "p_id64", "p_id101"]  # iconic / mid-novel / non-iconic
LITERAL_PREFIX_WORDS = 25


def build_prompts(records):
    by_id = {r["paragraph_id"]: r for r in records}
    sem_prompts, lit_prompts, labels = [], [], []
    for pid in PARAGRAPH_IDS:
        rec = by_id[pid]
        sem = (
            f"The following is a passage by {rec['author_name']}. "
            f"The passage describes: {rec['detail']}\n\nPassage:\n"
        )
        lit_words = rec["paragraph_text"].split()[:LITERAL_PREFIX_WORDS]
        lit = " ".join(lit_words)
        sem_prompts.append(sem)
        lit_prompts.append(lit)
        labels.append(pid)
    return sem_prompts, lit_prompts, labels


def _resolve_layers(model):
    """Try a few common paths to the transformer-blocks ModuleList in the wrapped HF model."""
    candidates = [
        ("model.model.layers", lambda m: m.model.layers),
        ("model.layers", lambda m: m.layers),
        ("model.transformer.h", lambda m: m.transformer.h),
    ]
    for name, fn in candidates:
        try:
            layers = fn(model)
            n = len(layers)
            print(f"  resolved transformer blocks via {name!r}, len={n}")
            return layers, name
        except (AttributeError, TypeError):
            continue
    raise RuntimeError("could not find transformer blocks on model")


def capture_residuals(model_path: Path, prompts: list[str], dtype=torch.bfloat16):
    """Returns list[prompt_index] -> Tensor[num_layers, hidden_dim] (last token, fp32 cpu)."""
    print(f"Loading {model_path.name} ...")
    model = LanguageModel(str(model_path), device_map="cuda", torch_dtype=dtype, dispatch=True)
    num_layers = model.config.num_hidden_layers
    print(f"  {num_layers} layers, hidden_size={model.config.hidden_size}")
    layers, layers_path = _resolve_layers(model)

    results = []
    for i, prompt in enumerate(prompts):
        saved = []
        try:
            with model.trace(prompt):
                for L in range(num_layers):
                    saved.append(layers[L].output[0].save())
            print(f"  prompt {i}: collected {len(saved)} proxies; sample type={type(saved[0]).__name__}")
        except Exception as e:
            print(f"  prompt {i}: trace failed: {type(e).__name__}: {e}")
            raise

        per_layer = []
        for L, s in enumerate(saved):
            t = s if isinstance(s, torch.Tensor) else getattr(s, "value", s)
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(f"saved proxy did not materialize to a tensor: {type(t).__name__}")
            if L == 0 and i == 0:
                print(f"  (debug) layer-0 residual tensor shape={tuple(t.shape)}, dtype={t.dtype}")
            # nnsight may return [batch, seq, hidden] or already-squeezed [seq, hidden]
            if t.dim() == 3:
                vec = t[0, -1, :]
            elif t.dim() == 2:
                vec = t[-1, :]
            else:
                raise RuntimeError(f"unexpected residual tensor dim {t.dim()}, shape {tuple(t.shape)}")
            per_layer.append(vec.detach().cpu().float())
        results.append(torch.stack(per_layer))  # [num_layers, hidden_dim]
        print(f"  prompt {i}: stack {tuple(results[-1].shape)}")

    del model
    torch.cuda.empty_cache()
    return results


def per_layer_cosine(a_list, b_list):
    """a_list, b_list: lists of [num_layers, hidden_dim] tensors. Returns [num_prompts, num_layers]."""
    out = []
    for a, b in zip(a_list, b_list):
        per_layer = torch.cosine_similarity(a, b, dim=1)
        out.append(per_layer.numpy())
    return np.stack(out)


def main():
    records = json.loads(DATASET.read_text())
    print(f"Loaded {len(records)} paragraphs from {DATASET.name}")

    sem_prompts, lit_prompts, labels = build_prompts(records)
    all_prompts = sem_prompts + lit_prompts
    print(f"Prompts: {len(sem_prompts)} semantic + {len(lit_prompts)} literal = {len(all_prompts)} total")
    for i, p in enumerate(all_prompts):
        kind = "SEM" if i < len(sem_prompts) else "LIT"
        pid = labels[i % len(labels)]
        print(f"  [{kind} {pid}] {p[:80].replace(chr(10), ' / ')!r}...")

    print("\n=== Capturing residuals on BASE ===")
    base_acts = capture_residuals(BASE_MODEL, all_prompts)

    print("\n=== Capturing residuals on INSTRUCT ===")
    instruct_acts = capture_residuals(INSTRUCT_MODEL, all_prompts)

    print("\n=== Computing per-layer cosine similarity ===")
    cosines = per_layer_cosine(base_acts, instruct_acts)
    print(f"  cosines shape {cosines.shape} (prompts x layers)")

    n_sem = len(sem_prompts)
    sem_curve = cosines[:n_sem].mean(axis=0)
    lit_curve = cosines[n_sem:].mean(axis=0)

    out_npz = OUT_DIR / "exp01_cosines.npz"
    np.savez(
        out_npz,
        cosines=cosines,
        sem_curve=sem_curve,
        lit_curve=lit_curve,
        paragraph_ids=np.array(labels),
        sem_prompts=np.array(sem_prompts, dtype=object),
        lit_prompts=np.array(lit_prompts, dtype=object),
    )
    print(f"  saved {out_npz}")

    # Plot
    num_layers = cosines.shape[1]
    layers = np.arange(num_layers)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Lighter per-prompt traces, bold averages
    for i in range(n_sem):
        ax.plot(layers, cosines[i], color="C0", alpha=0.25, lw=1)
    for i in range(n_sem, len(all_prompts)):
        ax.plot(layers, cosines[i], color="C1", alpha=0.25, lw=1)
    ax.plot(layers, sem_curve, color="C0", lw=2.4, marker="o", label="Semantic-trigger (Format A) — mean")
    ax.plot(layers, lit_curve, color="C1", lw=2.4, marker="s", label="Literal-trigger (25-word prefix) — mean")

    ax.set_xlabel("Transformer block index (0 = embeddings, last = pre-unembed)")
    ax.set_ylabel("cos(base, Instruct) — last-token residual stream")
    ax.set_title(
        "Llama-3.1-8B: where alignment changes the residual\n"
        "(lower = more divergent = stronger alignment effect at that layer)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = OUT_DIR / "exp01_layer_diff.png"
    fig.savefig(plot_path, dpi=130)
    print(f"  saved {plot_path}")

    print("\n=== Summary ===")
    print(f"  Semantic-trigger mean cos by layer (min={sem_curve.min():.3f} at layer {sem_curve.argmin()}, "
          f"max={sem_curve.max():.3f} at layer {sem_curve.argmax()})")
    print(f"  Literal-trigger  mean cos by layer (min={lit_curve.min():.3f} at layer {lit_curve.argmin()}, "
          f"max={lit_curve.max():.3f} at layer {lit_curve.argmax()})")


if __name__ == "__main__":
    main()
