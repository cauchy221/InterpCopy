"""Find SAE features that fire selectively on Atwood-style semantic prompts.

We pass paired prompt families through Llama-3.1-8B (base) and encode the
last-token residual stream at a mid-late layer through a pretrained
Llama-Scope SAE. For each feature we report the difference of mean
activation between Atwood prompts and matched controls (same prompt
template, different author/content). The top-ranked features are
candidates for what the model uses to address Atwood-style content at
this layer.

Run on a single H200:

    srun --partition=debug-h200x4 --nodes=1 --ntasks=1 --gres=gpu:h200:1 \\
        --cpus-per-task=8 --mem=128G --time=0:30:00 \\
        bash -c 'source ~/envs/tt/bin/activate && \\
                 python scripts/exp02_atwood_features.py'
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from nnsight import LanguageModel
from sae_lens import SAE

REPO = Path("/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/InterpCopy")
DATASET = REPO / "datasets" / "output_Margaret_Atwood_-_The_Handmaids_Tale.json"
BASE_MODEL = REPO / "hf_cache" / "models" / "Llama-3.1-8B"
OUT_DIR = REPO / "outputs" / "interp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Llama-Scope residual-stream SAE, 32x expansion (~131K features), at layer 19 (~60% depth).
SAE_RELEASE = "llama_scope_lxr_32x"
SAE_LAYER = 19
SAE_ID = f"l{SAE_LAYER}r_32x"

PARAGRAPH_IDS = ["p_id1", "p_id64", "p_id101"]  # iconic / mid-novel / non-iconic
TOP_K = 30


def build_prompts(records):
    """Return (atwood_prompts, control_prompts). Same Format A template, different content."""
    by_id = {r["paragraph_id"]: r for r in records}
    atwood = []
    for pid in PARAGRAPH_IDS:
        rec = by_id[pid]
        atwood.append(
            f"The following is a passage by {rec['author_name']}. "
            f"The passage describes: {rec['detail']}\n\nPassage:\n"
        )
    controls = [
        "The following is a passage by Albert Einstein. The passage describes: "
        "A physicist explains the concept of special relativity, walking through "
        "how observers in different inertial frames disagree about simultaneity "
        "and how time dilates at velocities approaching the speed of light.\n\nPassage:\n",

        "The following is a passage by Julia Child. The passage describes: "
        "A French cook gives step-by-step instructions for preparing beef "
        "bourguignon, including how to brown the meat, deglaze the pan with "
        "red wine, and slowly braise the stew with pearl onions and mushrooms.\n\nPassage:\n",

        "The following is a passage by Steve Wozniak. The passage describes: "
        "An engineer recalls the development of the Apple I computer in a "
        "Silicon Valley garage in 1976, describing the design tradeoffs and "
        "the moment they first booted the prototype.\n\nPassage:\n",
    ]
    return atwood, controls


def _resolve_layers(model):
    """Find the transformer-blocks ModuleList on the wrapped HF model."""
    candidates = [
        ("model.model.layers", lambda m: m.model.layers),
        ("model.layers", lambda m: m.layers),
    ]
    for name, fn in candidates:
        try:
            layers = fn(model)
            print(f"  resolved transformer blocks via {name!r}, len={len(layers)}")
            return layers
        except (AttributeError, TypeError):
            continue
    raise RuntimeError("could not find transformer blocks on model")


def capture_layer_residuals(model_path: Path, prompts: list[str], layer_idx: int,
                            dtype=torch.bfloat16) -> torch.Tensor:
    """Return [n_prompts, hidden_dim] of last-token residuals at `layer_idx`, fp32 cpu."""
    print(f"Loading {model_path.name} ...")
    model = LanguageModel(str(model_path), device_map="cuda", torch_dtype=dtype, dispatch=True)
    print(f"  hidden_size={model.config.hidden_size}, num_layers={model.config.num_hidden_layers}")
    layers = _resolve_layers(model)

    residuals = []
    for i, prompt in enumerate(prompts):
        saved = []
        with model.trace(prompt):
            saved.append(layers[layer_idx].output[0].save())
        t = saved[0] if isinstance(saved[0], torch.Tensor) else getattr(saved[0], "value", saved[0])
        if t.dim() == 3:
            vec = t[0, -1, :]
        elif t.dim() == 2:
            vec = t[-1, :]
        else:
            raise RuntimeError(f"unexpected residual shape {tuple(t.shape)}")
        residuals.append(vec.detach().cpu().float())
        print(f"  prompt {i}: residual shape {tuple(vec.shape)}")

    del model
    torch.cuda.empty_cache()
    return torch.stack(residuals)  # [n_prompts, hidden_dim]


def main():
    records = json.loads(DATASET.read_text())
    print(f"Loaded {len(records)} paragraphs from {DATASET.name}")

    atwood_prompts, control_prompts = build_prompts(records)
    all_prompts = atwood_prompts + control_prompts
    n_atwood, n_control = len(atwood_prompts), len(control_prompts)
    print(f"Prompts: {n_atwood} Atwood + {n_control} control")

    print(f"\n=== Capturing residuals at layer {SAE_LAYER} ===")
    residuals = capture_layer_residuals(BASE_MODEL, all_prompts, SAE_LAYER)
    print(f"  residuals shape {tuple(residuals.shape)}")

    print(f"\n=== Loading SAE: {SAE_RELEASE} / {SAE_ID} ===")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda", dtype="float32")
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    n_features = sae.W_enc.shape[1] if hasattr(sae, "W_enc") else None
    print(f"  SAE loaded; n_features ~ {n_features}")

    print("\n=== Encoding residuals through SAE ===")
    feats = sae.encode(residuals.cuda())  # [n_prompts, n_features]
    feats = feats.detach().cpu().float().numpy()
    print(f"  feature activations shape {feats.shape}")

    atwood_acts = feats[:n_atwood].mean(axis=0)
    control_acts = feats[n_atwood:].mean(axis=0)
    delta = atwood_acts - control_acts

    top_idx = np.argsort(-delta)[:TOP_K]

    print(f"\n=== Top {TOP_K} Atwood-selective features (Δ = mean_atwood − mean_control) ===")
    print(f"{'rank':>4}  {'feat_id':>8}  {'Δ':>8}  {'atwood':>8}  {'control':>8}")
    for rank, fi in enumerate(top_idx):
        print(f"{rank+1:>4}  {fi:>8d}  {delta[fi]:+8.3f}  {atwood_acts[fi]:8.3f}  {control_acts[fi]:8.3f}")

    out_npz = OUT_DIR / "exp02_atwood_features.npz"
    np.savez(
        out_npz,
        atwood_acts=atwood_acts,
        control_acts=control_acts,
        delta=delta,
        top_idx=top_idx,
        all_feats=feats,
        layer=SAE_LAYER,
        sae_release=SAE_RELEASE,
        sae_id=SAE_ID,
        atwood_prompts=np.array(atwood_prompts, dtype=object),
        control_prompts=np.array(control_prompts, dtype=object),
    )
    print(f"\nsaved {out_npz}")

    fig, ax = plt.subplots(figsize=(8.5, max(5.5, 0.22 * TOP_K)))
    ranks = np.arange(TOP_K)
    width = 0.4
    ax.barh(ranks - width / 2, atwood_acts[top_idx], height=width, color="C0", label="Atwood mean")
    ax.barh(ranks + width / 2, control_acts[top_idx], height=width, color="C7", label="Control mean")
    ax.set_yticks(ranks)
    ax.set_yticklabels([f"f{fi}" for fi in top_idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"SAE feature activation at residual layer {SAE_LAYER}")
    ax.set_title(
        f"Llama-3.1-8B base · {SAE_RELEASE}/{SAE_ID}\n"
        f"Top {TOP_K} features by (Atwood − control) selectivity"
    )
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    plot_path = OUT_DIR / "exp02_atwood_features.png"
    fig.savefig(plot_path, dpi=130)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
