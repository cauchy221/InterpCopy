"""Feature-level suppression: same SAE on base vs Instruct activations.

Extends exp02 by running the Llama-Scope base SAE on Llama-3.1-8B-Instruct
activations as well, so we can compare per-feature activations between the
two models on the same prompts. Methodological note: the SAE was trained on
base-model activations only, so its reconstruction quality on Instruct will
be lower — but the feature IDs remain comparable, which is what's needed to
ask whether features that fire selectively for a given content type in the
base model fire less in the Instruct model.

Run on a single H200:

    srun --partition=debug-h200x4 --nodes=1 --ntasks=1 --gres=gpu:h200:1 \\
        --cpus-per-task=8 --mem=128G --time=0:30:00 \\
        bash -c 'source ~/envs/tt/bin/activate && \\
                 python scripts/exp03_atwood_features_base_vs_instruct.py'
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
INSTRUCT_MODEL = REPO / "hf_cache" / "models" / "Llama-3.1-8B-Instruct"
OUT_DIR = REPO / "outputs" / "interp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAE_RELEASE = "llama_scope_lxr_32x"
SAE_LAYER = 19
SAE_ID = f"l{SAE_LAYER}r_32x"

PARAGRAPH_IDS = ["p_id1", "p_id64", "p_id101"]
TOP_K = 30  # how many top base-Atwood-selective features to track on Instruct


def build_prompts(records):
    by_id = {r["paragraph_id"]: r for r in records}
    atwood = [
        f"The following is a passage by {by_id[pid]['author_name']}. "
        f"The passage describes: {by_id[pid]['detail']}\n\nPassage:\n"
        for pid in PARAGRAPH_IDS
    ]
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
    for name, fn in [("model.model.layers", lambda m: m.model.layers),
                     ("model.layers", lambda m: m.layers)]:
        try:
            layers = fn(model)
            print(f"  resolved transformer blocks via {name!r}, len={len(layers)}")
            return layers
        except (AttributeError, TypeError):
            continue
    raise RuntimeError("could not find transformer blocks on model")


def capture_layer_residuals(model_path: Path, prompts, layer_idx, dtype=torch.bfloat16):
    print(f"Loading {model_path.name} ...")
    model = LanguageModel(str(model_path), device_map="cuda", torch_dtype=dtype, dispatch=True)
    layers = _resolve_layers(model)
    residuals = []
    for i, prompt in enumerate(prompts):
        saved = []
        with model.trace(prompt):
            saved.append(layers[layer_idx].output[0].save())
        t = saved[0] if isinstance(saved[0], torch.Tensor) else getattr(saved[0], "value", saved[0])
        vec = t[0, -1, :] if t.dim() == 3 else t[-1, :]
        residuals.append(vec.detach().cpu().float())
        print(f"  prompt {i}: residual shape {tuple(vec.shape)}")
    del model
    torch.cuda.empty_cache()
    return torch.stack(residuals)


def main():
    records = json.loads(DATASET.read_text())
    atwood_prompts, control_prompts = build_prompts(records)
    all_prompts = atwood_prompts + control_prompts
    n_atwood = len(atwood_prompts)

    print(f"=== Capturing layer-{SAE_LAYER} residuals: BASE ===")
    base_resid = capture_layer_residuals(BASE_MODEL, all_prompts, SAE_LAYER)
    print(f"  shape {tuple(base_resid.shape)}")

    print(f"\n=== Capturing layer-{SAE_LAYER} residuals: INSTRUCT ===")
    instruct_resid = capture_layer_residuals(INSTRUCT_MODEL, all_prompts, SAE_LAYER)
    print(f"  shape {tuple(instruct_resid.shape)}")

    print(f"\n=== Loading SAE: {SAE_RELEASE} / {SAE_ID} ===")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda", dtype="float32")
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()

    print("\n=== Encoding both ===")
    base_feats = sae.encode(base_resid.cuda()).detach().cpu().float().numpy()
    inst_feats = sae.encode(instruct_resid.cuda()).detach().cpu().float().numpy()
    print(f"  base feature acts shape {base_feats.shape}; instruct shape {inst_feats.shape}")

    # Per-feature means
    base_atw = base_feats[:n_atwood].mean(axis=0)
    base_ctl = base_feats[n_atwood:].mean(axis=0)
    inst_atw = inst_feats[:n_atwood].mean(axis=0)
    inst_ctl = inst_feats[n_atwood:].mean(axis=0)

    base_delta = base_atw - base_ctl  # Atwood-selectivity in base
    inst_delta = inst_atw - inst_ctl  # Atwood-selectivity in Instruct

    top_idx = np.argsort(-base_delta)[:TOP_K]  # base's top Atwood-selective features

    # Suppression on Atwood prompts: instruct activation as a fraction of base activation.
    # 0 = fully suppressed, 1 = unchanged, >1 = boosted.
    eps = 1e-6
    ratio = inst_atw[top_idx] / np.maximum(base_atw[top_idx], eps)

    print(f"\n=== Top {TOP_K} base-Atwood-selective features and their Instruct activation ===")
    print(f"{'rank':>4}  {'feat':>8}  {'base_atw':>9}  {'inst_atw':>9}  "
          f"{'base_Δ':>8}  {'inst_Δ':>8}  {'inst/base':>9}")
    for rank, fi in enumerate(top_idx):
        print(
            f"{rank+1:>4}  {fi:>8d}  {base_atw[fi]:9.3f}  {inst_atw[fi]:9.3f}  "
            f"{base_delta[fi]:+8.3f}  {inst_delta[fi]:+8.3f}  {ratio[rank]:9.3f}"
        )

    n_suppressed = int(np.sum(ratio < 0.5))
    n_silenced = int(np.sum(inst_atw[top_idx] == 0))
    print(f"\nSummary on top {TOP_K}: {n_silenced} features go fully silent in Instruct "
          f"(inst_atw == 0); {n_suppressed} have <50% of base activation.")

    out_npz = OUT_DIR / "exp03_base_vs_instruct.npz"
    np.savez(
        out_npz,
        base_atw=base_atw, base_ctl=base_ctl,
        inst_atw=inst_atw, inst_ctl=inst_ctl,
        base_delta=base_delta, inst_delta=inst_delta,
        top_idx=top_idx, ratio=ratio,
        layer=SAE_LAYER, sae_release=SAE_RELEASE, sae_id=SAE_ID,
    )
    print(f"\nsaved {out_npz}")

    # Plot: scatter of base vs Instruct activation on Atwood prompts, top features.
    # Diagonal = no suppression; below diagonal = suppression.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: scatter
    ax = axes[0]
    ax.scatter(base_atw[top_idx], inst_atw[top_idx], s=40, alpha=0.75, color="C3", zorder=3)
    max_v = max(base_atw[top_idx].max(), inst_atw[top_idx].max()) * 1.05
    ax.plot([0, max_v], [0, max_v], "k--", alpha=0.4, lw=1, label="y = x (no change)")
    ax.plot([0, max_v], [0, max_v * 0.5], color="gray", linestyle=":", alpha=0.5,
            label="y = 0.5x (50% suppression)")
    for fi in top_idx[:5]:
        ax.annotate(f"f{fi}", (base_atw[fi], inst_atw[fi]),
                    xytext=(3, 3), textcoords="offset points", fontsize=7, color="C3")
    ax.set_xlabel("Base activation on Atwood prompts")
    ax.set_ylabel("Instruct activation on Atwood prompts")
    ax.set_title(f"Top {TOP_K} base-Atwood-selective features\n(same Llama-Scope SAE applied to both models)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, max_v)
    ax.set_ylim(0, max_v)

    # Right: bar plot of base vs Instruct activations side-by-side
    ax = axes[1]
    ranks = np.arange(TOP_K)
    width = 0.4
    ax.barh(ranks - width / 2, base_atw[top_idx], height=width, color="C0", label="Base")
    ax.barh(ranks + width / 2, inst_atw[top_idx], height=width, color="C3", label="Instruct")
    ax.set_yticks(ranks)
    ax.set_yticklabels([f"f{fi}" for fi in top_idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Activation on Atwood prompts (layer {SAE_LAYER})")
    ax.set_title("Per-feature suppression by alignment")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Llama-3.1-8B base vs Instruct: feature-level alignment suppression",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    plot_path = OUT_DIR / "exp03_base_vs_instruct.png"
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
