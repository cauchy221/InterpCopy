"""Quick statistical sanity check on a LoRA adapter safetensors file.

Loads every tensor on CPU, reports NaN/Inf presence, magnitude distribution,
and a representative sample of layer-level stats. Catches the obvious failure
modes (corrupt save, exploded weights, all-zero tensors) but does NOT prove
the adapter produces coherent generations — only inference can do that.

Usage:
    python scripts/_diag_adapter_sanity.py <path/to/adapter_model.safetensors>
"""
import statistics as st
import sys
import torch
from safetensors import safe_open

path = sys.argv[1]
print(f"=== Sanity check: {path} ===")

with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
    print(f"Total tensors: {len(keys)}")

    lora_a_keys = [k for k in keys if "lora_a" in k.lower()]
    lora_b_keys = [k for k in keys if "lora_b" in k.lower()]
    other_keys = [k for k in keys if k not in lora_a_keys and k not in lora_b_keys]
    print(f"LoRA A tensors: {len(lora_a_keys)}  |  LoRA B tensors: {len(lora_b_keys)}  |  Other: {len(other_keys)}")
    print(f"Sample keys: {keys[:3]}")
    print()

    has_nan, has_inf = False, False
    abs_means, stds = [], []
    all_zero_count, near_zero_count, huge_count = 0, 0, 0
    dtypes = set()
    sample_rows = []
    n_scanned = 0
    sample_every = max(1, len(keys) // 8)

    for i, k in enumerate(keys):
        t = f.get_tensor(k)
        dtypes.add(str(t.dtype))
        t32 = t.float()
        if torch.isnan(t32).any().item():
            has_nan = True
        if torch.isinf(t32).any().item():
            has_inf = True
        am = t32.abs().mean().item()
        sd = t32.std().item() if t32.numel() > 1 else 0.0
        abs_means.append(am)
        stds.append(sd)
        if am == 0.0:
            all_zero_count += 1
        if am < 1e-8:
            near_zero_count += 1
        if am > 1.0:
            huge_count += 1
        if i < 4 or i % sample_every == 0:
            sample_rows.append((k, tuple(t.shape), str(t.dtype), am, sd,
                                t32.min().item(), t32.max().item()))
        n_scanned += 1

print(f"Tensors scanned: {n_scanned}")
print(f"dtypes present:  {dtypes}")
print(f"NaN anywhere:    {has_nan}    Inf anywhere: {has_inf}")
print(f"All-zero tensors: {all_zero_count}   Near-zero (<1e-8): {near_zero_count}   Huge (|mean|>1.0): {huge_count}")
print(f"|mean(abs)| across tensors:  median={st.median(abs_means):.4e}  min={min(abs_means):.4e}  max={max(abs_means):.4e}")
print(f"|std|       across tensors:  median={st.median(stds):.4e}  min={min(stds):.4e}  max={max(stds):.4e}")
print()
print(f"{'name':70s} {'shape':22s} {'dtype':12s} {'meanabs':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
for k, sh, dt, am, sd, mn, mx in sample_rows:
    print(f"{k[:70]:70s} {str(sh):22s} {dt:12s} {am:10.3e} {sd:10.3e} {mn:10.3e} {mx:10.3e}")
