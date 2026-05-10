"""LoRA finetune recipe with DCP base-model loading.

A thin wrapper that monkey-patches torchtune so the upstream
`recipes.lora_finetune_distributed.recipe_main` loads the base model from a
PyTorch DCP directory instead of HF safetensors. The DCP load lets each FSDP
rank read only its own shard from disk in parallel, sidestepping the rank-0
read + per-tensor NCCL broadcast pattern that hangs at 405B scale.

Set the env var DCP_CKPT_DIR to enable. When unset, behaviour matches upstream
exactly.

Use:
    DCP_CKPT_DIR=/path/to/dcp \\
        python -m torchtune._cli.tune run --nproc_per_node=8 \\
            recipes_dcp/lora_finetune_distributed_dcp.py \\
            --config configs/405b_lora.yaml

The upstream torchtune `recipes/` package is guarded against import (its
`__init__.py` raises ModuleNotFoundError), so we live in `recipes_dcp/` and
load the upstream recipe via `importlib.util.spec_from_file_location` rather
than `from recipes.lora_finetune_distributed import recipe_main`.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torchtune
import torchtune.training as _tt_training
from torchtune.training import MODEL_KEY
from torchtune.training.checkpointing import _checkpointer as _ckpt_mod


_DCP_ENV_VAR = "DCP_CKPT_DIR"


def _dcp_dir() -> str | None:
    return os.environ.get(_DCP_ENV_VAR)


# --- patch 1: skip the slow HF safetensors read when DCP is configured -------
_orig_hf_load = _ckpt_mod.FullModelHFCheckpointer.load_checkpoint


@functools.wraps(_orig_hf_load)
def _patched_hf_load(self):
    dcp = _dcp_dir()
    if not dcp:
        return _orig_hf_load(self)
    print(
        f"[dcp] {_DCP_ENV_VAR}={dcp} is set; skipping HF safetensors read "
        f"(weights will be populated by DCP load)",
        flush=True,
    )
    self._weight_map = {}  # only used at save time; harmless empty
    return {MODEL_KEY: {}}


_ckpt_mod.FullModelHFCheckpointer.load_checkpoint = _patched_hf_load


# --- patch 2: replace rank-0-broadcast load with DCP parallel load ----------
_orig_load_from_full = _tt_training.load_from_full_model_state_dict


def _materialize_meta_then_dcp_load(model, device, dcp):
    """Materialise meta-device params on ``device``, then populate base weights from DCP.

    `model.to_empty(device=device)` is the only API that reliably allocates the
    FSDP DTensor `_local_tensor` on the destination device for every parameter.
    It is destructive (overwrites already-initialised LoRA weights with garbage),
    so the caller must re-initialise LoRA modules afterwards.
    """
    from torch.distributed.checkpoint import FileSystemReader, load as dcp_load
    from torch.nn.modules.module import _IncompatibleKeys

    print(f"[dcp] materialising tensors on {device}", flush=True)
    model.to_empty(device=device)

    print(f"[dcp] reading {dcp}", flush=True)
    reader = FileSystemReader(dcp)
    metadata = reader.read_metadata()
    saved_keys = {
        fqn[len(MODEL_KEY) + 1 :]
        for fqn in metadata.state_dict_metadata
        if fqn.startswith(f"{MODEL_KEY}.")
    }
    sharded_sd = model.state_dict()
    to_load = {k: v for k, v in sharded_sd.items() if k in saved_keys}
    extra = sorted(saved_keys - set(to_load.keys()))
    if extra:
        print(
            f"[dcp] {len(extra)} DCP keys not in model.state_dict() "
            f"(first 5: {extra[:5]})",
            flush=True,
        )
    print(
        f"[dcp] loading {len(to_load)} keys "
        f"(model has {len(sharded_sd)} total, DCP has {len(saved_keys)})",
        flush=True,
    )
    dcp_load(state_dict={MODEL_KEY: to_load}, storage_reader=reader)
    return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])


@functools.wraps(_orig_load_from_full)
def _patched_load_from_full(model, full_sd, device, strict=False, cpu_offload=False):
    dcp = _dcp_dir()
    if not dcp or full_sd:
        # No DCP configured, OR this call has real data (e.g. LoRA adapter
        # weights when resuming from a previous run). Use the upstream path.
        return _orig_load_from_full(
            model, full_sd, device, strict=strict, cpu_offload=cpu_offload
        )

    # Snapshot LoRA / DoRA modules so we can re-init them after to_empty().
    from torchtune.modules.peft import LoRALinear, DoRALinear  # noqa: WPS433

    lora_modules = [
        m for m in model.modules() if isinstance(m, (LoRALinear, DoRALinear))
    ]

    result = _materialize_meta_then_dcp_load(model, device, dcp)

    # Re-initialise LoRA params and any RoPE buffers that to_empty clobbered.
    # Wrap in `torch.device(device)` so tensors freshly created inside
    # rope_init (notably `torch.arange(...)` for the RoPE freqs) land on
    # `device` instead of CPU. Upstream lora_finetune_distributed.py:497
    # does this via `with self._device:` — without it, the rope `theta` /
    # `cache` buffers end up on CPU and the first forward fails with
    # "Expected all tensors to be on the same device".
    print(
        f"[dcp] re-initialising {len(lora_modules)} LoRA modules "
        "+ any RoPE buffers",
        flush=True,
    )
    with torch.device(device):
        for m in lora_modules:
            m.initialize_parameters()
        for m in model.modules():
            if hasattr(m, "rope_init"):
                m.rope_init()

    return result


_tt_training.load_from_full_model_state_dict = _patched_load_from_full


# --- load upstream recipe (bypassing the recipes/__init__.py guard) ---------
def _load_upstream_recipe_main():
    upstream_path = (
        Path(torchtune.__file__).parent.parent
        / "recipes"
        / "lora_finetune_distributed.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_upstream_lora_finetune_distributed", str(upstream_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load upstream recipe at {upstream_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.recipe_main


recipe_main = _load_upstream_recipe_main()


if __name__ == "__main__":
    sys.exit(recipe_main())
