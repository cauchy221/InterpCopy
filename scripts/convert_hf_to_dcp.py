"""Convert an HF safetensors checkpoint to PyTorch DCP format for fast distributed loading.

Single-process conversion. Uses torchtune's FullModelHFCheckpointer to load + remap
HF keys into torchtune format (incl. Q/K rotary-format permutation), then writes the
state dict as a PyTorch DCP. After conversion, FSDP training can populate each rank's
shard from DCP in parallel — sidestepping the rank-0-read + per-tensor-NCCL-broadcast
pattern that hangs at 405B scale.

Memory budget: holds the full bf16 state dict in CPU RAM during the DCP write
(~812 GB for Llama-3.1-405B). Size the slurm `--mem` accordingly.

Usage:
    python convert_hf_to_dcp.py <hf_dir> <dcp_out_dir> [--model-type LLAMA3] [--num-shards 191]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemWriter, save
from torchtune import training
from torchtune.training.checkpointing import FullModelHFCheckpointer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("hf_dir", help="HF checkpoint directory (containing config.json and model-*-of-*.safetensors)")
    p.add_argument("dcp_out_dir", help="Output directory for the DCP checkpoint")
    p.add_argument("--model-type", default="LLAMA3", help="torchtune model type (default: LLAMA3)")
    p.add_argument("--num-shards", type=int, default=191, help="Total HF shards (default: 191 for 405B)")
    p.add_argument("--thread-count", type=int, default=16, help="DCP writer threads")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hf_dir = Path(args.hf_dir)
    dcp_dir = Path(args.dcp_out_dir)
    if not hf_dir.is_dir():
        sys.exit(f"hf_dir does not exist: {hf_dir}")
    if not (hf_dir / "config.json").exists():
        sys.exit(f"missing config.json in {hf_dir}")
    dcp_dir.mkdir(parents=True, exist_ok=True)

    # DCP save() needs an initialized process group, even at world_size=1.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group("gloo", rank=0, world_size=1)

    # FullModelHFCheckpointer requires an output_dir for the *save* path; we never save through it,
    # but the ctor mkdirs it. Use a throwaway tempdir so we don't pollute the real output dir.
    with tempfile.TemporaryDirectory(prefix="hf2dcp_unused_") as scratch:
        print(
            f"[convert] loading HF checkpoint via torchtune's FullModelHFCheckpointer\n"
            f"          src: {hf_dir}\n"
            f"          model_type: {args.model_type}, num_shards: {args.num_shards}",
            flush=True,
        )
        load_t0 = time.perf_counter()
        ckpt = FullModelHFCheckpointer(
            checkpoint_dir=str(hf_dir),
            checkpoint_files={
                "filename_format": "model-{}-of-{}.safetensors",
                "max_filename": f"{args.num_shards:05d}",
            },
            model_type=args.model_type,
            output_dir=scratch,
        )
        loaded = ckpt.load_checkpoint()
        load_secs = time.perf_counter() - load_t0
        model_sd = loaded[training.MODEL_KEY]
        n_keys = len(model_sd)
        n_bytes = sum(t.numel() * t.element_size() for t in model_sd.values())
        print(
            f"[convert] HF load + key remap: {load_secs:.1f}s, "
            f"{n_keys} keys, {n_bytes / 1e9:.1f} GB",
            flush=True,
        )

    # Drop the wrapping dict and any references the checkpointer is holding so peak
    # memory during DCP write is just the state dict.
    del ckpt, loaded
    gc.collect()

    print(
        f"[convert] writing DCP to {dcp_dir} (thread_count={args.thread_count}) ...",
        flush=True,
    )
    save_t0 = time.perf_counter()
    save(
        state_dict={training.MODEL_KEY: model_sd},
        storage_writer=FileSystemWriter(
            str(dcp_dir),
            thread_count=args.thread_count,
            single_file_per_rank=False,
            sync_files=False,
        ),
    )
    print(f"[convert] DCP write: {time.perf_counter() - save_t0:.1f}s", flush=True)

    dist.destroy_process_group()
    print("[convert] done", flush=True)


if __name__ == "__main__":
    main()
