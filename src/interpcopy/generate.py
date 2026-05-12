"""Batch generation for the COLM evaluation set.

Reads a JSON of paragraph records, generates N completions per paragraph, and
writes a JSON in the COLM paper's format ready for bmc@5 eval:

    {
      ...original fields (paragraph_id, paragraph_text, instruction, ...),
      "generations": [{"generated_text": "..."}, ...]
    }

Uses vLLM for fast batched generation. Works for both the base model
(no `--adapter`) and a finetuned LoRA (`--adapter path/to/epoch_N`).

Resume behavior: if the output file already exists, any paragraph whose
`generations` field is non-empty is skipped. Crash-safe via periodic flushes.

Example (base):
    python -m interpcopy.generate \\
        --model $HF_HOME/models/Llama-3.1-405B-Instruct \\
        --input datasets/output_Margaret_Atwood_-_The_Handmaids_Tale.json \\
        --output outputs/base_405b_generations.json \\
        --n 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Base model path (HF format dir) or HF id")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory (omit for base model)")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON (list of paragraph records)")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--n", type=int, default=100, help="Completions per paragraph (paper default: 100)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (paper default: 1.0)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (paper uses temp=1.0 with no explicit top-p)")
    parser.add_argument("--max_tokens_factor", type=float, default=1.8,
                        help="max_new_tokens = int(word_count * factor). Default 1.8 = word_count * 1.3 tok/word + 40%% buffer")
    parser.add_argument("--max_tokens_floor", type=int, default=256, help="Minimum max_new_tokens")
    parser.add_argument("--max_tokens_ceil", type=int, default=1500, help="Maximum max_new_tokens")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Prompt + generation budget")
    parser.add_argument("--save_every", type=int, default=5, help="Flush output JSON every N paragraphs")
    parser.add_argument("--lora_rank", type=int, default=32, help="Must match adapter's rank when using --adapter")
    args = parser.parse_args()

    # Import vllm lazily so --help works without it installed.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # vLLM 0.8.5 hardcodes a 40s execute_model RPC timeout
    # (vllm/v1/executor/multiproc_executor.py:EXECUTE_MODEL_TIMEOUT_S).
    # The first call with --adapter lazy-loads the LoRA on each TP worker
    # inside that RPC, which blows the 40s budget at 405B/TP=8.
    # Upstream fix is PR #19544 (vLLM 0.9+); until we upgrade, raise it.
    import vllm.v1.executor.multiproc_executor as _vllm_mpe
    _vllm_mpe.EXECUTE_MODEL_TIMEOUT_S = max(_vllm_mpe.EXECUTE_MODEL_TIMEOUT_S, 1800)

    records = json.loads(args.input.read_text())
    print(f"loaded {len(records)} paragraphs from {args.input}", flush=True)

    # Resume: anything already done in --output keeps its generations.
    results_by_id: dict = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        for r in existing:
            results_by_id[r["paragraph_id"]] = r
        done = sum(1 for r in existing if r.get("generations"))
        print(f"resuming — {done} paragraphs already complete", flush=True)
    todo = [r for r in records if not results_by_id.get(r["paragraph_id"], {}).get("generations")]
    print(f"{len(todo)} paragraphs to generate", flush=True)

    if not todo:
        print("nothing to do", flush=True)
        return

    # Init vLLM once, reuse across all paragraphs.
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=False,
    )
    if args.adapter:
        llm_kwargs.update(enable_lora=True, max_loras=1, max_lora_rank=args.lora_rank)
    print(f"initializing vLLM: {llm_kwargs}", flush=True)
    llm = LLM(**llm_kwargs)
    lora_request = LoRARequest("adapter", 1, str(args.adapter)) if args.adapter else None

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        # Write in input order.
        out_list = [results_by_id.get(r["paragraph_id"], r) for r in records]
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(out_list, indent=2, ensure_ascii=False))
        tmp.replace(args.output)

    start = time.time()
    for i, rec in enumerate(todo):
        wc = int(rec.get("word_count", 500))
        max_tokens = max(args.max_tokens_floor, min(args.max_tokens_ceil, int(wc * args.max_tokens_factor)))

        messages = [{"role": "user", "content": rec["instruction"]}]
        sampling = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max_tokens,
        )

        t0 = time.time()
        outputs = llm.chat(messages=messages, sampling_params=sampling, lora_request=lora_request)
        elapsed = time.time() - t0

        completions = [{"generated_text": c.text} for c in outputs[0].outputs]
        results_by_id[rec["paragraph_id"]] = {**rec, "generations": completions}

        done = sum(1 for r in results_by_id.values() if r.get("generations"))
        print(
            f"[{done}/{len(records)}] {rec['paragraph_id']}  "
            f"max_new={max_tokens}  n={args.n}  took={elapsed:.1f}s  "
            f"eta={((time.time()-start)/(i+1))*(len(todo)-i-1)/60:.1f}min",
            flush=True,
        )

        if (i + 1) % args.save_every == 0:
            flush()

    flush()
    print(f"done. wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
