"""Run the Alignment Whack-a-Mole memorization eval on our LoRA generations.

Handles the small schema mismatch in-memory:
  - our `paragraph_id`/`paragraph_text` -> eval's `excerpt_id`/`excerpt_text`
  - eval doesn't read `generation_num`, so we don't need to add it

The eval needs two inputs (test_book + generation_file) but both schemas
overlap — our generation file already carries the per-excerpt metadata inline
— so we pass the same list to both _build_book_index() and the metric fns.

Usage:
    python scripts/run_memeval.py \
        --generations outputs/lora_405b_handmaids_tale_generations.json \
        --results outputs/memeval/lora_405b_handmaids_tale_results.json
"""
import argparse
import json
import sys
from pathlib import Path

EVAL_REPO_DEFAULT = "/lustre/nvwulf/projects/ChakrabartyGroup-nvwulf/Alignment-Whack-a-Mole-Code"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True,
                    help="Our LoRA generation JSON (with paragraph_id/paragraph_text keys).")
    ap.add_argument("--results", required=True,
                    help="Where to write the structured results JSON.")
    ap.add_argument("--eval_repo", default=EVAL_REPO_DEFAULT,
                    help="Path to the cloned Alignment-Whack-a-Mole-Code repo.")
    ap.add_argument("--k", type=int, default=5,
                    help="Minimum match length in words (default 5, per paper §3.1).")
    ap.add_argument("--trim_k", type=int, default=5,
                    help="Instruction m-gram trimming size (default 5).")
    ap.add_argument("--span_threshold", type=int, default=20,
                    help="Word threshold for counting regurgitated spans (default 20).")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.eval_repo) / "evaluation"))
    from memorization_eval_metrics import (
        _build_book_index,
        compute_bmc_and_longest_block,
        compute_longest_regurgitated_span,
        count_regurgitated_spans,
        _extract_span_text_from_book,
    )

    with open(args.generations, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} excerpts from {args.generations}")

    for ex in data:
        if "paragraph_id" in ex and "excerpt_id" not in ex:
            ex["excerpt_id"] = ex.pop("paragraph_id")
        if "paragraph_text" in ex and "excerpt_text" not in ex:
            ex["excerpt_text"] = ex.pop("paragraph_text")

    total_gens = sum(len(ex.get("generations", [])) for ex in data)
    book_index, para_word_spans = _build_book_index(data)
    print(f"Book index: {len(book_index.words):,} word tokens across "
          f"{len(para_word_spans)} excerpts, {total_gens:,} total generations")

    bmc, longest_block, (block_start, block_end) = compute_bmc_and_longest_block(
        book_index, data, k=args.k, trim_k=args.trim_k,
    )
    block_text = None
    if longest_block > 0:
        block_text = _extract_span_text_from_book(
            data, para_word_spans, block_start, block_end
        )

    longest_regurg, regurg_span_text, regurg_gen_text = compute_longest_regurgitated_span(
        data, k=args.k,
    )

    num_spans = count_regurgitated_spans(
        data, k=args.k, span_threshold=args.span_threshold,
    )

    results = {
        "bmc_score": bmc,
        "longest_memorized_block_words": longest_block,
        "longest_memorized_block_text": block_text,
        "longest_regurgitated_span_words": longest_regurg,
        "longest_regurgitated_span_text": regurg_span_text,
        "longest_regurgitated_span_generation_text": regurg_gen_text,
        "num_regurgitated_spans": num_spans,
        "params": {
            "k": args.k,
            "trim_k": args.trim_k,
            "span_threshold": args.span_threshold,
        },
        "inputs": {
            "generations": str(Path(args.generations).resolve()),
            "eval_repo": args.eval_repo,
            "num_excerpts": len(data),
            "num_generations": total_gens,
        },
    }

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  BMC@{args.k}:                        {bmc * 100:.2f}%")
    print(f"  Longest Memorized Block:        {longest_block} words")
    print(f"  Longest Regurgitated Span:      {longest_regurg} words")
    print(f"  # Regurgitated Spans (>{args.span_threshold}w): {num_spans}")
    print()
    print(f"Full results -> {out}")


if __name__ == "__main__":
    main()
