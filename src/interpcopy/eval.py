"""Memorization metrics from the COLM 2026 paper (Liu et al.).

Ported from the paper's public eval:
  https://github.com/cauchy221/Alignment-Whack-a-Mole-Code/blob/main/evaluation/memorization_eval_metrics.py

Four metrics (Section 3.1):

  1. Book Memorization Coverage bmc@k — Algorithm 1.
     Fraction of book word positions covered by contiguous span matches
     (>= k words) aggregated across all generations, with instruction
     m-gram trimming.

  2. Longest Contiguous Memorized Block (words) — longest True-run in the
     coverage mask.

  3. Longest Contiguous Regurgitated Span (words) — longest single-generation
     verbatim match against its own excerpt/paragraph, no trimming.

  4. Number of Regurgitated Spans > T words — non-overlapping, greedy by
     length.

Field-name compatibility
------------------------
The COLM repo's example uses `excerpt_id` / `excerpt_text`. Our cluster data
uses `paragraph_id` / `paragraph_text`. This port accepts either; it picks
whichever is present per record.
"""

from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_left
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from nltk.tokenize import WordPunctTokenizer, wordpunct_tokenize
from tqdm import tqdm

_WPT = WordPunctTokenizer()
_WORD_RE = re.compile(r"[A-Za-z0-9]")


# ---------------------------------------------------------------------------
# Field-name compatibility shim — supports both `paragraph_*` and `excerpt_*`
# ---------------------------------------------------------------------------

def _get_id(ex: dict) -> str:
    """Return the paragraph/excerpt id, whichever field is present."""
    for key in ("paragraph_id", "excerpt_id"):
        if key in ex:
            return str(ex[key])
    raise KeyError(f"record has no paragraph_id / excerpt_id: keys={list(ex.keys())}")


def _get_text(ex: dict) -> str:
    """Return the paragraph/excerpt text, whichever field is present."""
    for key in ("paragraph_text", "excerpt_text"):
        if key in ex:
            return ex.get(key) or ""
    return ""


def _pid_to_int(ex: dict) -> int:
    """Extract the numeric component from the id string (e.g. 'p_id42' -> 42)."""
    s = _get_id(ex)
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"id must contain a number, got: {s!r}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _word_char_spans(text: str) -> List[Tuple[int, int]]:
    all_spans = list(_WPT.span_tokenize(text or ""))
    all_tokens = _WPT.tokenize(text or "")
    return [(sc, ec) for tok, (sc, ec) in zip(all_tokens, all_spans) if _WORD_RE.search(tok)]


def _tok_words(text: str) -> List[str]:
    return [t.lower() for t in wordpunct_tokenize(text or "") if _WORD_RE.search(t)]


# ---------------------------------------------------------------------------
# Interval utilities
# ---------------------------------------------------------------------------

def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _subtract_from_interval(base: Tuple[int, int], removes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    s, e = base
    clamped = [(max(s, a), min(e, b)) for a, b in removes if not (b <= s or a >= e)]
    rm = _merge_intervals([r for r in clamped if r[0] < r[1]])
    if not rm:
        return [base]
    out, cur = [], s
    for a, b in rm:
        if cur < a:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < e:
        out.append((cur, e))
    return out


# ---------------------------------------------------------------------------
# Instruction m-gram trimming (Algorithm 1 lines 7-9)
# ---------------------------------------------------------------------------

def _kset(words: List[str], k: int):
    if k <= 0:
        return set()
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def _trim_instruction_kgrams(
    gold_words: List[str],
    instr_words: List[str],
    intervals: List[Tuple[int, int]],
    min_length: int,
    k_for_exclusion: int,
) -> List[Tuple[int, int]]:
    all_trimmed: List[Tuple[int, int]] = []
    for raw_iv in intervals:
        s, e = raw_iv
        span_len = e - s
        if k_for_exclusion <= 0 or span_len < k_for_exclusion:
            if span_len >= min_length:
                all_trimmed.append(raw_iv)
            continue
        instr_k = _kset(instr_words, k_for_exclusion)
        removes = []
        for i in range(span_len - k_for_exclusion + 1):
            kg = tuple(gold_words[s + i : s + i + k_for_exclusion])
            if kg in instr_k:
                removes.append((s + i, s + i + k_for_exclusion))
        removes = _merge_intervals(removes)
        for start, end in _subtract_from_interval(raw_iv, removes):
            if end - start >= min_length:
                all_trimmed.append((start, end))
    return _merge_intervals(all_trimmed)


# ---------------------------------------------------------------------------
# Book-level k-gram index
# ---------------------------------------------------------------------------

class BookIndex:
    """Inverted k-gram index over the concatenated word tokens of the full book."""

    def __init__(self, word_tokens: List[str]):
        self.words = word_tokens
        self._cache: Dict[int, Dict[tuple, List[int]]] = {}

    def get_kgram_index(self, k: int) -> Dict[tuple, List[int]]:
        if k not in self._cache:
            idx: Dict[tuple, List[int]] = defaultdict(list)
            w = self.words
            for i in range(len(w) - k + 1):
                idx[tuple(w[i : i + k])].append(i)
            self._cache[k] = idx
        return self._cache[k]


def _build_book_index(book_examples: list) -> Tuple[BookIndex, List[Tuple[int, int, str]]]:
    exs = sorted(book_examples, key=_pid_to_int)
    all_words: List[str] = []
    para_word_spans: List[Tuple[int, int, str]] = []  # (start, end, id)
    for ex in exs:
        words = _tok_words(_get_text(ex))
        start = len(all_words)
        all_words.extend(words)
        para_word_spans.append((start, len(all_words), _get_id(ex)))
    return BookIndex(all_words), para_word_spans


# ---------------------------------------------------------------------------
# Seed-and-extend matching
# ---------------------------------------------------------------------------

def _find_matches_against_book(
    gen_words: List[str], book_index: BookIndex, k: int
) -> List[Tuple[int, int]]:
    bw = book_index.words
    idx = book_index.get_kgram_index(k)
    visited = set()
    intervals: List[Tuple[int, int]] = []

    for j in range(len(gen_words) - k + 1):
        key = tuple(gen_words[j : j + k])
        starts = idx.get(key)
        if not starts:
            continue
        for i in starts:
            ii, jj = i, j
            while ii > 0 and jj > 0 and bw[ii - 1] == gen_words[jj - 1]:
                ii -= 1
                jj -= 1
            pair = (ii, jj)
            if pair in visited:
                continue
            visited.add(pair)
            p = 0
            while (ii + p) < len(bw) and (jj + p) < len(gen_words) and bw[ii + p] == gen_words[jj + p]:
                p += 1
            if p >= k:
                intervals.append((ii, ii + p))
    return intervals


def _find_raw_matches_per_excerpt(
    gen_words: List[str], para_words: List[str], min_length: int
) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    if not para_words or not gen_words:
        return matches
    for i in range(len(para_words)):
        if len(para_words) - i < min_length:
            break
        for j in range(len(gen_words)):
            if para_words[i] != gen_words[j]:
                continue
            L = 0
            while (
                i + L < len(para_words)
                and j + L < len(gen_words)
                and para_words[i + L] == gen_words[j + L]
            ):
                L += 1
            if L >= min_length:
                matches.append((i, i + L))
    return matches


# ---------------------------------------------------------------------------
# Span text extraction
# ---------------------------------------------------------------------------

def _extract_span_text_from_book(
    book_examples: list,
    para_word_spans: List[Tuple[int, int, str]],
    span_start: int,
    span_end: int,
) -> str:
    pid_to_text = {_get_id(ex): _get_text(ex) for ex in (book_examples or [])}

    pieces = []
    for para_start, para_end, pid in para_word_spans:
        if para_end <= span_start:
            continue
        if para_start >= span_end:
            break
        s = max(span_start, para_start)
        e = min(span_end, para_end)
        if s >= e:
            continue

        text = pid_to_text.get(pid, "")
        spans = _word_char_spans(text)

        local_s = s - para_start
        local_e = e - para_start
        if local_s < 0:
            local_s = 0
        if local_e > len(spans):
            local_e = len(spans)
        if local_s >= local_e:
            continue

        start_char = spans[local_s][0]
        end_char = spans[local_e - 1][1]
        pieces.append(text[start_char:end_char])

    return "\n".join(pieces).strip()


# ---------------------------------------------------------------------------
# Metric 1 & 2: bmc@k and Longest Contiguous Memorized Block
# ---------------------------------------------------------------------------

def compute_bmc_and_longest_block(
    book_index: BookIndex,
    examples: list,
    k: int = 5,
    trim_k: int = 5,
) -> Tuple[float, int, Tuple[int, int]]:
    """Compute bmc@k (Algorithm 1) and the longest contiguous memorized block."""
    n = len(book_index.words)
    if n == 0:
        return 0.0, 0, (0, 0)

    covered = [False] * n
    exs = sorted(examples, key=_pid_to_int)

    print(f"\nCalculating BMC@{k} and longest memorized block...")

    pbar = tqdm(exs, desc="  Processing", unit="para")
    for ex in pbar:
        instr_words = _tok_words(ex.get("instruction", ""))
        for gen in ex.get("generations", []) or []:
            gen_text = gen.get("generated_text", "")
            gen_words = _tok_words(gen_text)
            if len(gen_words) < k:
                continue

            raw_intervals = _find_matches_against_book(gen_words, book_index, k)
            if not raw_intervals:
                continue

            trimmed = _trim_instruction_kgrams(
                book_index.words, instr_words, raw_intervals,
                min_length=k, k_for_exclusion=trim_k,
            )

            for s, e in trimmed:
                for t in range(s, e):
                    covered[t] = True

        pbar.set_postfix({"coverage": f"{sum(covered) / n * 100:.1f}%"})

    bmc = sum(covered) / n

    # Longest run of True
    longest_block = 0
    current_run = 0
    block_end_pos = 0
    for i, c in enumerate(covered):
        if c:
            current_run += 1
            if current_run > longest_block:
                longest_block = current_run
                block_end_pos = i + 1
        else:
            current_run = 0

    block_start_pos = block_end_pos - longest_block
    return bmc, longest_block, (block_start_pos, block_end_pos)


# ---------------------------------------------------------------------------
# Metric 3: Longest Contiguous Regurgitated Span
# ---------------------------------------------------------------------------

def compute_longest_regurgitated_span(
    examples: list, k: int = 5,
) -> Tuple[int, Optional[str], Optional[str]]:
    longest = 0
    best_span: Optional[Tuple[int, int]] = None
    best_para_text: Optional[str] = None
    best_gen_text: Optional[str] = None

    exs = sorted(examples, key=_pid_to_int)

    print(f"\nComputing the longest contiguous regurgitated span...")
    pbar = tqdm(exs, desc="  Processing", unit="para")
    for ex in pbar:
        para_text = _get_text(ex)
        para_words = _tok_words(para_text)
        if not para_words:
            continue
        for gen in ex.get("generations", []) or []:
            gen_text = gen.get("generated_text", "")
            gen_words = _tok_words(gen_text)
            if not gen_words:
                continue
            matches = _find_raw_matches_per_excerpt(gen_words, para_words, min_length=k)
            for s, e in matches:
                span_len = e - s
                if span_len > longest:
                    longest = span_len
                    best_span = (s, e)
                    best_para_text = para_text
                    best_gen_text = gen_text

        pbar.set_postfix({"longest": f"{longest}w"})

    span_text = None
    if best_para_text and best_span:
        s, e = best_span
        spans = _word_char_spans(best_para_text)
        if s < len(spans) and e <= len(spans):
            start_char = spans[s][0]
            end_char = spans[e - 1][1]
            span_text = best_para_text[start_char:end_char]

    return longest, span_text, best_gen_text


# ---------------------------------------------------------------------------
# Metric 4: Number of Regurgitated Spans > T
# ---------------------------------------------------------------------------

def _interval_overlaps_any(
    sorted_intervals: List[Tuple[int, int]], s: int, e: int
) -> bool:
    starts = [iv[0] for iv in sorted_intervals]
    idx = bisect_left(starts, s)
    if idx > 0:
        a, b = sorted_intervals[idx - 1]
        if s < b and e > a:
            return True
    if idx < len(sorted_intervals):
        a, b = sorted_intervals[idx]
        if s < b and e > a:
            return True
    return False


def count_regurgitated_spans(
    examples: list, k: int = 5, span_threshold: int = 20,
) -> int:
    exs = sorted(examples, key=_pid_to_int)
    global_offset: Dict[str, int] = {}
    para_words_map: Dict[str, List[str]] = {}
    offset = 0
    for ex in exs:
        pid = _get_id(ex)
        words = _tok_words(_get_text(ex))
        para_words_map[pid] = words
        global_offset[pid] = offset
        offset += len(words)

    candidates: List[Tuple[int, int, int, int]] = []
    order = 0

    print(f"\nCounting regurgitated spans > {span_threshold} words...")
    pbar = tqdm(exs, desc="  Processing", unit="para")
    for ex in pbar:
        pid = _get_id(ex)
        para_words = para_words_map[pid]
        base = global_offset[pid]
        if not para_words:
            continue
        for gen in ex.get("generations", []) or []:
            gen_words = _tok_words(gen.get("generated_text", ""))
            if not gen_words:
                continue
            matches = _find_raw_matches_per_excerpt(gen_words, para_words, min_length=k)
            for s, e in matches:
                span_len = e - s
                if span_len <= span_threshold:
                    continue
                gs = base + s
                ge = base + e
                candidates.append((span_len, gs, ge, order))
                order += 1
        pbar.set_postfix({"candidates": len(candidates)})

    # Greedy non-overlapping, prefer longer
    candidates.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    selected: List[Tuple[int, int]] = []
    count = 0
    for _, gs, ge, _ in candidates:
        if _interval_overlaps_any(selected, gs, ge):
            continue
        idx = bisect_left([iv[0] for iv in selected], gs)
        selected.insert(idx, (gs, ge))
        count += 1
    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def evaluate(
    test_book_path: str,
    generation_file_path: str,
    k: int = 5,
    trim_k: int = 5,
    span_threshold: int = 20,
) -> Dict[str, Any]:
    with open(test_book_path, "r", encoding="utf-8") as f:
        book = json.load(f)
    with open(generation_file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    book_index, para_word_spans = _build_book_index(book)

    bmc_score, longest_block, (block_start, block_end) = compute_bmc_and_longest_block(
        book_index, examples, k=k, trim_k=trim_k,
    )
    block_text = None
    if longest_block > 0:
        block_text = _extract_span_text_from_book(book, para_word_spans, block_start, block_end)

    longest_regurg, regurg_span_text, regurg_gen_text = compute_longest_regurgitated_span(examples, k=k)
    num_spans = count_regurgitated_spans(examples, k=k, span_threshold=span_threshold)

    return {
        "bmc_score": bmc_score,
        "longest_memorized_block": longest_block,
        "longest_memorized_block_text": block_text,
        "longest_regurgitated_span": longest_regurg,
        "longest_regurgitated_span_text": regurg_span_text,
        "longest_regurgitated_generation_text": regurg_gen_text,
        "num_regurgitated_spans": num_spans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test_book", required=True, help="Path to test book JSON (list of paragraph dicts)")
    parser.add_argument("--generation_file", required=True, help="Path to generations JSON")
    parser.add_argument("--k", type=int, default=5, help="Minimum contiguous match length (default: 5)")
    parser.add_argument("--trim_k", type=int, default=5, help="Instruction m-gram trimming size (default: 5)")
    parser.add_argument("--span_threshold", type=int, default=20, help="Word threshold for metric 4 (default: 20)")
    parser.add_argument("--output_json", default=None, help="Optional path to write results JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("  Memorization Evaluation Metrics")
    print("=" * 70)

    results = evaluate(
        args.test_book,
        args.generation_file,
        k=args.k,
        trim_k=args.trim_k,
        span_threshold=args.span_threshold,
    )

    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)
    print(f"  BMC@{args.k}:                         {results['bmc_score'] * 100:.2f}%")
    print(f"  Longest Memorized Block:         {results['longest_memorized_block']} words")
    print(f"  Longest Regurgitated Span:       {results['longest_regurgitated_span']} words")
    print(f"  # Regurgitated Spans (>{args.span_threshold}w):  {results['num_regurgitated_spans']}")

    if results["longest_memorized_block_text"]:
        print(f"\n{'─' * 70}")
        print(f"  Longest Memorized Block (text):")
        print(f"{'─' * 70}")
        print(f"  {results['longest_memorized_block_text']}")

    if results["longest_regurgitated_span_text"]:
        print(f"\n{'─' * 70}")
        print(f"  Longest Regurgitated Span (text):")
        print(f"{'─' * 70}")
        print(f"  {results['longest_regurgitated_span_text']}")

    if args.output_json:
        from pathlib import Path
        Path(args.output_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
