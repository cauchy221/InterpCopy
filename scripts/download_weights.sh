#!/usr/bin/env bash
# Download Llama-3.1 Instruct weights (8B, 70B, 405B) to Lustre via tune download.
#
# Usage:
#   bash scripts/download_weights.sh           # all three
#   bash scripts/download_weights.sh 8b        # just one
#   bash scripts/download_weights.sh 8b 70b    # a subset

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN not set — source .env first}"
: "${HF_HOME:?HF_HOME not set — source .env first}"

MODELS_DIR="$HF_HOME/models"
mkdir -p "$MODELS_DIR"

declare -A REPOS=(
  [8b]="meta-llama/Llama-3.1-8B-Instruct"
  [70b]="meta-llama/Llama-3.1-70B-Instruct"
  [405b]="meta-llama/Llama-3.1-405B-Instruct"
)

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(8b 70b 405b)
fi

for t in "${targets[@]}"; do
  repo="${REPOS[$t]:-}"
  if [[ -z "$repo" ]]; then
    echo "unknown target: $t (pick from: ${!REPOS[*]})" >&2
    exit 1
  fi
  out="$MODELS_DIR/$(basename "$repo")"
  echo ">>> downloading $repo -> $out"
  tune download "$repo" \
    --output-dir "$out" \
    --hf-token "$HF_TOKEN" \
    --ignore-patterns "original/consolidated.*"  # skip raw-format shards; we only need HF format
done

echo "done"
