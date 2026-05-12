#!/usr/bin/env bash
# Idempotently raise vLLM 0.8.5's hardcoded execute_model RPC timeout.
#
# vLLM 0.8.5 sets EXECUTE_MODEL_TIMEOUT_S = 40 in
# vllm/v1/executor/multiproc_executor.py and exposes no env-var override.
# The first execute_model RPC after engine init has to (a) lazy-load the
# LoRA adapter on each TP worker and (b) run the first prefill; at
# 405B/TP=8 with rank-32 LoRA that exceeds 40s and the engine dies with
# "RPC call to execute_model timed out". Two of our jobs (30017, 30046)
# died at exactly 41s and 40s respectively. An in-process monkey-patch
# does not work because vllm.utils._maybe_force_spawn() forces the
# `spawn` start method whenever CUDA is initialized; the EngineCore
# subprocess then re-imports modules from disk and parent state is lost.
# So we patch the constant on disk.
#
# Upstream fix is vllm-project/vllm#19544 (lands in 0.9.x as
# VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS). Remove this script after upgrading.
#
# Idempotent: safe to re-run; only writes if current value < target.

set -euo pipefail

TARGET_VALUE="${VLLM_EXECUTE_MODEL_TIMEOUT_S:-1800}"

# Find the installed file via the active python env.
PYBIN="${PYTHON:-python}"
# vLLM logs INFO/WARNING lines to stdout during import on some platforms;
# the actual __file__ is the last line.
FILE="$("$PYBIN" -c "import vllm.v1.executor.multiproc_executor as m; print(m.__file__)" 2>/dev/null | tail -n1 || true)"
if [[ -z "$FILE" || ! -f "$FILE" ]]; then
  echo "vllm patch: could not locate multiproc_executor.py via $PYBIN — skipping" >&2
  exit 0
fi

CUR="$(grep -E '^EXECUTE_MODEL_TIMEOUT_S = [0-9]+$' "$FILE" | head -1 | awk '{print $3}')"
if [[ -z "${CUR:-}" ]]; then
  echo "vllm patch: could not parse EXECUTE_MODEL_TIMEOUT_S in $FILE — bailing without changes" >&2
  exit 1
fi

if (( CUR >= TARGET_VALUE )); then
  echo "vllm patch: EXECUTE_MODEL_TIMEOUT_S already $CUR (>= $TARGET_VALUE) — no change"
  exit 0
fi

sed -i.bak "s/^EXECUTE_MODEL_TIMEOUT_S = ${CUR}\$/EXECUTE_MODEL_TIMEOUT_S = ${TARGET_VALUE}/" "$FILE"
NEW="$(grep -E '^EXECUTE_MODEL_TIMEOUT_S = [0-9]+$' "$FILE" | head -1 | awk '{print $3}')"
if [[ "$NEW" != "$TARGET_VALUE" ]]; then
  echo "vllm patch: write verification failed (got $NEW, wanted $TARGET_VALUE) — aborting" >&2
  exit 1
fi
echo "vllm patch: EXECUTE_MODEL_TIMEOUT_S $CUR -> $NEW in $FILE"
