#!/usr/bin/env bash
# One-command env bootstrap. Idempotent, safe to re-source.
#
# Creates ~/envs/tt (uv venv), installs this project's deps, and sources .env
# so HF_TOKEN / HF_HOME are live in the current shell.
#
# Usage (must be sourced, not executed, so `activate` sticks in your shell):
#   source scripts/setup_env.sh

# NOTE: no `set -e` — this script is sourced, so a failing command must not
# kill the user's login shell. We return on error instead.

_repo="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_env_dir="${INTERPCOPY_ENV_DIR:-$HOME/envs/tt}"

# --- modules ---
if command -v module >/dev/null 2>&1; then
  module load Python/3.10.15 CUDA/13.1 || { echo "module load failed" >&2; return 1 2>/dev/null || exit 1; }
fi

# --- install uv if missing (user-local, no sudo) ---
if ! command -v uv >/dev/null 2>&1; then
  echo ">>> uv not found — installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh || {
    echo "uv install failed" >&2
    return 1 2>/dev/null || exit 1
  }
  # installer puts uv in ~/.local/bin; make sure it's on PATH for this shell
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- venv ---
if [[ ! -d "$_env_dir" ]]; then
  uv venv "$_env_dir" --python 3.10 || { echo "uv venv failed" >&2; return 1 2>/dev/null || exit 1; }
fi

# shellcheck disable=SC1091
source "$_env_dir/bin/activate" || { echo "activate failed" >&2; return 1 2>/dev/null || exit 1; }

# --- install project deps ---
uv pip install -e "$_repo" || { echo "uv pip install failed" >&2; return 1 2>/dev/null || exit 1; }

# --- load .env ---
if [[ -f "$_repo/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$_repo/.env"
  set +a
else
  echo "warning: $_repo/.env not found — HF_TOKEN will be missing" >&2
fi

echo "env ready: $_env_dir"
echo "HF_HOME=${HF_HOME:-unset}"
