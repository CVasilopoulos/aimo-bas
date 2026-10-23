#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${HF_CACHE:-$HERE/hf-cache}/hub"
for pair in "Qwen/Qwen3.5-0.8B 2fc06364715b967f1860aea9cf38778875588b17" "Qwen/Qwen3-0.6B c1899de289a04d12100db370d81485cdf75e47ca"; do
  read -r model revision <<< "$pair"
  uv run --quiet --no-project --with huggingface_hub hf download "$model" --revision "$revision" --cache-dir "$CACHE" > /dev/null
  dir="$CACHE/models--${model//\//--}"
  mkdir -p "$dir/refs"
  printf %s "$revision" > "$dir/refs/main"
  echo "ok $model $revision"
done
