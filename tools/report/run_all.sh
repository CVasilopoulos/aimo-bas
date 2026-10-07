#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for script in runtime stats public_data perturbation_examples config_tables leaderboard; do
  uv run --quiet "$HERE/$script.py"
done
