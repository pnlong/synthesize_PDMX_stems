#!/usr/bin/env bash
# Mix all eight ablation conditions into sibling *_summable trees (no overwrite of raw).
# Does not write mixture.* files (listening uses stems only).
#
# Usage:
#   bash synthesis/mix_ablations.sh
#   JOBS=20 bash synthesis/mix_ablations.sh
set -euo pipefail

JOBS="${JOBS:-20}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODES=(basic slakh ddsp_basic ddsp_slakh)

for mode in "${MODES[@]}"; do
  echo "=== mix ${mode} (raw → ${mode}_summable) ==="
  uv run python -m synthesis.mix --render-mode "$mode" --no-overwrite -j "$JOBS"
  echo "=== mix ${mode} --realify → ${mode}_realify_summable ==="
  uv run python -m synthesis.mix --render-mode "$mode" --realify --no-overwrite -j "$JOBS"
done

echo "Done. Listening prefers *_summable when LISTENING_PREFER_SUMMABLE=True (shared/config.py)."
