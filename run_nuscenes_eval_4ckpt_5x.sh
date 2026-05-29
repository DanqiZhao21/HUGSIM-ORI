#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if [ "$#" -eq 0 ]; then
  exec pixi run python "$SCRIPT_DIR/utils/nuscenes_eval_runner.py" --slots 0:0 1:1 2:2 3:3
fi

exec pixi run python "$SCRIPT_DIR/utils/nuscenes_eval_runner.py" "$@"
