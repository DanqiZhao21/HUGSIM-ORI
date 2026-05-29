#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
UNIAD_REPO="${UNIAD_REPO:-/root/clone/UniAD_SIM}"
UNIAD_CONDA_ENV="${UNIAD_CONDA_ENV:-uniad2.0}"
UNIAD_PYTHON_BIN="${UNIAD_PYTHON_BIN:-/root/miniconda3/envs/${UNIAD_CONDA_ENV}/bin/python}"
UNIAD_CONFIG="${UNIAD_CONFIG:-$UNIAD_REPO/projects/configs/stage2_e2e/base_e2e.py}"
UNIAD_CHECKPOINT="${UNIAD_CHECKPOINT:-$UNIAD_REPO/ckpts/uniad_base_e2e.pth}"

cd "$UNIAD_REPO"
echo "$PWD"

UNIAD_TMPDIR="${2}/uniad_tmp"
mkdir -p "$UNIAD_TMPDIR"

CUDA_VISIBLE_DEVICES="${1}" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 "$UNIAD_PYTHON_BIN" -u -X faulthandler tools/closeloop/e2e.py \
  "$UNIAD_CONFIG" \
  "$UNIAD_CHECKPOINT" \
  --launcher none \
  --eval bbox \
  --tmpdir "$UNIAD_TMPDIR" \
  --output "$2"
