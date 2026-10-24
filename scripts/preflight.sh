#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1
.venv/bin/python -m cmbranch.preflight --config "${1:-configs/branch_p025.json}" --report validation/preflight.json
