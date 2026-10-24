#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1
check_python="${COVERMEM_TEST_PYTHON:-.venv/bin/python}"
"$check_python" -m compileall -q cmbranch tests supervision
"$check_python" -m unittest discover -s tests -v
"$check_python" -m unittest discover -s supervision/tests -v
"$check_python" -m cmbranch.preflight --config "${1:-configs/branch_p025.json}" --assets-only
