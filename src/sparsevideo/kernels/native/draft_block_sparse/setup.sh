#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

export BLOCK_SPARSE_ATTN_CUDA_ARCHS="${BLOCK_SPARSE_ATTN_CUDA_ARCHS:-80}"
export BLOCK_SPARSE_ATTN_BUILD_MODE="${BLOCK_SPARSE_ATTN_BUILD_MODE:-full}"

cd "${SCRIPT_DIR}"
"${PYTHON}" setup.py build_ext --inplace
