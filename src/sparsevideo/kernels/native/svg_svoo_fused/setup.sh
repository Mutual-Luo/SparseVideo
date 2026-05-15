#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/../build}"

export BUILD_DIR
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PATH="$(dirname "${PYTHON}"):${PATH}"

"${PYTHON}" "${ROOT_DIR}/build.py"
