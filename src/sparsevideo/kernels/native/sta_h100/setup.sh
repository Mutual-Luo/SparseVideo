#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
FASTVIDEO_KERNEL_BUILD_TK="${FASTVIDEO_KERNEL_BUILD_TK:-ON}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90a}"
if [[ "${FASTVIDEO_KERNEL_BUILD_TK}" == "ON" && -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST="9.0a"
fi
CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$("${PYTHON}" - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DPython_EXECUTABLE="${PYTHON}" \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
  -DFASTVIDEO_KERNEL_BUILD_TK="${FASTVIDEO_KERNEL_BUILD_TK}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${SCRIPT_DIR}"
cmake --build "${BUILD_DIR}" --target fastvideo_kernel_ops --parallel "${MAX_JOBS:-8}"
