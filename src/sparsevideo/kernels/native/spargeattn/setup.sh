#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

cd "${SCRIPT_DIR}"
"${PYTHON}" setup.py build_ext --inplace
