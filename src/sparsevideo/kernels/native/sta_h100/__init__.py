from __future__ import annotations

import importlib.util
from pathlib import Path

sta_fwd = None

_ROOT = Path(__file__).resolve().parent
_CANDIDATES = [
    *_ROOT.glob("fastvideo_kernel_ops*.so"),
    *_ROOT.glob("_C/fastvideo_kernel_ops*.so"),
    *_ROOT.glob("build/**/fastvideo_kernel_ops*.so"),
]

for _candidate in _CANDIDATES:
    _spec = importlib.util.spec_from_file_location("fastvideo_kernel_ops", _candidate)
    if _spec is None or _spec.loader is None:
        continue
    try:
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        sta_fwd = getattr(_module, "sta_fwd", None)
    except Exception:
        sta_fwd = None
    if sta_fwd is not None:
        break

__all__ = ["sta_fwd"]
