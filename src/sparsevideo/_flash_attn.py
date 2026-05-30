"""Helpers for requiring flash-attention at method call time.

flash-attention is NOT a hard install-time dependency of sparsevideo because it
requires a matching CUDA toolkit and must be compiled from source.  Instead,
methods that need it call one of the functions below, which give the user a clear
error with install instructions if flash-attention is missing.

Methods that use flash-attention:
  svg1   — flash_attn_varlen_func  (Wan/HunyuanVideo sparse path)
  draft  — flash_attn_varlen_func  (HunyuanVideo dense-attention path)
  adacluster — flash_attn_func     (cluster sparse attention)
"""
from __future__ import annotations

_INSTALL_MSG = "flash-attention is required by sparsevideo. Please install flash-attention first."


def require_flash_attn_varlen_func():
    """Return flash_attn_varlen_func or raise ImportError with install guidance."""
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        return flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def require_flash_attn_func():
    """Return flash_attn_func or raise ImportError with install guidance."""
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc
