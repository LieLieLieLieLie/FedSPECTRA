"""Compatibility bootstrap for the pinned Windows Python 3.9 environment.

The host's legacy CryptoAPI and Winsock providers currently fail during the
initialization paths used by CPython 3.9.  PYTHONHASHSEED is set by the launch
scripts.  This module supplies deterministic non-security randomness for
library initialization and a minimal asyncio shim before importing PyTorch.
It does not affect experiment RNGs, which are seeded explicitly.
"""

from __future__ import annotations


def bootstrap() -> None:
    import os
    import random
    import sys
    import types

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def _compat_urandom(n: int) -> bytes:
        return bytes(((73 * i + 41) % 256) for i in range(n))

    os.urandom = _compat_urandom  # type: ignore[assignment]
    random._urandom = _compat_urandom  # type: ignore[attr-defined]
    if "asyncio" not in sys.modules:
        shim = types.ModuleType("asyncio")
        shim.iscoroutinefunction = lambda _: False
        shim.coroutines = types.SimpleNamespace(_is_coroutine=object())
        sys.modules["asyncio"] = shim
