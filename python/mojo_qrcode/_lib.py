"""ctypes bridge to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "qrcode.mojo")
LIB = os.environ.get("MOJO_QRCODE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-qrcode.so"
)

I = ctypes.c_int64
_SIGNATURES = {
    "mqr_rs_encode": ([I] * 11, I),
    "mqr_interleave_rs": ([I] * 15, I),
    "mqr_map_data": ([I] * 6, I),
    "mqr_lost_point": ([I] * 3, I),
    "mqr_best_mask": ([I] * 7, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and (not os.path.exists(SRC) or os.path.getmtime(LIB) >= os.path.getmtime(SRC))
    ):
        return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run `pixi run build` before importing")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return address
