"""Honest same-process benchmarks against qrcode 8.2."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import qrcode  # noqa: E402
import qrcode.base as upstream_base  # noqa: E402
import qrcode.util as upstream_util  # noqa: E402

import mojo_qrcode  # noqa: E402
from mojo_qrcode import util  # noqa: E402


def timeit(function, number: int, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            function()
        best = min(best, (time.perf_counter() - start) / number)
    return best


def upstream_rs(data: bytes, nsym: int) -> bytes:
    generator = upstream_base.Polynomial([1], 0)
    for i in range(nsym):
        generator *= upstream_base.Polynomial([1, upstream_base.gexp(i)], 0)
    remainder = upstream_base.Polynomial(list(data), len(generator) - 1) % generator
    offset = len(remainder) - nsym
    return bytes(
        remainder[i + offset] if i + offset >= 0 else 0 for i in range(nsym)
    )


def encode_ours(payload: bytes, version: int, mask):
    qr = mojo_qrcode.QRCode(
        version=version,
        error_correction=mojo_qrcode.ERROR_CORRECT_M,
        border=0,
        mask_pattern=mask,
    )
    qr.add_data(payload, optimize=0)
    qr.make(fit=False)
    return qr.modules


def encode_upstream(payload: bytes, version: int, mask):
    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.ERROR_CORRECT_M,
        border=0,
        mask_pattern=mask,
    )
    qr.add_data(payload, optimize=0)
    qr.make(fit=False)
    return qr.modules


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    rs_data = bytes((i * 149 + 37) % 256 for i in range(121))
    large = bytes((i * 73 + 19) % 256 for i in range(2000))
    medium = bytes((i * 43 + 11) % 256 for i in range(600))
    small = b"hello"

    ours_data = util.QRData(large, mode=util.MODE_8BIT_BYTE)
    upstream_data = upstream_util.QRData(
        large, mode=upstream_util.MODE_8BIT_BYTE
    )

    cases = [
        (
            "RS parity, 121 data + 30 ECC",
            lambda: util.rs_encode(rs_data, 30),
            lambda: upstream_rs(rs_data, 30),
            300,
        ),
        (
            "Codewords, version 40-M, 2000 B",
            lambda: util.create_data(
                40, mojo_qrcode.ERROR_CORRECT_M, [ours_data]
            ),
            lambda: upstream_util.create_data(
                40, qrcode.ERROR_CORRECT_M, [upstream_data]
            ),
            80,
        ),
        (
            "QR matrix, version 20-M, mask 3",
            lambda: encode_ours(medium, 20, 3),
            lambda: encode_upstream(medium, 20, 3),
            30,
        ),
        (
            "QR matrix, version 20-M, best mask",
            lambda: encode_ours(medium, 20, None),
            lambda: encode_upstream(medium, 20, None),
            10,
        ),
        (
            "QR matrix, version 1-M, best mask",
            lambda: encode_ours(small, 1, None),
            lambda: encode_upstream(small, 1, None),
            100,
        ),
    ]

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| Case | mojo-qrcode | qrcode 8.2 | Upstream / Mojo | Result |")
    print("|---|---:|---:|---:|---|")
    for name, ours, upstream, number in cases:
        ours_value = ours()
        upstream_value = upstream()
        assert ours_value == upstream_value
        ours_time = timeit(ours, number)
        upstream_time = timeit(upstream, number)
        ratio = upstream_time / ours_time
        result = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {ours_time * 1e3:.3f} ms | "
            f"{upstream_time * 1e3:.3f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
