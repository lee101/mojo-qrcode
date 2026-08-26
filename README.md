# mojo-qrcode

`mojo-qrcode` is a standalone Mojo-accelerated port of the compute-heavy QR
encoding path in Python's [`qrcode`](https://github.com/lincolnloop/python-qrcode)
package. It generates standards-compatible QR matrices with Reed-Solomon error
correction and offers a Python API matching the covered `qrcode` subset.

The implementation is tested against `qrcode` 8.2 across every QR version and
error-correction level. It does not call upstream at runtime.

## Coverage

Covered:

- QR versions 1 through 40
- L, M, Q, and H error-correction levels
- numeric, alphanumeric, and UTF-8 byte modes
- upstream-compatible data chunk optimization and automatic version fitting
- all eight mask patterns and automatic lowest-penalty mask selection
- QR GF(256) Reed-Solomon generation and multi-block interleaving
- `QRCode`, `add_data`, `make`, `get_matrix`, `make_image`, `print_ascii`,
  `print_tty`, `clear`, and the top-level `make` shortcut
- basic Pillow PNG output with `fill_color` and `back_color`

Not covered:

- decoding or Reed-Solomon error correction of damaged input
- Kanji-mode input encoding (upstream `qrcode` does not implement it either)
- the full public surface of `qrcode.util`, `qrcode.base`, and image base classes
- SVG, PyPNG, styled module drawers, color masks, embedded images, and custom
  context-aware image factories
- Pillow output options beyond the basic square-module image described above
- the `qr` command-line application

## Install

Clone the repository, then install the pinned environment and build the shared
library:

```bash
pixi install
pixi run build
```

The Pixi environment sets `PYTHONPATH=python`. `qrcode` 8.2 is installed only
as a parity-test and benchmark reference.

## Usage

The API is intentionally shaped so existing covered code only needs an import
alias:

```python
import mojo_qrcode as qrcode

qr = qrcode.QRCode(
    version=None,
    error_correction=qrcode.ERROR_CORRECT_Q,
    box_size=8,
    border=4,
)
qr.add_data("https://example.com/mojo")
qr.make(fit=True)

print(qr.version)
matrix = qr.get_matrix()
qr.make_image(fill_color="black", back_color="white").save("qr.png")
```

Run it inside the environment with `pixi run python your_script.py`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz
(Linux x86-64). Each row uses identical input and asserts identical output
before timing. The ratio is upstream time divided by Mojo time, so values
above 1 mean `mojo-qrcode` is faster.

| Case | mojo-qrcode | qrcode 8.2 | Upstream / Mojo | Result |
|---|---:|---:|---:|---|
| RS parity, 121 data + 30 ECC | 0.019 ms | 1.405 ms | 72.28x | faster |
| Codewords, version 40-M, 2000 B | 0.752 ms | 24.391 ms | 32.42x | faster |
| QR matrix, version 20-M, mask 3 | 0.525 ms | 9.327 ms | 17.77x | faster |
| QR matrix, version 20-M, best mask | 2.991 ms | 59.594 ms | 19.93x | faster |
| QR matrix, version 1-M, best mask | 0.236 ms | 2.219 ms | 9.39x | faster |

These are local measurements, not portable performance guarantees. The small
version-1 case includes Python object construction, NumPy staging, and FFI
overhead; the larger cases better expose the accelerated kernels.

The CPU path packs byte payloads a byte at a time, exposes the backing
`bytearray` to NumPy without a copy, reuses contiguous matrix templates during
mask selection, and runs all eight mask trials behind one FFI call. Template
copies and dark-module scoring use SIMD plus scalar remainder loops.
Reed-Solomon generators shared by a block layout are built once.

No GPU or multithreaded path is included. QR placement and scoring are
branch-heavy, low-arithmetic-intensity kernels (well below roughly two FLOPs
per byte moved), while even the largest matrix is only 177 by 177; transfer and
thread-launch overhead would dominate this workload.

## How it works

Python handles the public objects, mode selection, bit packing, QR capacity
tables, fixed patterns, and Pillow integration. One Mojo compilation unit
exports four C-ABI functions covering three hot-path areas:

- GF(256) generator construction, systematic Reed-Solomon parity, and QR block
  interleaving
- zig-zag data placement with the selected QR mask
- ISO mask scoring: runs, 2x2 blocks, finder-like patterns, and dark-module
  balance

Buffers cross `ctypes` as integer addresses because exported Mojo functions
cannot carry inferred pointer origins. The Mojo side reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values. Codewords and matrices are
contiguous `uint8` arrays; Reed-Solomon block sizes are contiguous `int64`
arrays. Python validates shapes, ranges, contiguity, and non-null addresses;
the exported functions also receive buffer lengths and reject inconsistent
arguments before dereferencing. Python owns every allocation and keeps it
alive for the complete call, so the shared library does not transfer memory
ownership or retain pointers.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite contains 296 tests, including exhaustive codeword parity for
all 160 version/error-correction combinations, all eight masks, independent
Reed-Solomon comparisons, full-matrix parity through version 40, image parity,
SIMD and bit-packing tail cases, and API behavior.
