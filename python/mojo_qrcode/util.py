"""QR bit packing, mode selection, and Mojo-backed encoding kernels."""

from __future__ import annotations

import math
import re

import numpy as np

from . import base, exceptions
from ._lib import addr, lib

MODE_NUMBER = 1
MODE_ALPHA_NUM = 2
MODE_8BIT_BYTE = 4
MODE_KANJI = 8

MODE_SIZE_SMALL = {
    MODE_NUMBER: 10,
    MODE_ALPHA_NUM: 9,
    MODE_8BIT_BYTE: 8,
    MODE_KANJI: 8,
}
MODE_SIZE_MEDIUM = {
    MODE_NUMBER: 12,
    MODE_ALPHA_NUM: 11,
    MODE_8BIT_BYTE: 16,
    MODE_KANJI: 10,
}
MODE_SIZE_LARGE = {
    MODE_NUMBER: 14,
    MODE_ALPHA_NUM: 13,
    MODE_8BIT_BYTE: 16,
    MODE_KANJI: 12,
}

ALPHA_NUM = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
RE_ALPHA_NUM = re.compile(b"^[" + re.escape(ALPHA_NUM) + rb"]*\Z")
NUMBER_LENGTH = {3: 10, 2: 7, 1: 4}

PATTERN_POSITION_TABLE = [
    [], [6,18], [6,22], [6,26], [6,30], [6,34], [6,22,38], [6,24,42],
    [6,26,46], [6,28,50], [6,30,54], [6,32,58], [6,34,62], [6,26,46,66],
    [6,26,48,70], [6,26,50,74], [6,30,54,78], [6,30,56,82], [6,30,58,86],
    [6,34,62,90], [6,28,50,72,94], [6,26,50,74,98], [6,30,54,78,102],
    [6,28,54,80,106], [6,32,58,84,110], [6,30,58,86,114],
    [6,34,62,90,118], [6,26,50,74,98,122], [6,30,54,78,102,126],
    [6,26,52,78,104,130], [6,30,56,82,108,134], [6,34,60,86,112,138],
    [6,30,58,86,114,142], [6,34,62,90,118,146],
    [6,30,54,78,102,126,150], [6,24,50,76,102,128,154],
    [6,28,54,80,106,132,158], [6,32,58,84,110,136,162],
    [6,26,54,82,110,138,166], [6,30,58,86,114,142,170],
]

G15 = 0b10100110111
G18 = 0b1111100100101
G15_MASK = 0b101010000010010
PAD0 = 0xEC
PAD1 = 0x11

BIT_LIMIT_TABLE = [
    [0]
    + [
        8 * sum(block.data_count for block in base.rs_blocks(version, level))
        for version in range(1, 41)
    ]
    for level in range(4)
]

_EXP = np.asarray([base.EXP_TABLE[i % 255] for i in range(512)], dtype=np.uint8)
_LOG = np.asarray(base.LOG_TABLE, dtype=np.uint8)


def BCH_digit(data: int) -> int:
    return data.bit_length()


def BCH_type_info(data: int) -> int:
    value = data << 10
    while BCH_digit(value) >= BCH_digit(G15):
        value ^= G15 << (BCH_digit(value) - BCH_digit(G15))
    return ((data << 10) | value) ^ G15_MASK


def BCH_type_number(data: int) -> int:
    value = data << 12
    while BCH_digit(value) >= BCH_digit(G18):
        value ^= G18 << (BCH_digit(value) - BCH_digit(G18))
    return (data << 12) | value


def pattern_position(version: int):
    return PATTERN_POSITION_TABLE[version - 1]


def mask_func(pattern: int):
    if pattern == 0:
        return lambda i, j: (i + j) % 2 == 0
    if pattern == 1:
        return lambda i, j: i % 2 == 0
    if pattern == 2:
        return lambda i, j: j % 3 == 0
    if pattern == 3:
        return lambda i, j: (i + j) % 3 == 0
    if pattern == 4:
        return lambda i, j: (i // 2 + j // 3) % 2 == 0
    if pattern == 5:
        return lambda i, j: (i * j) % 2 + (i * j) % 3 == 0
    if pattern == 6:
        return lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0
    if pattern == 7:
        return lambda i, j: ((i * j) % 3 + (i + j) % 2) % 2 == 0
    raise TypeError(f"Bad mask pattern: {pattern}")


def mode_sizes_for_version(version: int):
    if version < 10:
        return MODE_SIZE_SMALL
    if version < 27:
        return MODE_SIZE_MEDIUM
    return MODE_SIZE_LARGE


def length_in_bits(mode: int, version: int) -> int:
    if mode not in (MODE_NUMBER, MODE_ALPHA_NUM, MODE_8BIT_BYTE, MODE_KANJI):
        raise TypeError(f"Invalid mode ({mode})")
    check_version(version)
    return mode_sizes_for_version(version)[mode]


def check_version(version: int):
    if version < 1 or version > 40:
        raise ValueError(f"Invalid version (was {version}, expected 1 to 40)")


def lost_point(modules) -> int:
    raw = np.asarray(modules)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] == 0:
        raise ValueError("modules must be a non-empty square matrix")
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError("modules must contain only boolean, 0, or 1 values")
    matrix = np.ascontiguousarray(raw, dtype=np.uint8)
    result = int(lib().mqr_lost_point(addr(matrix), matrix.size, matrix.shape[0]))
    if result < 0:
        raise RuntimeError("Mojo mask scoring rejected its validated buffers")
    return result


def matrix_array(modules):
    if isinstance(modules, np.ndarray):
        raw = modules
    else:
        raw = np.asarray(
            [[2 if value is None else value for value in row] for row in modules],
        )
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] == 0:
        raise ValueError("modules must be a non-empty square matrix")
    try:
        valid = np.all((raw == 0) | (raw == 1) | (raw == 2))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("modules must contain only boolean, None, 0, 1, or 2 values")
    return np.array(raw, dtype=np.uint8, order="C", copy=True)


def optimal_data_chunks(data, minimum=4):
    data = to_bytestring(data)
    num_pattern = rb"\d"
    alpha_pattern = b"[" + re.escape(ALPHA_NUM) + b"]"
    if len(data) <= minimum:
        num_pattern = re.compile(b"^" + num_pattern + b"+$")
        alpha_pattern = re.compile(b"^" + alpha_pattern + b"+$")
    else:
        repeat = b"{" + str(minimum).encode("ascii") + b",}"
        num_pattern = re.compile(num_pattern + repeat)
        alpha_pattern = re.compile(alpha_pattern + repeat)
    for is_num, chunk in _optimal_split(data, num_pattern):
        if is_num:
            yield QRData(chunk, mode=MODE_NUMBER, check_data=False)
        else:
            for is_alpha, sub_chunk in _optimal_split(chunk, alpha_pattern):
                mode = MODE_ALPHA_NUM if is_alpha else MODE_8BIT_BYTE
                yield QRData(sub_chunk, mode=mode, check_data=False)


def _optimal_split(data: bytes, pattern):
    while data:
        match = re.search(pattern, data)
        if not match:
            break
        start, end = match.start(), match.end()
        if start:
            yield False, data[:start]
        yield True, data[start:end]
        data = data[end:]
    if data:
        yield False, data


def to_bytestring(data) -> bytes:
    if not isinstance(data, bytes):
        data = str(data).encode("utf-8")
    return data


def optimal_mode(data: bytes) -> int:
    if data.isdigit():
        return MODE_NUMBER
    if RE_ALPHA_NUM.match(data):
        return MODE_ALPHA_NUM
    return MODE_8BIT_BYTE


class QRData:
    def __init__(self, data, mode=None, check_data=True):
        if check_data:
            data = to_bytestring(data)
        if mode is None:
            self.mode = optimal_mode(data)
        else:
            self.mode = mode
            if mode not in (MODE_NUMBER, MODE_ALPHA_NUM, MODE_8BIT_BYTE):
                raise TypeError(f"Invalid mode ({mode})")
            if check_data and mode < optimal_mode(data):
                raise ValueError(f"Provided data can not be represented in mode {mode}")
        self.data = data

    def __len__(self):
        return len(self.data)

    def write(self, buffer):
        if self.mode == MODE_NUMBER:
            for i in range(0, len(self.data), 3):
                chars = self.data[i : i + 3]
                buffer.put(int(chars), NUMBER_LENGTH[len(chars)])
        elif self.mode == MODE_ALPHA_NUM:
            for i in range(0, len(self.data), 2):
                chars = self.data[i : i + 2]
                if len(chars) == 2:
                    buffer.put(
                        ALPHA_NUM.find(chars[0:1]) * 45
                        + ALPHA_NUM.find(chars[1:2]),
                        11,
                    )
                else:
                    buffer.put(ALPHA_NUM.find(chars), 6)
        else:
            buffer.put_bytes(self.data)

    def __repr__(self):
        return repr(self.data)


class BitBuffer:
    def __init__(self):
        self.buffer = bytearray()
        self.length = 0

    def __repr__(self):
        return ".".join(str(value) for value in self.buffer)

    def get(self, index):
        return ((self.buffer[index // 8] >> (7 - index % 8)) & 1) == 1

    def put(self, num, length):
        while length:
            offset = self.length & 7
            if offset == 0:
                self.buffer.append(0)
            available = 8 - offset
            width = min(length, available)
            shift = length - width
            self.buffer[-1] |= (
                (num >> shift) & ((1 << width) - 1)
            ) << (available - width)
            self.length += width
            length -= width

    def put_bytes(self, values):
        if not values:
            return
        offset = self.length & 7
        if offset == 0:
            self.buffer.extend(values)
        else:
            left_shift = 8 - offset
            for value in values:
                self.buffer[-1] |= value >> offset
                self.buffer.append((value << left_shift) & 0xFF)
        self.length += len(values) * 8

    def __len__(self):
        return self.length

    def put_bit(self, bit):
        byte = self.length // 8
        if len(self.buffer) <= byte:
            self.buffer.append(0)
        if bit:
            self.buffer[byte] |= 0x80 >> (self.length % 8)
        self.length += 1


def rs_encode(data, nsym: int) -> bytes:
    """Return `nsym` QR-field Reed-Solomon parity bytes for `data`."""
    if not 1 <= nsym <= 255:
        raise ValueError("nsym must be in 1..255")
    source = np.frombuffer(bytes(data), dtype=np.uint8)
    if source.size == 0:
        source = np.zeros(1, dtype=np.uint8)
        source_len = 0
    else:
        source_len = source.size
    parity = np.empty(nsym, dtype=np.uint8)
    generator = np.empty(nsym + 1, dtype=np.uint8)
    status = lib().mqr_rs_encode(
        addr(source),
        source_len,
        nsym,
        addr(parity),
        parity.size,
        addr(generator),
        generator.size,
        addr(_EXP),
        _EXP.size,
        addr(_LOG),
        _LOG.size,
    )
    if status != 0:
        raise RuntimeError(f"Mojo Reed-Solomon kernel failed with status {status}")
    return parity.tobytes()


def create_bytes(buffer: BitBuffer, rs_blocks: list[base.RSBlock]):
    data = np.frombuffer(buffer.buffer, dtype=np.uint8)
    dc_counts = np.asarray([block.data_count for block in rs_blocks], dtype=np.int64)
    ec_counts = np.asarray(
        [block.total_count - block.data_count for block in rs_blocks], dtype=np.int64
    )
    total = int(sum(block.total_count for block in rs_blocks))
    result = np.empty(total, dtype=np.uint8)
    max_ec = int(ec_counts.max())
    ec_work = np.empty(len(rs_blocks) * max_ec, dtype=np.uint8)
    generator = np.empty(max_ec + 1, dtype=np.uint8)
    written = lib().mqr_interleave_rs(
        addr(data),
        data.size,
        addr(dc_counts),
        addr(ec_counts),
        len(rs_blocks),
        addr(result),
        result.size,
        addr(ec_work),
        ec_work.size,
        addr(generator),
        generator.size,
        addr(_EXP),
        _EXP.size,
        addr(_LOG),
        _LOG.size,
    )
    if written != total:
        raise RuntimeError(f"Mojo wrote {written} codewords, expected {total}")
    return result.tolist()


def create_data(version, error_correction, data_list):
    buffer = BitBuffer()
    for data in data_list:
        buffer.put(data.mode, 4)
        buffer.put(len(data), length_in_bits(data.mode, version))
        data.write(buffer)

    blocks = base.rs_blocks(version, error_correction)
    bit_limit = sum(block.data_count * 8 for block in blocks)
    if len(buffer) > bit_limit:
        raise exceptions.DataOverflowError(
            f"Code length overflow. Data size ({len(buffer)}) > "
            f"size available ({bit_limit})"
        )
    for _ in range(min(bit_limit - len(buffer), 4)):
        buffer.put_bit(False)
    remainder = len(buffer) % 8
    for _ in range(8 - remainder if remainder else 0):
        buffer.put_bit(False)
    for i in range((bit_limit - len(buffer)) // 8):
        buffer.put(PAD0 if i % 2 == 0 else PAD1, 8)
    return create_bytes(buffer, blocks)


def map_data(modules, data, mask_pattern: int, *, return_array=False):
    matrix = matrix_array(modules)
    if not isinstance(mask_pattern, int) or not 0 <= mask_pattern <= 7:
        raise ValueError("mask_pattern must be an integer in 0..7")
    raw_codewords = np.asarray(data)
    if raw_codewords.ndim != 1:
        raise ValueError("data must be a one-dimensional byte sequence")
    try:
        in_byte_range = np.all(
            (raw_codewords >= 0)
            & (raw_codewords <= 255)
            & (raw_codewords == np.floor(raw_codewords))
        )
    except (TypeError, ValueError):
        in_byte_range = False
    if not in_byte_range:
        raise ValueError("data values must be in 0..255")
    codewords = np.ascontiguousarray(raw_codewords, dtype=np.uint8)
    if codewords.size == 0:
        codewords = np.zeros(1, dtype=np.uint8)
        codeword_len = 0
    else:
        codeword_len = codewords.size
    status = lib().mqr_map_data(
        addr(matrix),
        matrix.size,
        matrix.shape[0],
        addr(codewords),
        codeword_len,
        mask_pattern,
    )
    if status != 0:
        raise RuntimeError(f"Mojo data-placement kernel failed with status {status}")
    if return_array:
        return matrix
    return matrix.astype(bool).tolist()
