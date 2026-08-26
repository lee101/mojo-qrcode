from __future__ import annotations

import sys
from bisect import bisect_left
from typing import NamedTuple

import numpy as np

from . import constants, exceptions, util
from .image.base import BaseImage

precomputed_qr_blanks = {}
precomputed_qr_arrays = {}


def make(data=None, **kwargs):
    qr = QRCode(**kwargs)
    qr.add_data(data)
    return qr.make_image()


def _check_box_size(size):
    if int(size) <= 0:
        raise ValueError(f"Invalid box size (was {size}, expected larger than 0)")


def _check_border(size):
    if int(size) < 0:
        raise ValueError(
            f"Invalid border value (was {size}, expected 0 or larger than that)"
        )


def _check_mask_pattern(pattern):
    if pattern is None:
        return
    if not isinstance(pattern, int):
        raise TypeError(
            f"Invalid mask pattern (was {type(pattern)}, expected int)"
        )
    if pattern < 0 or pattern > 7:
        raise ValueError(f"Mask pattern should be in range(8) (got {pattern})")


def copy_2d_array(value):
    return [row[:] for row in value]


class ActiveWithNeighbors(NamedTuple):
    NW: bool
    N: bool
    NE: bool
    W: bool
    me: bool
    E: bool
    SW: bool
    S: bool
    SE: bool

    def __bool__(self):
        return self.me


class QRCode:
    def __init__(
        self,
        version=None,
        error_correction=constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
        image_factory=None,
        mask_pattern=None,
    ):
        _check_box_size(box_size)
        _check_border(border)
        self.version = version
        self.error_correction = int(error_correction)
        self.box_size = int(box_size)
        self.border = int(border)
        self.mask_pattern = mask_pattern
        self.image_factory = image_factory
        if image_factory is not None:
            assert issubclass(image_factory, BaseImage)
        self.clear()

    @property
    def version(self):
        if self._version is None:
            self.best_fit()
        return self._version

    @version.setter
    def version(self, value):
        if value is not None:
            value = int(value)
            util.check_version(value)
        self._version = value

    @property
    def mask_pattern(self):
        return self._mask_pattern

    @mask_pattern.setter
    def mask_pattern(self, pattern):
        _check_mask_pattern(pattern)
        self._mask_pattern = pattern

    def clear(self):
        self.modules = [[]]
        self.modules_count = 0
        self.data_cache = None
        self._data_array = None
        self.data_list = []

    def add_data(self, data, optimize=20):
        if isinstance(data, util.QRData):
            self.data_list.append(data)
        elif optimize:
            self.data_list.extend(util.optimal_data_chunks(data, minimum=optimize))
        else:
            self.data_list.append(util.QRData(data))
        self.data_cache = None
        self._data_array = None

    def make(self, fit=True):
        if fit or self._version is None:
            self.best_fit(start=self.version)
        if self.mask_pattern is None:
            self.makeImpl(False, self.best_mask_pattern())
        else:
            self.makeImpl(False, self.mask_pattern)

    def makeImpl(self, test, mask_pattern):
        self.modules_count = self.version * 4 + 17
        if self.version in precomputed_qr_arrays:
            self.modules = precomputed_qr_arrays[self.version].copy()
        else:
            self.modules = [
                [None] * self.modules_count for _ in range(self.modules_count)
            ]
            self.setup_position_probe_pattern(0, 0)
            self.setup_position_probe_pattern(self.modules_count - 7, 0)
            self.setup_position_probe_pattern(0, self.modules_count - 7)
            self.setup_position_adjust_pattern()
            self.setup_timing_pattern()
            precomputed_qr_blanks[self.version] = copy_2d_array(self.modules)
            precomputed_qr_arrays[self.version] = util.matrix_array(self.modules)
            self.modules = precomputed_qr_arrays[self.version].copy()

        self.setup_type_info(test, mask_pattern)
        if self.version >= 7:
            self.setup_type_number(test)
        self._ensure_data_cache()
        self.modules = util.map_data(
            self.modules,
            self._data_array,
            mask_pattern,
            return_array=test,
        )

    def _ensure_data_cache(self):
        if self.data_cache is None:
            self.data_cache = util.create_data(
                self.version, self.error_correction, self.data_list
            )
            self._data_array = np.asarray(self.data_cache, dtype=np.uint8)
        elif self._data_array is None:
            self._data_array = np.asarray(self.data_cache, dtype=np.uint8)

    def setup_position_probe_pattern(self, row, col):
        for r in range(-1, 8):
            if not 0 <= row + r < self.modules_count:
                continue
            for c in range(-1, 8):
                if not 0 <= col + c < self.modules_count:
                    continue
                dark = (
                    (0 <= r <= 6 and c in (0, 6))
                    or (0 <= c <= 6 and r in (0, 6))
                    or (2 <= r <= 4 and 2 <= c <= 4)
                )
                self.modules[row + r][col + c] = dark

    def best_fit(self, start=None):
        if start is None:
            start = 1
        util.check_version(start)
        mode_sizes = util.mode_sizes_for_version(start)
        buffer = util.BitBuffer()
        for data in self.data_list:
            buffer.put(data.mode, 4)
            buffer.put(len(data), mode_sizes[data.mode])
            data.write(buffer)
        self.version = bisect_left(
            util.BIT_LIMIT_TABLE[self.error_correction], len(buffer), start
        )
        if self.version == 41:
            raise exceptions.DataOverflowError()
        if mode_sizes is not util.mode_sizes_for_version(self.version):
            self.best_fit(start=self.version)
        return self.version

    def best_mask_pattern(self):
        self.modules_count = self.version * 4 + 17
        if self.version not in precomputed_qr_arrays:
            self.makeImpl(True, 0)
        else:
            self._ensure_data_cache()
        return util.best_mask_pattern(
            precomputed_qr_arrays[self.version], self._data_array
        )

    def setup_timing_pattern(self):
        for r in range(8, self.modules_count - 8):
            if self.modules[r][6] is None:
                self.modules[r][6] = r % 2 == 0
        for c in range(8, self.modules_count - 8):
            if self.modules[6][c] is None:
                self.modules[6][c] = c % 2 == 0

    def setup_position_adjust_pattern(self):
        positions = util.pattern_position(self.version)
        for row in positions:
            for col in positions:
                if self.modules[row][col] is not None:
                    continue
                for r in range(-2, 3):
                    for c in range(-2, 3):
                        self.modules[row + r][col + c] = (
                            abs(r) == 2 or abs(c) == 2 or (r == 0 and c == 0)
                        )

    def setup_type_number(self, test):
        bits = util.BCH_type_number(self.version)
        for i in range(18):
            value = not test and ((bits >> i) & 1) == 1
            self.modules[i // 3][i % 3 + self.modules_count - 11] = value
            self.modules[i % 3 + self.modules_count - 11][i // 3] = value

    def setup_type_info(self, test, mask_pattern):
        bits = util.BCH_type_info((self.error_correction << 3) | mask_pattern)
        for i in range(15):
            value = not test and ((bits >> i) & 1) == 1
            if i < 6:
                self.modules[i][8] = value
            elif i < 8:
                self.modules[i + 1][8] = value
            else:
                self.modules[self.modules_count - 15 + i][8] = value

            if i < 8:
                self.modules[8][self.modules_count - i - 1] = value
            elif i < 9:
                self.modules[8][15 - i] = value
            else:
                self.modules[8][14 - i] = value
        self.modules[self.modules_count - 8][8] = not test

    def map_data(self, data, mask_pattern):
        self.modules = util.map_data(self.modules, data, mask_pattern)

    def get_matrix(self):
        if self.data_cache is None:
            self.make()
        if not self.border:
            return self.modules
        width = len(self.modules) + self.border * 2
        result = [[False] * width for _ in range(self.border)]
        side = [False] * self.border
        result.extend(side + row + side for row in self.modules)
        result.extend([[False] * width for _ in range(self.border)])
        return result

    def make_image(self, image_factory=None, **kwargs):
        _check_box_size(self.box_size)
        if self.data_cache is None:
            self.make()
        if image_factory is not None:
            assert issubclass(image_factory, BaseImage)
        else:
            image_factory = self.image_factory
            if image_factory is None:
                from .image.pil import PilImage

                image_factory = PilImage
        image = image_factory(
            self.border,
            self.modules_count,
            self.box_size,
            qrcode_modules=self.modules,
            **kwargs,
        )
        if image.needs_drawrect:
            for row in range(self.modules_count):
                for col in range(self.modules_count):
                    if self.modules[row][col]:
                        image.drawrect(row, col)
        if image.needs_processing:
            image.process()
        return image

    def print_ascii(self, out=None, tty=False, invert=False):
        if out is None:
            out = sys.stdout
        if tty and not out.isatty():
            raise OSError("Not a tty")
        if self.data_cache is None:
            self.make()
        codes = [bytes((code,)).decode("cp437") for code in (255, 223, 220, 219)]
        if tty:
            invert = True
        if invert:
            codes.reverse()

        def get_module(row, col):
            if invert and self.border and max(row, col) >= self.modules_count + self.border:
                return 1
            if min(row, col) < 0 or max(row, col) >= self.modules_count:
                return 0
            return int(self.modules[row][col])

        for row in range(-self.border, self.modules_count + self.border, 2):
            if tty:
                if not invert or row < self.modules_count + self.border - 1:
                    out.write("\x1b[48;5;232m")
                out.write("\x1b[38;5;255m")
            for col in range(-self.border, self.modules_count + self.border):
                out.write(codes[get_module(row, col) + (get_module(row + 1, col) << 1)])
            if tty:
                out.write("\x1b[0m")
            out.write("\n")
        out.flush()

    def print_tty(self, out=None):
        if out is None:
            out = sys.stdout
        if not out.isatty():
            raise OSError("Not a tty")
        if self.data_cache is None:
            self.make()
        out.write("\x1b[1;47m" + " " * (self.modules_count * 2 + 4) + "\x1b[0m\n")
        for row in self.modules:
            out.write("\x1b[1;47m  \x1b[40m")
            for value in row:
                out.write("  " if value else "\x1b[1;47m  \x1b[40m")
            out.write("\x1b[1;47m  \x1b[0m\n")
        out.write("\x1b[1;47m" + " " * (self.modules_count * 2 + 4) + "\x1b[0m\n")
        out.flush()

    def is_constrained(self, row, col):
        return (
            0 <= row < len(self.modules)
            and 0 <= col < len(self.modules[row])
        )

    def active_with_neighbors(self, row, col):
        values = []
        for r in range(row - 1, row + 2):
            for c in range(col - 1, col + 2):
                values.append(self.is_constrained(r, c) and bool(self.modules[r][c]))
        return ActiveWithNeighbors(*values)
