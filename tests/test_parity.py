from __future__ import annotations

import io

import numpy as np
import pytest
import qrcode
import qrcode.base as upstream_base
import qrcode.util as upstream_util

import mojo_qrcode
from mojo_qrcode import base, constants, util
from mojo_qrcode.exceptions import DataOverflowError


def upstream_rs(data: bytes, nsym: int) -> bytes:
    generator = upstream_base.Polynomial([1], 0)
    for i in range(nsym):
        generator *= upstream_base.Polynomial([1, upstream_base.gexp(i)], 0)
    remainder = upstream_base.Polynomial(list(data), len(generator) - 1) % generator
    offset = len(remainder) - nsym
    return bytes(
        remainder[i + offset] if i + offset >= 0 else 0 for i in range(nsym)
    )


def test_public_constants_match_upstream():
    assert (
        constants.ERROR_CORRECT_L,
        constants.ERROR_CORRECT_M,
        constants.ERROR_CORRECT_Q,
        constants.ERROR_CORRECT_H,
    ) == (
        qrcode.ERROR_CORRECT_L,
        qrcode.ERROR_CORRECT_M,
        qrcode.ERROR_CORRECT_Q,
        qrcode.ERROR_CORRECT_H,
    )


def test_all_rs_block_layouts_match_upstream():
    assert base.RS_BLOCK_TABLE == upstream_base.RS_BLOCK_TABLE
    for version in range(1, 41):
        for level in range(4):
            assert base.rs_blocks(version, level) == upstream_base.rs_blocks(
                version, level
            )


@pytest.mark.parametrize("nsym", [7, 10, 13, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30])
@pytest.mark.parametrize("length", [1, 19, 121])
def test_reed_solomon_matches_upstream(nsym, length):
    data = np.random.default_rng(nsym * 1000 + length).integers(
        0, 256, length, dtype=np.uint8
    ).tobytes()
    assert util.rs_encode(data, nsym) == upstream_rs(data, nsym)


@pytest.mark.parametrize(
    "prefix,prefix_width",
    [(0, 0), (1, 1), (0b101, 3), (0b1011011, 7), (0xA5, 8)],
)
def test_bit_buffer_bulk_bytes_cross_byte_tail(prefix, prefix_width):
    values = bytes((i * 73 + 19) % 256 for i in range(23))
    ours = util.BitBuffer()
    reference = upstream_util.BitBuffer()
    ours.put(prefix, prefix_width)
    reference.put(prefix, prefix_width)
    ours.put_bytes(values)
    for value in values:
        reference.put(value, 8)
    assert len(ours) == len(reference)
    assert list(ours.buffer) == reference.buffer


@pytest.mark.parametrize("version", range(1, 41))
@pytest.mark.parametrize("level", range(4))
def test_codewords_match_upstream_all_versions(version, level):
    ours = util.create_data(version, level, [util.QRData(b"A")])
    reference = upstream_util.create_data(
        version, level, [upstream_util.QRData(b"A")]
    )
    assert ours == reference


@pytest.mark.parametrize(
    "payload,level",
    [
        ("8675309", qrcode.ERROR_CORRECT_M),
        ("HELLO WORLD", qrcode.ERROR_CORRECT_Q),
        ("lower-case and unicode: \u2603", qrcode.ERROR_CORRECT_H),
        (bytes(range(256)) * 3, qrcode.ERROR_CORRECT_L),
        (b"mixed-12345678901234567890-lower", qrcode.ERROR_CORRECT_M),
    ],
)
def test_auto_fit_and_selected_matrix_match_upstream(payload, level):
    ours = mojo_qrcode.QRCode(error_correction=level, border=0)
    reference = qrcode.QRCode(error_correction=level, border=0)
    ours.add_data(payload)
    reference.add_data(payload)
    ours.make()
    reference.make()
    assert ours.version == reference.version
    assert ours.data_cache == reference.data_cache
    assert ours.modules == reference.modules


@pytest.mark.parametrize("version", range(1, 41))
def test_fixed_mask_matrix_matches_upstream_all_versions(version):
    kwargs = dict(
        version=version,
        error_correction=qrcode.ERROR_CORRECT_M,
        border=0,
        mask_pattern=version % 8,
    )
    ours = mojo_qrcode.QRCode(**kwargs)
    reference = qrcode.QRCode(**kwargs)
    ours.add_data(b"A", optimize=0)
    reference.add_data(b"A", optimize=0)
    ours.make(fit=False)
    reference.make(fit=False)
    assert ours.modules == reference.modules


@pytest.mark.parametrize("mask", range(8))
def test_each_mask_matrix_matches_upstream(mask):
    kwargs = dict(
        version=10,
        error_correction=qrcode.ERROR_CORRECT_Q,
        border=0,
        mask_pattern=mask,
    )
    ours = mojo_qrcode.QRCode(**kwargs)
    reference = qrcode.QRCode(**kwargs)
    payload = b"Mojo mask parity 0123456789" * 4
    ours.add_data(payload, optimize=0)
    reference.add_data(payload, optimize=0)
    ours.make(fit=False)
    reference.make(fit=False)
    assert ours.modules == reference.modules


@pytest.mark.parametrize("size", [21, 45, 97, 177])
def test_mask_penalty_matches_upstream(size):
    matrix = np.random.default_rng(size).integers(
        0, 2, size=(size, size), dtype=np.uint8
    ).astype(bool).tolist()
    assert util.lost_point(matrix) == upstream_util.lost_point(matrix)


def test_mask_penalty_simd_remainder_matches_upstream():
    size = 23
    matrix = np.random.default_rng(2301).integers(
        0, 2, size=(size, size), dtype=np.uint8
    ).astype(bool).tolist()
    assert util.lost_point(matrix) == upstream_util.lost_point(matrix)


@pytest.mark.parametrize("modules", [[], [[0, 1]], [[0, 256], [1, 0]]])
def test_mask_penalty_rejects_unsafe_matrices(modules):
    with pytest.raises(ValueError):
        util.lost_point(modules)


def test_map_data_rejects_silent_byte_narrowing():
    modules = [[None] * 21 for _ in range(21)]
    for data in ([0, 256], [1.5], ["1"]):
        with pytest.raises(ValueError):
            util.map_data(modules, data, 0)
    with pytest.raises(ValueError):
        util.map_data(modules, [0], 8)


def test_ffi_rejects_invalid_lengths_without_dereferencing():
    ffi = util.lib()
    assert ffi.mqr_rs_encode(0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0) == -1
    assert ffi.mqr_lost_point(0, 0, 0) == -1


@pytest.mark.parametrize("version", [1, 7, 10, 20, 27, 40])
def test_bch_and_pattern_tables_match_upstream(version):
    assert util.pattern_position(version) == upstream_util.pattern_position(version)
    assert util.BCH_type_number(version) == upstream_util.BCH_type_number(version)
    for level in range(4):
        for mask in range(8):
            value = (level << 3) | mask
            assert util.BCH_type_info(value) == upstream_util.BCH_type_info(value)


def test_mode_selection_and_optimization_match_upstream():
    payload = "A1abc12345def1HELLOa"
    ours = mojo_qrcode.QRCode()
    reference = qrcode.QRCode()
    ours.add_data(payload, optimize=4)
    reference.add_data(payload, optimize=4)
    assert [(item.mode, item.data) for item in ours.data_list] == [
        (item.mode, item.data) for item in reference.data_list
    ]
    ours.make()
    reference.make()
    assert ours.version == reference.version == 2
    assert ours.modules == reference.modules


def test_qrdata_modes_and_repr():
    assert util.QRData("123").mode == util.MODE_NUMBER
    assert util.QRData("HELLO 42").mode == util.MODE_ALPHA_NUM
    assert util.QRData("lower").mode == util.MODE_8BIT_BYTE
    assert repr(util.QRData(b"hello")) == repr(b"hello")


def test_overflow_matches_upstream():
    ours = mojo_qrcode.QRCode(version=1)
    ours.add_data("abcdefghijklmno")
    with pytest.raises(DataOverflowError):
        ours.make(fit=False)


@pytest.mark.parametrize("version", [0, 41])
def test_invalid_version(version):
    with pytest.raises(ValueError):
        mojo_qrcode.QRCode(version=version)


@pytest.mark.parametrize("mask", [-1, 8, 42])
def test_invalid_mask_value(mask):
    with pytest.raises(ValueError):
        mojo_qrcode.QRCode(mask_pattern=mask)


def test_invalid_mask_type():
    with pytest.raises(TypeError):
        mojo_qrcode.QRCode(mask_pattern="0")


@pytest.mark.parametrize("border", [-1, -10])
def test_invalid_border(border):
    with pytest.raises(ValueError):
        mojo_qrcode.QRCode(border=border)


def test_get_matrix_border_matches_upstream():
    ours = mojo_qrcode.QRCode(border=3)
    reference = qrcode.QRCode(border=3)
    ours.add_data("border")
    reference.add_data("border")
    assert ours.get_matrix() == reference.get_matrix()


def test_pil_image_pixels_and_png_match_upstream():
    kwargs = dict(
        version=4,
        error_correction=qrcode.ERROR_CORRECT_H,
        box_size=3,
        border=2,
        mask_pattern=5,
    )
    ours = mojo_qrcode.QRCode(**kwargs)
    reference = qrcode.QRCode(**kwargs)
    ours.add_data("image parity", optimize=0)
    reference.add_data("image parity", optimize=0)
    ours_image = ours.make_image()
    reference_image = reference.make_image()
    assert np.array_equal(
        np.asarray(ours_image.get_image()), np.asarray(reference_image.get_image())
    )
    stream = io.BytesIO()
    ours_image.save(stream)
    assert stream.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_make_shortcut_returns_saveable_image():
    image = mojo_qrcode.make("https://example.com", box_size=2, border=1)
    assert image.pixel_size == (image.width + 2) * 2
    stream = io.BytesIO()
    image.save(stream)
    assert len(stream.getvalue()) > 100


def test_ascii_output_matches_upstream():
    ours = mojo_qrcode.QRCode(border=0, mask_pattern=2)
    reference = qrcode.QRCode(border=0, mask_pattern=2)
    ours.add_data("ASCII")
    reference.add_data("ASCII")
    ours_stream = io.StringIO()
    reference_stream = io.StringIO()
    ours.print_ascii(out=ours_stream)
    reference.print_ascii(out=reference_stream)
    assert ours_stream.getvalue() == reference_stream.getvalue()


class _TTYBuffer(io.StringIO):
    def isatty(self):
        return True


def test_tty_outputs_match_upstream():
    ours = mojo_qrcode.QRCode(border=2, mask_pattern=2)
    reference = qrcode.QRCode(border=2, mask_pattern=2)
    ours.add_data("TTY")
    reference.add_data("TTY")
    for method in ("print_ascii", "print_tty"):
        ours_stream = _TTYBuffer()
        reference_stream = _TTYBuffer()
        getattr(ours, method)(out=ours_stream)
        getattr(reference, method)(out=reference_stream)
        assert ours_stream.getvalue() == reference_stream.getvalue()


def test_clear_allows_reuse():
    qr = mojo_qrcode.QRCode()
    qr.add_data("first")
    qr.make()
    qr.clear()
    qr.add_data("second")
    qr.make()
    reference = qrcode.QRCode()
    reference.add_data("second")
    reference.make()
    assert qr.modules == reference.modules
