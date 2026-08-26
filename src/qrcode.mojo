"""QR code Reed-Solomon, placement, and mask-scoring kernels."""

from std.sys.info import simd_width_of

comptime BytePtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def gf_mul(a: UInt8, b: UInt8, exp_table: BytePtr, log_table: BytePtr) -> UInt8:
    if a == 0 or b == 0:
        return 0
    return exp_table[Int(log_table[Int(a)]) + Int(log_table[Int(b)])]


def make_generator(
    degree: Int, generator: BytePtr, exp_table: BytePtr, log_table: BytePtr
):
    for i in range(degree + 1):
        generator[i] = 0
    generator[0] = 1
    for i in range(degree):
        var root = exp_table[i]
        for reverse in range(i + 1):
            var j = i + 1 - reverse
            generator[j] ^= gf_mul(
                generator[j - 1], root, exp_table, log_table
            )


def rs_remainder(
    data: BytePtr,
    data_len: Int,
    degree: Int,
    parity: BytePtr,
    generator: BytePtr,
    exp_table: BytePtr,
    log_table: BytePtr,
):
    make_generator(degree, generator, exp_table, log_table)
    rs_remainder_generated(
        data,
        data_len,
        degree,
        parity,
        generator,
        exp_table,
        log_table,
    )


def rs_remainder_generated(
    data: BytePtr,
    data_len: Int,
    degree: Int,
    parity: BytePtr,
    generator: BytePtr,
    exp_table: BytePtr,
    log_table: BytePtr,
):
    for i in range(degree):
        parity[i] = 0
    for i in range(data_len):
        var factor = data[i] ^ parity[0]
        for j in range(degree - 1):
            parity[j] = parity[j + 1]
        parity[degree - 1] = 0
        if factor != 0:
            for j in range(degree):
                parity[j] ^= gf_mul(
                    generator[j + 1], factor, exp_table, log_table
                )


@export("mqr_rs_encode")
def mqr_rs_encode(
    data_addr: Int,
    data_len: Int,
    degree: Int,
    parity_addr: Int,
    parity_len: Int,
    generator_addr: Int,
    generator_len: Int,
    exp_addr: Int,
    exp_len: Int,
    log_addr: Int,
    log_len: Int,
) abi("C") -> Int:
    if (
        data_addr == 0
        or data_len < 0
        or degree < 1
        or degree > 255
        or parity_addr == 0
        or parity_len < degree
        or generator_addr == 0
        or generator_len < degree + 1
        or exp_addr == 0
        or exp_len < 510
        or log_addr == 0
        or log_len < 256
    ):
        return -1
    rs_remainder(
        BytePtr(unsafe_from_address=data_addr),
        data_len,
        degree,
        BytePtr(unsafe_from_address=parity_addr),
        BytePtr(unsafe_from_address=generator_addr),
        BytePtr(unsafe_from_address=exp_addr),
        BytePtr(unsafe_from_address=log_addr),
    )
    return 0


@export("mqr_interleave_rs")
def mqr_interleave_rs(
    data_addr: Int,
    data_len: Int,
    dc_counts_addr: Int,
    ec_counts_addr: Int,
    block_count: Int,
    result_addr: Int,
    result_len: Int,
    ec_work_addr: Int,
    ec_work_len: Int,
    generator_addr: Int,
    generator_len: Int,
    exp_addr: Int,
    exp_len: Int,
    log_addr: Int,
    log_len: Int,
) abi("C") -> Int:
    if (
        data_addr == 0
        or data_len < 0
        or dc_counts_addr == 0
        or ec_counts_addr == 0
        or block_count < 1
        or result_addr == 0
        or result_len < 1
        or ec_work_addr == 0
        or generator_addr == 0
        or exp_addr == 0
        or exp_len < 510
        or log_addr == 0
        or log_len < 256
    ):
        return -1
    var data = BytePtr(unsafe_from_address=data_addr)
    var dc_counts = I64Ptr(unsafe_from_address=dc_counts_addr)
    var ec_counts = I64Ptr(unsafe_from_address=ec_counts_addr)
    var result = BytePtr(unsafe_from_address=result_addr)
    var ec_work = BytePtr(unsafe_from_address=ec_work_addr)
    var generator = BytePtr(unsafe_from_address=generator_addr)
    var exp_table = BytePtr(unsafe_from_address=exp_addr)
    var log_table = BytePtr(unsafe_from_address=log_addr)

    var max_dc = 0
    var max_ec = 0
    var expected_data = 0
    var expected_result = 0
    for block in range(block_count):
        var dc = Int(dc_counts[block])
        var ec = Int(ec_counts[block])
        if dc < 0 or ec < 1:
            return -2
        max_dc = max(max_dc, dc)
        max_ec = max(max_ec, ec)
        expected_data += dc
        expected_result += dc + ec
    if (
        expected_data > data_len
        or expected_result > result_len
        or ec_work_len < block_count * max_ec
        or generator_len < max_ec + 1
    ):
        return -3
    for block in range(block_count):
        if Int(ec_counts[block]) != max_ec:
            return -4

    var write_index = 0
    for column in range(max_dc):
        var data_offset = 0
        for block in range(block_count):
            var dc = Int(dc_counts[block])
            if column < dc:
                result[write_index] = data[data_offset + column]
                write_index += 1
            data_offset += dc

    make_generator(max_ec, generator, exp_table, log_table)
    var data_offset = 0
    for block in range(block_count):
        var dc = Int(dc_counts[block])
        var ec = Int(ec_counts[block])
        rs_remainder_generated(
            data + data_offset,
            dc,
            ec,
            ec_work + block * max_ec,
            generator,
            exp_table,
            log_table,
        )
        data_offset += dc

    for column in range(max_ec):
        for block in range(block_count):
            if column < Int(ec_counts[block]):
                result[write_index] = ec_work[block * max_ec + column]
                write_index += 1
    return write_index


def mask_applies(pattern: Int, row: Int, col: Int) -> Bool:
    var product = row * col
    if pattern == 0:
        return (row + col) % 2 == 0
    if pattern == 1:
        return row % 2 == 0
    if pattern == 2:
        return col % 3 == 0
    if pattern == 3:
        return (row + col) % 3 == 0
    if pattern == 4:
        return (row // 2 + col // 3) % 2 == 0
    if pattern == 5:
        return product % 2 + product % 3 == 0
    if pattern == 6:
        return (product % 2 + product % 3) % 2 == 0
    return (product % 3 + (row + col) % 2) % 2 == 0


def map_data_kernel(
    matrix: BytePtr,
    size: Int,
    data: BytePtr,
    data_len: Int,
    pattern: Int,
) -> Int:
    var direction = -1
    var row = size - 1
    var bit_index = 7
    var byte_index = 0
    var col = size - 1

    while col > 0:
        if col == 6:
            col -= 1
        while True:
            for offset in range(2):
                var c = col - offset
                if matrix[row * size + c] == 2:
                    var dark = False
                    if byte_index < data_len:
                        dark = (
                            (data[byte_index] >> UInt8(bit_index)) & UInt8(1)
                        ) == UInt8(1)
                    if mask_applies(pattern, row, c):
                        dark = not dark
                    matrix[row * size + c] = 1 if dark else 0
                    bit_index -= 1
                    if bit_index < 0:
                        byte_index += 1
                        bit_index = 7
            row += direction
            if row < 0 or row >= size:
                row -= direction
                direction = -direction
                break
        col -= 2
    return 0


@export("mqr_map_data")
def mqr_map_data(
    matrix_addr: Int,
    matrix_len: Int,
    size: Int,
    data_addr: Int,
    data_len: Int,
    pattern: Int,
) abi("C") -> Int:
    if (
        matrix_addr == 0
        or size < 1
        or matrix_len < size * size
        or data_addr == 0
        or data_len < 0
        or pattern < 0
        or pattern > 7
    ):
        return -1
    return map_data_kernel(
        BytePtr(unsafe_from_address=matrix_addr),
        size,
        BytePtr(unsafe_from_address=data_addr),
        data_len,
        pattern,
    )


def same(matrix: BytePtr, a: Int, b: Int) -> Bool:
    return matrix[a] == matrix[b]


def lost_point_kernel(matrix: BytePtr, size: Int) -> Int:
    var penalty = 0

    for row in range(size):
        var run = 1
        for col in range(1, size):
            if same(matrix, row * size + col, row * size + col - 1):
                run += 1
            else:
                if run >= 5:
                    penalty += run - 2
                run = 1
        if run >= 5:
            penalty += run - 2

    for col in range(size):
        var run = 1
        for row in range(1, size):
            if same(matrix, row * size + col, (row - 1) * size + col):
                run += 1
            else:
                if run >= 5:
                    penalty += run - 2
                run = 1
        if run >= 5:
            penalty += run - 2

    for row in range(size - 1):
        for col in range(size - 1):
            var v = matrix[row * size + col]
            if (
                matrix[row * size + col + 1] == v
                and matrix[(row + 1) * size + col] == v
                and matrix[(row + 1) * size + col + 1] == v
            ):
                penalty += 3

    for row in range(size):
        for col in range(size - 10):
            if (
                matrix[row * size + col] == 1
                and matrix[row * size + col + 1] == 0
                and matrix[row * size + col + 2] == 1
                and matrix[row * size + col + 3] == 1
                and matrix[row * size + col + 4] == 1
                and matrix[row * size + col + 5] == 0
                and matrix[row * size + col + 6] == 1
                and matrix[row * size + col + 7] == 0
                and matrix[row * size + col + 8] == 0
                and matrix[row * size + col + 9] == 0
                and matrix[row * size + col + 10] == 0
            ) or (
                matrix[row * size + col] == 0
                and matrix[row * size + col + 1] == 0
                and matrix[row * size + col + 2] == 0
                and matrix[row * size + col + 3] == 0
                and matrix[row * size + col + 4] == 1
                and matrix[row * size + col + 5] == 0
                and matrix[row * size + col + 6] == 1
                and matrix[row * size + col + 7] == 1
                and matrix[row * size + col + 8] == 1
                and matrix[row * size + col + 9] == 0
                and matrix[row * size + col + 10] == 1
            ):
                penalty += 40

    for col in range(size):
        for row in range(size - 10):
            if (
                matrix[row * size + col] == 1
                and matrix[(row + 1) * size + col] == 0
                and matrix[(row + 2) * size + col] == 1
                and matrix[(row + 3) * size + col] == 1
                and matrix[(row + 4) * size + col] == 1
                and matrix[(row + 5) * size + col] == 0
                and matrix[(row + 6) * size + col] == 1
                and matrix[(row + 7) * size + col] == 0
                and matrix[(row + 8) * size + col] == 0
                and matrix[(row + 9) * size + col] == 0
                and matrix[(row + 10) * size + col] == 0
            ) or (
                matrix[row * size + col] == 0
                and matrix[(row + 1) * size + col] == 0
                and matrix[(row + 2) * size + col] == 0
                and matrix[(row + 3) * size + col] == 0
                and matrix[(row + 4) * size + col] == 1
                and matrix[(row + 5) * size + col] == 0
                and matrix[(row + 6) * size + col] == 1
                and matrix[(row + 7) * size + col] == 1
                and matrix[(row + 8) * size + col] == 1
                and matrix[(row + 9) * size + col] == 0
                and matrix[(row + 10) * size + col] == 1
            ):
                penalty += 40

    var dark = 0
    comptime W = simd_width_of[DType.float64]()
    var module_count = size * size
    var vector_end = module_count - module_count % W
    for i in range(0, vector_end, W):
        dark += Int(matrix.load[width=W](i).reduce_add())
    for i in range(vector_end, module_count):
        dark += 1 if matrix[i] == 1 else 0
    var imbalance = abs(dark * 100 - size * size * 50)
    penalty += (imbalance // (size * size * 5)) * 10
    return penalty


@export("mqr_lost_point")
def mqr_lost_point(matrix_addr: Int, matrix_len: Int, size: Int) abi("C") -> Int:
    if matrix_addr == 0 or size < 1 or matrix_len < size * size:
        return -1
    return lost_point_kernel(
        BytePtr(unsafe_from_address=matrix_addr), size
    )


def copy_matrix(source: BytePtr, destination: BytePtr, count: Int):
    comptime W = simd_width_of[DType.float64]()
    var vector_end = count - count % W
    for i in range(0, vector_end, W):
        destination.store(i, source.load[width=W](i))
    for i in range(vector_end, count):
        destination[i] = source[i]


def clear_test_type_info(matrix: BytePtr, size: Int):
    for i in range(15):
        if i < 6:
            matrix[i * size + 8] = 0
        elif i < 8:
            matrix[(i + 1) * size + 8] = 0
        else:
            matrix[(size - 15 + i) * size + 8] = 0

        if i < 8:
            matrix[8 * size + size - i - 1] = 0
        elif i < 9:
            matrix[8 * size + 15 - i] = 0
        else:
            matrix[8 * size + 14 - i] = 0
    matrix[(size - 8) * size + 8] = 0
    if size >= 45:
        for i in range(18):
            matrix[(i // 3) * size + i % 3 + size - 11] = 0
            matrix[(i % 3 + size - 11) * size + i // 3] = 0


@export("mqr_best_mask")
def mqr_best_mask(
    template_addr: Int,
    template_len: Int,
    size: Int,
    data_addr: Int,
    data_len: Int,
    work_addr: Int,
    work_len: Int,
) abi("C") -> Int:
    var module_count = size * size
    if (
        template_addr == 0
        or size < 1
        or template_len < module_count
        or data_addr == 0
        or data_len < 0
        or work_addr == 0
        or work_len < module_count
    ):
        return -1
    var template = BytePtr(unsafe_from_address=template_addr)
    var data = BytePtr(unsafe_from_address=data_addr)
    var work = BytePtr(unsafe_from_address=work_addr)
    var best_pattern = 0
    var best_penalty = 0
    for pattern in range(8):
        copy_matrix(template, work, module_count)
        clear_test_type_info(work, size)
        _ = map_data_kernel(work, size, data, data_len, pattern)
        var penalty = lost_point_kernel(work, size)
        if pattern == 0 or penalty < best_penalty:
            best_penalty = penalty
            best_pattern = pattern
    return best_pattern
