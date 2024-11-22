"""SHA3 implementation in Python in functional style"""

import numpy as np
import time
from numba import cuda, uint8, uint32, uint64

# Fixed rate for Keccak-256
RATE = 136
BIT_LENGTH = 256

# Keccak round constants
_KECCAK_RC = np.array(
    [
        0x0000000000000001,
        0x0000000000008082,
        0x800000000000808A,
        0x8000000080008000,
        0x000000000000808B,
        0x0000000080000001,
        0x8000000080008081,
        0x8000000000008009,
        0x000000000000008A,
        0x0000000000000088,
        0x0000000080008009,
        0x000000008000000A,
        0x000000008000808B,
        0x800000000000008B,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x000000000000800A,
        0x800000008000000A,
        0x8000000080008081,
        0x8000000000008080,
        0x0000000080000001,
        0x8000000080008008,
    ],
    dtype=np.uint64,
)

# Domain separation byte
_DSBYTE = 0x01

# Number of Keccak rounds
_NUM_ROUNDS = 24


@cuda.jit(device=True, inline=True)
def _rol(x: uint64, s: uint64) -> uint64: # type: ignore
    """
    Rotates x left by s

    Args:
        x (int): The input value to rotate
        s (int): The number of bits to rotate by

    Returns:
        int: The rotated value
    """
    return (uint64(x) << uint64(s)) ^ (uint64(x) >> uint64(64 - s))


@cuda.jit(device=True, inline=True)
def _keccak_f(state: np.ndarray) -> np.ndarray:
    """
    The keccak_f permutation function, unrolled for performance

    Args:
        state (device array): The state array of the SHA-3 sponge construction

    Returns:
        device array: The updated state array after permutation
    """

    # 24 rounds of permutation
    for i in range(_NUM_ROUNDS):
        # Parity calculation unrolled
        bc0 = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20]
        bc1 = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21]
        bc2 = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22]
        bc3 = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23]
        bc4 = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24]

        # Theta unrolled
        t0 = bc4 ^ _rol(bc1, 1)
        t1 = bc0 ^ _rol(bc2, 1)
        t2 = bc1 ^ _rol(bc3, 1)
        t3 = bc2 ^ _rol(bc4, 1)
        t4 = bc3 ^ _rol(bc0, 1)

        state[0] ^= t0
        state[5] ^= t0
        state[10] ^= t0
        state[15] ^= t0
        state[20] ^= t0

        state[1] ^= t1
        state[6] ^= t1
        state[11] ^= t1
        state[16] ^= t1
        state[21] ^= t1

        state[2] ^= t2
        state[7] ^= t2
        state[12] ^= t2
        state[17] ^= t2
        state[22] ^= t2

        state[3] ^= t3
        state[8] ^= t3
        state[13] ^= t3
        state[18] ^= t3
        state[23] ^= t3

        state[4] ^= t4
        state[9] ^= t4
        state[14] ^= t4
        state[19] ^= t4
        state[24] ^= t4

        # Rho and Pi unrolled
        t1 = _rol(state[1], 1)
        t2 = _rol(state[10], 3)
        t3 = _rol(state[7], 6)
        t4 = _rol(state[11], 10)
        t5 = _rol(state[17], 15)
        t6 = _rol(state[18], 21)
        t7 = _rol(state[3], 28)
        t8 = _rol(state[5], 36)
        t9 = _rol(state[16], 45)
        t10 = _rol(state[8], 55)
        t11 = _rol(state[21], 2)
        t12 = _rol(state[24], 14)
        t13 = _rol(state[4], 27)
        t14 = _rol(state[15], 41)
        t15 = _rol(state[23], 56)
        t16 = _rol(state[19], 8)
        t17 = _rol(state[13], 25)
        t18 = _rol(state[12], 43)
        t19 = _rol(state[2], 62)
        t20 = _rol(state[20], 18)
        t21 = _rol(state[14], 39)
        t22 = _rol(state[22], 61)
        t23 = _rol(state[9], 20)
        t24 = _rol(state[6], 44)

        state[10] = t1
        state[7] = t2
        state[11] = t3
        state[17] = t4
        state[18] = t5
        state[3] = t6
        state[5] = t7
        state[16] = t8
        state[8] = t9
        state[21] = t10
        state[24] = t11
        state[4] = t12
        state[15] = t13
        state[23] = t14
        state[19] = t15
        state[13] = t16
        state[12] = t17
        state[2] = t18
        state[20] = t19
        state[14] = t20
        state[22] = t21
        state[9] = t22
        state[6] = t23
        state[1] = t24

        # Chi unrolled
        t0 = state[0] ^ ((~state[1]) & state[2])
        t1 = state[1] ^ ((~state[2]) & state[3])
        t2 = state[2] ^ ((~state[3]) & state[4])
        t3 = state[3] ^ ((~state[4]) & state[0])
        t4 = state[4] ^ ((~state[0]) & state[1])
        t5 = state[5] ^ ((~state[6]) & state[7])
        t6 = state[6] ^ ((~state[7]) & state[8])
        t7 = state[7] ^ ((~state[8]) & state[9])
        t8 = state[8] ^ ((~state[9]) & state[5])
        t9 = state[9] ^ ((~state[5]) & state[6])
        t10 = state[10] ^ ((~state[11]) & state[12])
        t11 = state[11] ^ ((~state[12]) & state[13])
        t12 = state[12] ^ ((~state[13]) & state[14])
        t13 = state[13] ^ ((~state[14]) & state[10])
        t14 = state[14] ^ ((~state[10]) & state[11])
        t15 = state[15] ^ ((~state[16]) & state[17])
        t16 = state[16] ^ ((~state[17]) & state[18])
        t17 = state[17] ^ ((~state[18]) & state[19])
        t18 = state[18] ^ ((~state[19]) & state[15])
        t19 = state[19] ^ ((~state[15]) & state[16])
        t20 = state[20] ^ ((~state[21]) & state[22])
        t21 = state[21] ^ ((~state[22]) & state[23])
        t22 = state[22] ^ ((~state[23]) & state[24])
        t23 = state[23] ^ ((~state[24]) & state[20])
        t24 = state[24] ^ ((~state[20]) & state[21])

        state[0] = t0
        state[1] = t1
        state[2] = t2
        state[3] = t3
        state[4] = t4
        state[5] = t5
        state[6] = t6
        state[7] = t7
        state[8] = t8
        state[9] = t9
        state[10] = t10
        state[11] = t11
        state[12] = t12
        state[13] = t13
        state[14] = t14
        state[15] = t15
        state[16] = t16
        state[17] = t17
        state[18] = t18
        state[19] = t19
        state[20] = t20
        state[21] = t21
        state[22] = t22
        state[23] = t23
        state[24] = t24

        state[0] ^= _KECCAK_RC[i]

    return state


@cuda.jit(device=True, inline=True)
def _absorb(
    state: np.ndarray, data: bytes, buf: np.ndarray, buf_idx: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Absorbs input data into the sponge construction

    Args:
        state (device array): The state array of the SHA-3 sponge construction
        rate (int): The rate of the sponge function
        data (device array): The input data to be absorbed
        buf (device array): The buffer to absorb the input into
        buf_idx (int): Current index in the buffer

    Returns:
        device array: The updated state array after absorbing
        device array: The updated buffer after absorbing
        int: The updated index in the buffer
    """
    todo = len(data)
    i = 0
    while todo > 0:
        cando = RATE - buf_idx
        willabsorb = min(cando, todo)
        for j in range(willabsorb):
            # Directly manipulate each byte rather than using numpy operations
            buf[buf_idx + j] ^= data[i + j]
        buf_idx += willabsorb
        if buf_idx == RATE:
            # Ensure _permute is also device-friendly
            state, buf, buf_idx = _permute(state, buf, buf_idx)
        todo -= willabsorb
        i += willabsorb

    return state, buf, buf_idx


@cuda.jit(device=True, inline=True)
def _squeeze(state: np.ndarray, buf: np.ndarray, buf_idx: int, output_ptr: np.ndarray):
    """
    Performs the squeeze operation of the sponge construction

    Args:
        state (device array): The state array of the SHA-3 sponge construction
        buf (device array): The buffer to squeeze the output into
        buf_idx (int): Current index in the buffer
        output_ptr (device array): Pointer to where the hash output should be written
    """

    tosqueeze = BIT_LENGTH // 8
    local_output_idx = 0  # Tracks where to insert bytes into output_buf

    # Squeeze output
    while tosqueeze > 0:
        cansqueeze = RATE - buf_idx
        willsqueeze = min(cansqueeze, tosqueeze)

        # Extract bytes from state and directly update output_buf
        for _ in range(willsqueeze):
            byte_index = buf_idx % 8
            byte_val = (state[buf_idx // 8] >> (byte_index * 8)) & 0xFF

            output_ptr[local_output_idx] = byte_val

            buf_idx += 1
            local_output_idx += 1

            # If we've processed a full rate's worth of data, permute
            if buf_idx == RATE:
                state, buf, buf_idx = _permute(state, buf, 0)

        tosqueeze -= willsqueeze


@cuda.jit(device=True, inline=True)
def _pad(
    state: np.ndarray, buf: np.ndarray, buf_idx: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Pads the input data in the buffer.

    Args:
        state (device array): The state array of the SHA-3 sponge construction
        buf (device array): The buffer to pad the input into
        buf_idx (int): Current index in the buffer

    Returns:
        device array: The updated state array after padding
        device array: The updated buffer after padding
        int: The updated index in the buffer
    """
    buf[buf_idx] ^= _DSBYTE
    buf[RATE - 1] ^= 0x80
    return _permute(state, buf, buf_idx)


@cuda.jit(device=True, inline=True)
def _permute(state, buf, buf_idx):
    """
    Permutes the internal state and buffer for thorough mixing.

    Args:
        state (device array): The state array of the SHA-3 sponge construction
        buf (device array): The buffer to permute
        buf_idx (int): Current index in the buffer

    Returns:
        device array: The updated state array after permutation
        device array: The updated buffer after permutation
        int: The updated index in the buffer
    """
    # Create a temporary state array
    temp_state = cuda.local.array(25, dtype=uint64)
    for i in range(25):
        temp_state[i] = 0

    # Process bytes to uint64
    for i in range(0, len(buf), 8):
        if i + 8 <= len(buf):  # Ensure there's enough data to read
            uint64_val = uint64(0)
            for j in range(8):
                uint64_val |= uint64(buf[i + j]) << (j * 8)
            temp_state[i // 8] = uint64_val

    # Manually perform bitwise XOR for each element
    for i in range(25):
        state[i] ^= temp_state[i]

    # Perform Keccak permutation
    state = _keccak_f(state)

    # Reset buf_idx and buf
    buf_idx = 0
    for i in range(200):
        buf[i] = 0

    return state, buf, buf_idx


@cuda.jit(device=True)
def keccak256_single_wrapped(data: bytes, output_ptr: np.ndarray):
    state = cuda.local.array(25, dtype=uint64)
    buf = cuda.local.array(200, dtype=uint8)
    keccak256_single(data, output_ptr, state, buf)


@cuda.jit(device=True)
def keccak256_single(data: bytes, # input, 85xu8
                     output_ptr: np.ndarray, # output, 32xu8
                     state: np.ndarray, # local, 25xu64
                     buf: np.ndarray): # local, 200xu8
    """Computes a single Keccak256 hash on the device

    Args:
        data (bytes): Input data to hash
        output_buf (array): Buffer to write the hash result to
        output_idx (int): Index in output buffer to write the result
    """
    buf_idx = 0

    # Initialize state and buffer
    for i in range(25):
        state[i] = 0
    for i in range(200):
        buf[i] = 0

    # Absorb data
    state, buf, buf_idx = _absorb(state, data, buf, buf_idx)

    # Pad
    state, buf, buf_idx = _pad(state, buf, buf_idx)

    # Squeeze
    _squeeze(state, buf, buf_idx, output_ptr)


@cuda.jit(device=True)
def score_func_leading_zeros(hash) -> np.int32:
    score = np.int32(0)
    for i in range(12, 32):  # hash[12:] in bytes
        if hash[i] != 0:
            break
        score += 1
    return score


@cuda.jit(device=True)
def score_func_uniswap_v4(hash: np.ndarray) -> np.int32:
    """
    Approximate score function for Uniswap v4 (optimized for CUDA)

    10 points for every leading 0 nibble
    40 points if the first 4 is followed by 3 more 4s
    20 points if the first nibble after the four 4s is NOT a 4
    20 points if the last 4 nibbles are 4s
    1 point for every 4

    Note: the address starts at hash[12:]
    """

    # Fast exit if the first 4 bytes are not zero
    word0 = (
        (uint32(hash[12]) << 24)
        | (uint32(hash[13]) << 16)
        | (uint32(hash[14]) << 8)
        | uint32(hash[15])
    )
    clz_word0 = cuda.clz(uint32(word0))

    # XXX avoid short-circuiting for verification purposes
    # if word0 != 0:
    #     return 0

    leading_zero_bits = clz_word0

    if clz_word0 == 32:
        # Count leading zero nibbles starting from byte 4
        # Combine bytes 4-7 into a 32-bit word
        word1 = (
            (uint32(hash[16]) << 24)
            | (uint32(hash[17]) << 16)
            | (uint32(hash[18]) << 8)
            | uint32(hash[19])
        )

        # Use CUDA clz to count leading zero bits in word1
        clz_word1 = cuda.clz(uint32(word1))
        leading_zero_bits += clz_word1

    leading_zero_nibbles = leading_zero_bits >> 2  # Divide by 4
    score = leading_zero_nibbles * 10

    nibble_idx = 24 + leading_zero_nibbles

    # Check for four consecutive 4s starting at nibble_idx
    # Create a mask based on whether nibble_idx is odd or even
    byte_idx = nibble_idx >> 1
    is_odd = nibble_idx & 1
    shift_amount = 8 - is_odd * 4

    # Combine 3 bytes and shift based on alignment
    overextended = (
        (uint32(hash[byte_idx]) << 16)
        | (uint32(hash[byte_idx + 1]) << 8)
        | uint32(hash[byte_idx + 2])
    )
    shifted = overextended >> shift_amount

    # Check for four consecutive 4s starting at nibble_idx
    four_fours = (shifted & 0xFFFF) == 0x4444
    if not four_fours:
        return 0

    score += 40

    # Check next nibble
    next_nibble = (overextended & 0xFF) >> (shift_amount - 4)
    score += (next_nibble != 0x4) * 20

    # NOTE: skipping this on device to save some time
    # ---
    # add 1 point for every 4 nibble
    # num_fours = 0
    # for i in range(12, 32):
    #     byte = hash[i]
    #     if byte >> 4 == 0x4:
    #         num_fours += 1
    #     if byte & 0xF == 0x4:
    #         num_fours += 1

    # if the last 4 nibbles are 4s
    if hash[30] == 0x44 and hash[31] == 0x44:
        score += 20

    return score


@cuda.jit
def create2_kernel(deployer_addr, salt, initcode_hash, output):
    data = cuda.local.array(85, dtype=uint8)
    data[0] = 0xFF
    for i in range(20):
        data[1 + i] = deployer_addr[i]
    for i in range(32):
        data[21 + i] = salt[i]
    for i in range(32):
        data[53 + i] = initcode_hash[i]

    keccak256_single_wrapped(data, output)


# can be called only from host code
@cuda.jit
def create2_search_kernel(
    input_template: np.ndarray,  # 85-byte template
    hashes_per_thread: int,
    global_best_scores: np.ndarray,  # array of shape=(blocks_per_grid, type=np.int32)
    global_best_salts: np.ndarray,  # array of shape=(blocks_per_grid, type=np.uint64)
):
    block_index = cuda.blockIdx.x
    thread_index = cuda.threadIdx.x  # index in the block, e.g. between 0 and 255
    threads_per_block = cuda.blockDim.x

    # allocate shared memory once per block
    smem_best_scores = cuda.shared.array(shape=(256,), dtype=np.int32)
    smem_best_salts = cuda.shared.array(shape=(256,), dtype=np.uint64)

    thread_best_score = np.int32(0)
    thread_best_salt = np.uint64(0)

    # copy the template from global to local memory
    data = cuda.local.array(85, dtype=uint8)
    for i in range(85):
        data[i] = input_template[i]

    # write block index as a 2-byte big endian integer
    data[45] = (block_index >> 8) & 0xFF
    data[46] = block_index & 0xFF

    # write thread index as a 2-byte big endian integer
    data[47] = (thread_index >> 8) & 0xFF
    data[48] = thread_index & 0xFF

    # allocate local array to store the hash output
    hash_output = cuda.local.array(32, dtype=uint8)

    # allocate local arrays to store the state and buffer out of the main loop
    state = cuda.local.array(25, dtype=uint64)
    buf = cuda.local.array(200, dtype=uint8)

    # global thread id, between 0 and total_number_of_threads
    for idx in range(hashes_per_thread):
        # fill in the template:
        # write idx in big endian at the last 4 bytes of salt
        data[49] = (idx >> 24) & 0xFF
        data[50] = (idx >> 16) & 0xFF
        data[51] = (idx >> 8) & 0xFF
        data[52] = idx & 0xFF

        # Compute hash
        keccak256_single(data, hash_output, state, buf)

        # Score the hash
        score = score_func_uniswap_v4(hash_output)
        if score > thread_best_score and score > 1:
            thread_best_score = score
            thread_best_salt = (
                (np.uint64(block_index) << 48)
                | (np.uint64(thread_index) << 32)
                | np.uint64(idx)
            )

    # Write per-thread best score and salt to block-level shared memory
    smem_best_scores[thread_index] = thread_best_score
    smem_best_salts[thread_index] = thread_best_salt
    cuda.syncthreads()

    # Reduction within block to find the best score and salt
    stride = threads_per_block // 2
    while stride > 0:
        if thread_index < stride:
            if smem_best_scores[thread_index + stride] > smem_best_scores[thread_index]:
                smem_best_scores[thread_index] = smem_best_scores[thread_index + stride]
                smem_best_salts[thread_index] = smem_best_salts[thread_index + stride]
        cuda.syncthreads()
        stride //= 2

    # Write block's best score and salt to global memory
    if thread_index == 0:
        global_best_scores[block_index] = smem_best_scores[0]
        global_best_salts[block_index] = smem_best_salts[0]


# host function
def create2_search(
    deployer_addr: bytes,
    initcode_hash: bytes,
    salt_prefix: bytes = b"\x00" * 20,
    num_hashes: int = 2**20,
    threads_per_block=256,
    hashes_per_thread=256,
) -> tuple[int, bytes]:
    # Calculate total number of threads and blocks
    total_threads = (num_hashes + hashes_per_thread - 1) // hashes_per_thread
    blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block

    # Prepare input
    assert len(salt_prefix) == 20
    run_id = int(time.time()).to_bytes(4, "big")

    input_template = (
        b"\xff" + deployer_addr + salt_prefix + run_id + b"\x00" * 8 + initcode_hash
    )
    assert len(input_template) == 85

    # Copy data to device
    input_template_d = to_device_array(input_template)

    # Allocate output arrays
    best_scores_d = cuda.device_array(shape=(blocks_per_grid,), dtype=np.int32)
    best_salts_d = cuda.device_array(shape=(blocks_per_grid,), dtype=np.uint64)

    # Launch the kernel
    create2_search_kernel[blocks_per_grid, threads_per_block](
        input_template_d,
        hashes_per_thread,
        best_scores_d,
        best_salts_d,
    )

    # Copy back the per-block best scores and salts
    best_scores_h = best_scores_d.copy_to_host()
    best_salts_h = best_salts_d.copy_to_host()

    # Find the overall best
    best_score = -1
    best_salt = None
    for i in range(blocks_per_grid):
        if best_scores_h[i] > best_score:
            best_score = best_scores_h[i]
            best_salt = best_salts_h[i]

    full_salt = salt_prefix + run_id + int(best_salt).to_bytes(8, "big")
    assert len(full_salt) == 32
    return int(best_score), full_salt


def create2_helper(deployer_addr: bytes, salt: bytes, initcode_hash: bytes) -> bytes:
    assert len(deployer_addr) == 20
    assert len(salt) == 32
    assert len(initcode_hash) == 32

    deployer_addr_d = to_device_array(deployer_addr)
    salt_d = to_device_array(salt)
    initcode_hash_d = to_device_array(initcode_hash)
    hash_output_d = cuda.device_array(32, dtype=np.uint8)

    create2_kernel[1, 1](deployer_addr_d, salt_d, initcode_hash_d, hash_output_d)
    hash_output_h = hash_output_d.copy_to_host()
    return bytes(hash_output_h)[12:]


def to_device_array(data: bytes) -> np.ndarray:
    return cuda.to_device(np.frombuffer(data, dtype=np.uint8))


@cuda.jit
def score_kernel(hash: np.ndarray, score_out: np.ndarray):
    """Kernel must modify its arguments instead of returning values"""
    score_out[0] = score_func_uniswap_v4(hash)


def score_address(addr_bytes: bytes) -> int:
    hash_d = to_device_array(addr_bytes)
    score_d = cuda.device_array(1, dtype=np.int32)
    score_kernel[1, 1](hash_d, score_d)
    return int(score_d.copy_to_host()[0])


def check_expected_score(addr_hexstr: str, expected_score: int):
    assert len(addr_hexstr) == 40 or len(addr_hexstr) == 42
    addr_hexstr = addr_hexstr[2:] if addr_hexstr.startswith("0x") else addr_hexstr
    left_pad = b"\x00" * 12
    hash = left_pad + bytes.fromhex(addr_hexstr)
    score = score_address(hash)
    print(f"checking that {hash.hex()} scores {score} == {expected_score} ", end="")
    if score == expected_score:
        print("✅")
    else:
        print(f"❌ {score=} != {expected_score=}")


def check_expected_addr(
    deployer_addr: bytes, salt: bytes, initcode_hash: bytes, expected_addr: bytes
):
    deployer_addr_d = to_device_array(deployer_addr)
    salt_d = to_device_array(salt)
    initcode_hash_d = to_device_array(initcode_hash)
    hash_output_d = cuda.device_array(32, dtype=np.uint8)

    create2_kernel[1, 1](deployer_addr_d, salt_d, initcode_hash_d, hash_output_d)
    hash_output_h = hash_output_d.copy_to_host()
    actual_hash = bytes(hash_output_h)

    print(
        f"checking that create2(deployer_addr={deployer_addr.hex()}, salt={salt.hex()}, initcode_hash={initcode_hash.hex()}) == {expected_addr.hex()} ",
        end="",
    )

    if actual_hash[12:] == expected_addr:
        print("✅")
    else:
        print("❌")
        print(f"{expected_addr.hex()=}")
        print(f"{actual_hash.hex()=}")


def main():
    zero_addr = b"\x00" * 20
    zero_bytes32 = b"\x00" * 32
    empty_hash = bytes.fromhex(
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    )
    check_expected_addr(
        zero_addr,
        zero_bytes32,
        zero_bytes32,
        bytes.fromhex("FFc4f52f884a02bCd5716744cD622127366F2edf"),
    )
    check_expected_addr(
        zero_addr,
        zero_bytes32,
        empty_hash,
        bytes.fromhex("E33C0C7F7df4809055C3ebA6c09CFe4BaF1BD9e0"),
    )

    check_expected_score("0x0000000000d34444cB22EA006470e100Eb014F2D", 0)
    check_expected_score("0x00000000004444d3cB22EA006470e100Eb014F2D", 160)
    check_expected_score("0x000000000444406D3bBA81Cd60aecDd06166f136", 150)

    print("measuring throughput... ")

    mp_count = cuda.current_context().device.MULTIPROCESSOR_COUNT
    threads_per_block = 256
    hashes_per_thread = 256
    num_hashes = threads_per_block * hashes_per_thread * mp_count * 8

    for hashes_per_thread in [2**i for i in range(10, 20)]:
        start_time = time.perf_counter()
        best_score, best_salt = create2_search(
            zero_addr,
            empty_hash,
            num_hashes=num_hashes,
            threads_per_block=threads_per_block,
            hashes_per_thread=hashes_per_thread,
        )
        end_time = time.perf_counter()
        throughput = num_hashes / (end_time - start_time)
        print(f"{throughput:,.0f} salts/sec with {hashes_per_thread} hashes/thread")


if __name__ == "__main__":
    main()
