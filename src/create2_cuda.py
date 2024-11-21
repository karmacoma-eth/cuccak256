"""SHA3 implementation in Python in functional style"""

import numpy as np
from numba import cuda, uint8, uint64

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


@cuda.jit(device=True)
def _rol(x: uint64, s: uint64) -> uint64:
    """
    Rotates x left by s

    Args:
        x (int): The input value to rotate
        s (int): The number of bits to rotate by

    Returns:
        int: The rotated value
    """
    return (uint64(x) << uint64(s)) ^ (uint64(x) >> uint64(64 - s))


@cuda.jit(device=True)
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


@cuda.jit(device=True)
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


@cuda.jit(device=True)
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


@cuda.jit(device=True)
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


@cuda.jit(device=True)
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
def keccak256_single(data: bytes, output_ptr: np.ndarray):
    """Computes a single Keccak256 hash on the device

    Args:
        data (bytes): Input data to hash
        output_buf (array): Buffer to write the hash result to
        output_idx (int): Index in output buffer to write the result
    """
    buf_idx = 0
    state = cuda.local.array(25, dtype=uint64)
    buf = cuda.local.array(200, dtype=uint8)

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
def score_func(hash):
    score = np.int32(0)
    for i in range(12, 32):  # hash[12:] in bytes
        if hash[i] != 0:
            break
        score += 1
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

    keccak256_single(data, output)


# can be called only from host code
@cuda.jit
def create2_search_kernel(
    deployer_addr,
    initcode_hash,
    hashes_per_thread,
    global_best_scores,
    global_best_salts,
):
    # Shared memory allocation
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bd = cuda.blockDim.x

    # Maximum threads per block (adjust as needed)
    max_threads_per_block = 256

    smem_best_scores = cuda.shared.array(shape=max_threads_per_block, dtype=np.int32)
    smem_best_salts = cuda.shared.array(shape=(max_threads_per_block, 32), dtype=uint8)

    thread_best_score = np.int32(0)
    thread_best_salt = cuda.local.array(32, dtype=np.uint8)

    thread_id = cuda.grid(1)
    start = thread_id * hashes_per_thread
    end = start + hashes_per_thread

    # Constants
    data_len = 1 + 20 + 32 + 32
    data = cuda.local.array(data_len, dtype=uint8)
    data[0] = 0xFF
    for i in range(20):
        data[1 + i] = deployer_addr[i]
    for i in range(32):
        data[21 + i] = 0
    for i in range(32):
        data[53 + i] = initcode_hash[i]

    hash_output = cuda.local.array(32, dtype=uint8)

    # Each thread processes its assigned salts
    for idx in range(start, end):
        # write idx at the last 4 bytes of salt
        data[49] = idx & 0xFF000000
        data[50] = idx & 0xFF0000
        data[51] = idx & 0xFF00
        data[52] = idx & 0xFF

        # Compute hash
        keccak256_single(data, hash_output)

        # Score the hash
        score = score_func(hash_output)
        if score > thread_best_score and score > 1:
            print("new best score = ", score)
            print("new best salt = ", idx)
            thread_best_score = score
            for i in range(32):
                thread_best_salt[i] = data[21 + i]

    # Write per-thread best score and salt to shared memory
    smem_best_scores[tx] = thread_best_score
    for i in range(32):
        smem_best_salts[tx, i] = thread_best_salt[i]
    cuda.syncthreads()

    # Reduction within block to find the best score and salt
    stride = bd // 2
    while stride > 0:
        if tx < stride:
            if smem_best_scores[tx + stride] > smem_best_scores[tx]:
                smem_best_scores[tx] = smem_best_scores[tx + stride]
                for i in range(32):
                    smem_best_salts[tx, i] = smem_best_salts[tx + stride, i]
        cuda.syncthreads()
        stride //= 2

    # Write block's best score and salt to global memory
    if tx == 0:
        global_best_scores[bx] = smem_best_scores[0]
        for i in range(32):
            global_best_salts[bx, i] = smem_best_salts[0, i]


# host function
def create2_search(
    deployer_addr: bytes,
    initcode_hash: bytes,
    num_hashes: int,
    threads_per_block=256,
    hashes_per_thread=256,
) -> bytes:
    # Calculate total number of threads and blocks
    total_threads = (num_hashes + hashes_per_thread - 1) // hashes_per_thread
    blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block

    # Prepare inputs
    deployer_addr_np = np.frombuffer(deployer_addr, dtype=np.uint8)
    initcode_hash_np = np.frombuffer(initcode_hash, dtype=np.uint8)

    # Copy data to device
    deployer_addr_d = cuda.to_device(deployer_addr_np)
    initcode_hash_d = cuda.to_device(initcode_hash_np)

    # Allocate output arrays
    global_best_scores = cuda.device_array(shape=(blocks_per_grid,), dtype=np.int32)
    global_best_salts = cuda.device_array(shape=(blocks_per_grid, 32), dtype=np.uint8)

    # Launch the kernel
    create2_search_kernel[blocks_per_grid, threads_per_block](
        deployer_addr_d,
        initcode_hash_d,
        hashes_per_thread,
        global_best_scores,
        global_best_salts,
    )

    # Copy back the per-block best scores and salts
    host_best_scores = global_best_scores.copy_to_host()
    host_best_salts = global_best_salts.copy_to_host()

    # Find the overall best
    best_score = -1
    best_salt = None
    for i in range(blocks_per_grid):
        if host_best_scores[i] > best_score:
            best_score = host_best_scores[i]
            best_salt = host_best_salts[i]

    return bytes(best_salt)


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


def print_device_info():
    cuda.detect()

    print("\ndevice info:")

    device = cuda.current_context().device
    print(f"\tCompute Capability: {device.compute_capability}")
    print(f"\tMax Threads per Block: {device.MAX_THREADS_PER_BLOCK}")
    print(f"\tMax Shared Memory per Block: {device.MAX_SHARED_MEMORY_PER_BLOCK} bytes")
    print(f"\tMultiprocessor Count: {device.MULTIPROCESSOR_COUNT}")
    print(f"\tWarp Size: {device.WARP_SIZE}")
    print(f"\tTotal Constant Memory: {device.TOTAL_CONSTANT_MEMORY} bytes")
    print(f"\tMax Registers per Block: {device.MAX_REGISTERS_PER_BLOCK}")
    print(f"\tClock Rate: {device.CLOCK_RATE / 1e3} MHz")
    print(f"\tMemory Clock Rate: {device.MEMORY_CLOCK_RATE / 1e3} MHz")
    print(f"\tL2 Cache Size: {device.L2_CACHE_SIZE} bytes")
    print(f"\tAsynchronous Engines: {device.ASYNC_ENGINE_COUNT}")
    print(f"\tCompute Mode: {device.COMPUTE_MODE}")
    print(f"\tConcurrent Kernels Support: {device.CONCURRENT_KERNELS}")
    print(f"\tDevice Overlap: {device.UNIFIED_ADDRESSING}")
    print(f"\tECC Enabled: {device.ECC_ENABLED}")
    print(f"\tKernel Execution Timeout: {device.KERNEL_EXEC_TIMEOUT}")


def to_device_array(data: bytes) -> np.ndarray:
    return cuda.to_device(np.frombuffer(data, dtype=np.uint8))


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
    import time

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

    mp_count = cuda.current_context().device.MULTIPROCESSOR_COUNT
    threads_per_block = 256
    hashes_per_thread = 256
    num_hashes = threads_per_block * hashes_per_thread * mp_count * 8

    print("measuring throughput... ", end="")
    start_time = time.perf_counter()
    best_salt = create2_search(
        zero_addr, empty_hash, num_hashes, threads_per_block, hashes_per_thread
    )
    end_time = time.perf_counter()
    throughput = num_hashes / (end_time - start_time)
    print(f"{throughput:,.0f} salts/sec")

    print(f"best salt: {best_salt.hex()}")
    print(f"{create2_helper(zero_addr, best_salt, empty_hash).hex()=}")


if __name__ == "__main__":
    main()
