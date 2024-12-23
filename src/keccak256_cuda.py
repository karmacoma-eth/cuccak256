"""SHA3 implementation in Python in functional style"""

from numba import cuda, uint64, uint8
from numpy import array, uint64 as np_uint64, uint8 as np_uint8

# Fixed rate for Keccak-256
RATE = 136
BIT_LENGTH = 256

# Keccak round constants
_KECCAK_RC = array(
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
    dtype=np_uint64,
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
def _keccak_f(state: array) -> array:
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
    state: array, data: bytes, buf: array, buf_idx: int
) -> tuple[array, array, int]:
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
def _squeeze(
    state: array, buf: array, buf_idx: int, output_buf: array, output_idx: int
):
    """
    Performs the squeeze operation of the sponge construction

    Args:
        state (device array): The state array of the SHA-3 sponge construction
        buf (device array): The buffer to squeeze the output into
        buf_idx (int): Current index in the buffer
        output_buf (device array): The output buffer to write the hash to
        output_idx (int): The index in the output buffer to write to
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

            output_buf[output_idx, local_output_idx] = byte_val

            buf_idx += 1
            local_output_idx += 1

            # If we've processed a full rate's worth of data, permute
            if buf_idx == RATE:
                state, buf, buf_idx = _permute(state, buf, 0)

        tosqueeze -= willsqueeze


@cuda.jit(device=True)
def _pad(state: array, buf: array, buf_idx: int) -> tuple[array, array, int]:
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


@cuda.jit
def keccak256_device(
    data_gpu: bytes, output_gpu: array, num_hashes: int, hashes_per_thread: int
):
    thread_id = cuda.grid(1)

    # Calculate starting index for this thread
    start = thread_id * hashes_per_thread
    end = min(start + hashes_per_thread, num_hashes)

    # Each thread processes hashes from start to end
    for idx in range(start, end):
        buf_idx = 0
        state = cuda.local.array(25, dtype=uint64)
        buf = cuda.local.array(200, dtype=uint8)

        # Initialize state and buffer
        for i in range(25):
            state[i] = 0
        for i in range(200):
            buf[i] = 0

        # Absorb data
        state, buf, buf_idx = _absorb(state, data_gpu, buf, buf_idx)

        # Pad
        state, buf, buf_idx = _pad(state, buf, buf_idx)

        # Squeeze
        _squeeze(state, buf, buf_idx, output_gpu, idx)


def keccak256(
    data: bytes, num_hashes=1, threads_per_block=256, hashes_per_thread=256
) -> bytes:
    # Calculate total number of threads needed
    total_threads = (num_hashes + hashes_per_thread - 1) // hashes_per_thread
    blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block

    # Convert input data to a numpy array
    data = np_uint8(list(data))
    data_gpu = cuda.to_device(data)

    # Allocate output buffer on the device
    output_gpu = cuda.device_array((num_hashes, BIT_LENGTH // 8), dtype=np_uint8)

    # Launch the kernel
    keccak256_device[blocks_per_grid, threads_per_block](
        data_gpu, output_gpu, num_hashes, hashes_per_thread
    )

    # Copy the result back to host memory
    output = output_gpu.copy_to_host()
    return output.tobytes()


if __name__ == "__main__":
    import time

    data_in = b""
    hash = keccak256(data_in)
    print(hash.hex())

    expected_hash = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    print(f"hash has expected value? {hash.hex() == expected_hash}")

    threads_per_block = 256
    hashes_per_thread = 256
    num_hashes = threads_per_block * hashes_per_thread * 256

    start_time = time.perf_counter()
    keccak256(
        b"",
        num_hashes=num_hashes,
        threads_per_block=threads_per_block,
        hashes_per_thread=hashes_per_thread,
    )
    end_time = time.perf_counter()
    throughput = num_hashes / (end_time - start_time)
    print(f"{throughput:,.0f} hashes/sec ({threads_per_block=}, {hashes_per_thread=})")
