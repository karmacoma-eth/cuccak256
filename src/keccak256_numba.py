"""SHA3 implementation in Python in functional style"""

from numba import njit
from numpy import array, uint64, uint8, empty, frombuffer, zeros

# Fixed rate for Keccak-256
RATE = 136
BIT_LENGTH = 256

# Keccak round constants
_KECCAK_RC = array([
  0x0000000000000001, 0x0000000000008082,
  0x800000000000808a, 0x8000000080008000,
  0x000000000000808b, 0x0000000080000001,
  0x8000000080008081, 0x8000000000008009,
  0x000000000000008a, 0x0000000000000088,
  0x0000000080008009, 0x000000008000000a,
  0x000000008000808b, 0x800000000000008b,
  0x8000000000008089, 0x8000000000008003,
  0x8000000000008002, 0x8000000000000080,
  0x000000000000800a, 0x800000008000000a,
  0x8000000080008081, 0x8000000000008080,
  0x0000000080000001, 0x8000000080008008],
  dtype=uint64)

# Domain separation byte
_DSBYTE = 0x01

# Number of Keccak rounds
_NUM_ROUNDS = 24

@njit
def _rol(x, s):
    """
    Rotates x left by s

    Args:
        x (int): The input value to rotate
        s (int): The number of bits to rotate by

    Returns:
        int: The rotated value
    """
    return ((uint64(x) << uint64(s)) ^ (uint64(x) >> uint64(64 - s)))

@njit
def _keccak_f(state):
    """
    The keccak_f permutation function, unrolled for performance

    Args:
        state (ndarray): The state array of the SHA-3 sponge construction

    Returns:
        ndarray: The permuted state array
    """
    # Temporary array for calculations
    bc = zeros(5, dtype=uint64)

    # 24 rounds of permutation
    for i in range(_NUM_ROUNDS):

        # Parity calculation unrolled
        bc[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20]
        bc[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21]
        bc[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22]
        bc[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23]
        bc[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24]

        # Theta unrolled
        t0 = bc[4] ^ _rol(bc[1], 1)
        t1 = bc[0] ^ _rol(bc[2], 1)
        t2 = bc[1] ^ _rol(bc[3], 1)
        t3 = bc[2] ^ _rol(bc[4], 1)
        t4 = bc[3] ^ _rol(bc[0], 1)

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
        t1  = _rol(state[1], 1)
        t2  = _rol(state[10], 3)
        t3  = _rol(state[7], 6)
        t4  = _rol(state[11], 10)
        t5  = _rol(state[17], 15)
        t6  = _rol(state[18], 21)
        t7  = _rol(state[3], 28)
        t8  = _rol(state[5], 36)
        t9  = _rol(state[16], 45)
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
        state[7]  = t2
        state[11] = t3
        state[17] = t4
        state[18] = t5
        state[3]  = t6
        state[5]  = t7
        state[16] = t8
        state[8]  = t9
        state[21] = t10
        state[24] = t11
        state[4]  = t12
        state[15] = t13
        state[23] = t14
        state[19] = t15
        state[13] = t16
        state[12] = t17
        state[2]  = t18
        state[20] = t19
        state[14] = t20
        state[22] = t21
        state[9]  = t22
        state[6]  = t23
        state[1]  = t24

        # Chi unrolled
        t0  = state[0] ^ ((~state[1]) & state[2])
        t1  = state[1] ^ ((~state[2]) & state[3])
        t2  = state[2] ^ ((~state[3]) & state[4])
        t3  = state[3] ^ ((~state[4]) & state[0])
        t4  = state[4] ^ ((~state[0]) & state[1])
        t5  = state[5] ^ ((~state[6]) & state[7])
        t6  = state[6] ^ ((~state[7]) & state[8])
        t7  = state[7] ^ ((~state[8]) & state[9])
        t8  = state[8] ^ ((~state[9]) & state[5])
        t9  = state[9] ^ ((~state[5]) & state[6])
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

        state[0]  = t0
        state[1]  = t1
        state[2]  = t2
        state[3]  = t3
        state[4]  = t4
        state[5]  = t5
        state[6]  = t6
        state[7]  = t7
        state[8]  = t8
        state[9]  = t9
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

@njit
def _absorb(state, data, buf, buf_idx):
    """
    Absorbs input data into the sponge construction.

    Args:
        state (ndarray): The state array of the SHA-3 sponge construction
        data (bytes): The input data to be absorbed
        buf (ndarray): The buffer to absorb the input into
        buf_idx (int): Current index in the buffer

    Returns:
        ndarray: The updated state array
        ndarray: The updated buffer
        int: The updated buffer index
    """
    todo = len(data)
    i = 0
    while todo > 0:
        cando = RATE - buf_idx
        willabsorb = min(cando, todo)
        buf[buf_idx:buf_idx + willabsorb] ^= \
            frombuffer(data[i:i+willabsorb], dtype=uint8)
        buf_idx += willabsorb
        if buf_idx == RATE:
            state, buf, buf_idx = _permute(state, buf, buf_idx)
        todo -= willabsorb
        i += willabsorb

    return state, buf, buf_idx

@njit
def _squeeze(state, buf, buf_idx):
    """
    Performs the squeeze operation of the sponge construction

    Args:
        state (ndarray): The state array of the SHA-3 sponge construction
        buf (ndarray): The buffer to squeeze the output into
        buf_idx (int): Current index in the buffer

    Returns:
        ndarray: The hash output
    """
    tosqueeze = BIT_LENGTH // 8
    output_bytes = empty(tosqueeze, dtype=uint8)  # Temporary storage for output bytes
    output_index = 0  # Tracks where to insert bytes into output_bytes

    while tosqueeze > 0:
        cansqueeze = RATE - buf_idx
        willsqueeze = min(cansqueeze, tosqueeze)

        # Extract bytes from state
        for _ in range(willsqueeze):
            byte_index = buf_idx % 8
            byte_val = (state[buf_idx // 8] >> (byte_index * 8)) & 0xFF
            output_bytes[output_index] = byte_val
            buf_idx += 1
            output_index += 1

            # If we've processed a full rate's worth of data, permute
            if buf_idx == RATE:
                state, buf, buf_idx = _permute(state, buf, 0)  # Reset buf_idx for simplicity

        tosqueeze -= willsqueeze

    return output_bytes

@njit
def _pad(state, buf, buf_idx):
    """
    Pads the input data in the buffer.

    Args:
        state (ndarray): The state array of the SHA-3 sponge construction
        buf (ndarray): The buffer to pad the input into
        buf_idx (int): Current index in the buffer

    Returns:
        ndarray: The updated state array
        ndarray: The updated buffer
        int: The updated buffer index
    """
    buf[buf_idx] ^= _DSBYTE
    buf[RATE - 1] ^= 0x80
    return _permute(state, buf, buf_idx)

@njit
def _permute(state, buf, buf_idx):
    """
    Permutes the internal state and buffer for thorough mixing.

    Args:
        state (ndarray): The state array of the SHA-3 sponge construction
        buf (ndarray): The buffer to permute
        buf_idx (int): Current index in the buffer

    Returns:
        ndarray: The updated state array
        ndarray: The updated buffer
        int: The updated buffer index
    """
    temp_state = zeros(len(state), dtype=uint64)

    # Process bytes to uint64
    for i in range(0, len(buf), 8):
        if i + 8 <= len(buf):  # Ensure there's enough data to read
            uint64_val = uint64(0)
            for j in range(8):
                uint64_val |= uint64(buf[i+j]) << (j * 8)
            temp_state[i//8] = uint64_val

    state ^= temp_state

    # Perform Keccak permutation
    state = _keccak_f(state)

    # Reset buf_idx and buf
    buf_idx = 0
    buf[:] = 0

    return state, buf, buf_idx

@njit
def _keccak256(data: bytes) -> array:
    buf_idx = 0
    state = zeros(25, dtype=uint64)
    buf = zeros(200, dtype=uint8)

    # Absorb data
    state, buf, buf_idx = _absorb(state, data, buf, buf_idx)

    # Pad in preparation for squeezing
    state, buf, buf_idx = _pad(state, buf, buf_idx)

    # Squeeze the hash
    return _squeeze(state, buf, buf_idx)


def keccak256(data: bytes) -> bytes:
    """
    Compute the SHA-3 hash of the input data.

    Args:
        data (bytes): The input data to hash.

    Returns:
        bytes: The hash of the input data.
    """
    return _keccak256(data).tobytes()

if __name__ == "__main__":
    import sys
    print(keccak256(sys.argv[1].encode()).hex())
