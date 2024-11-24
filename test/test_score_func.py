import numpy as np
import pytest

from numba import uint32, uint64

import sys

sys.path.append(".")
from src.create2_cuda import load_uint64


def load_uint64_little_endian(buf: np.ndarray, offset: int) -> np.uint64:
    """
    Loads 8 bytes from the buffer starting at the given offset and converts them to a uint64

    Args:
        buf (device array of uint8): The buffer to load from
        offset (int): The offset in the buffer to start loading from

    Returns:
        uint64: The loaded uint64 value
    """

    uint64_val = uint64(0)
    for j in range(8):
        uint64_val |= uint64(buf[offset + j]) << (j * 8)
    return uint64_val


def load_uint64_big_endian(buf: np.ndarray, offset: int) -> np.uint64:
    result = uint64(0)
    for i in range(8):
        result = (result << 8) | buf[offset + i]
    return result


def clz(x: np.uint32 | np.uint64) -> np.int32:
    """Count leading zeros in a 32-bit or 64-bit integer"""

    assert isinstance(x, np.uint32) or isinstance(x, np.uint64)
    size = 32 if isinstance(x, np.uint32) else 64

    if x == 0:
        return size
    # Built-in bit_length() gives position of highest set bit
    return size - int(x).bit_length()


def score_func_uniswap_v4_host(hash: np.ndarray) -> np.int32:
    """
    Approximate score function for Uniswap v4 (optimized for CUDA)

    10 points for every leading 0 nibble
    40 points if the first 4 is followed by 3 more 4s
    20 points if the first nibble after the four 4s is NOT a 4
    20 points if the last 4 nibbles are 4s
    1 point for every 4

    Note: the address starts at hash[12:]
    """

    # Fast exit if the first 2 bytes are not zero
    if hash[12] != 0 or hash[13] != 0:
        return 0

    next8 = load_uint64_big_endian(hash, 14)
    leading_zero_bits = 16 + clz(next8)
    leading_zero_nibbles = leading_zero_bits >> 2  # Divide by 4
    score = leading_zero_nibbles * 10

    # 24 is 2x12 (byte offset of the address in the hash)
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
    next_nibble = (overextended >> (shift_amount - 4)) & 0xF
    score += (next_nibble != 4) * 20
    print(f"next_nibble={hex(next_nibble)} {score=}")

    # add 1 point for every 4 nibble
    num_fours = 0
    for i in range(12, 32):
        byte = hash[i]
        if byte >> 4 == 0x4:
            num_fours += 1
        if byte & 0xF == 0x4:
            num_fours += 1

    # if the last 4 nibbles are 4s
    if hash[30] == 0x44 and hash[31] == 0x44:
        score += 20
        # print(f"last 4 nibbles are 4s, {score=}")

    score += num_fours
    # print(f"num_fours={num_fours} {score=}")

    return score


def score_addr(addr_hexstr: str) -> int:
    left_pad = b"\x44" * 12
    addr_hexstr = addr_hexstr[2:] if addr_hexstr.startswith("0x") else addr_hexstr
    hash = np.frombuffer(left_pad + bytes.fromhex(addr_hexstr), dtype=np.uint8)
    return score_func_uniswap_v4_host(hash)


@pytest.mark.parametrize(
    "addr_hexstr, expected_score, desc",
    [
        ("0x0000000000d34444cB22EA006470e100Eb014F2D", 0, "4x4s in wrong place"),
        ("0x000000000fCb2919EbCC5148761023400D1907DC", 0, "no 4x4s"),
        ("0x00000000a5Ef1c8ae981AfcF3e8B362097CC4444", 0, "4x4s in the wrong place"),
        ("0x0000000044449d1061679743a49F04B817fFde6c", 147, "basic"),
        ("0x000000004444e44Ba6FA1c49573F9c64E3AcAdb1", 148, "basic"),
        ("0x000000000444406D3bBA81Cd60aecDd06166f136", 154, "odd leading zero nibbles"),
        ("0x00000000004444d3cB22EA006470e100Eb014F2D", 166, "lots of zeros"),
        ("0x000000000444441e9ceed846080b0b91992c72f9", 136, "odd nibbles, 5x4s"),
        ("0x000000004444401e9ceed846080b0b91992c72f9", 126, "even nibbles, 5x4s"),
    ],
)
def test_uniswap_v4_score_func(addr_hexstr: str, expected_score: int, desc: str):
    """Test score function with various addresses"""

    print(f"{desc=}")
    assert score_addr(addr_hexstr) == expected_score


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(score_addr(sys.argv[1]))
    else:
        print("usage: test_score_func.py <address>")
        sys.exit(1)
