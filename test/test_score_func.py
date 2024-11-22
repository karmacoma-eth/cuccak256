import numpy as np
import pytest
from numba import uint32


def clz(x: np.uint32) -> np.int32:
    """Count leading zeros in a 32-bit integer"""
    if x == 0:
        return 32
    # Built-in bit_length() gives position of highest set bit
    return 32 - int(x).bit_length()


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
    if word0 != 0:
        return 0

    # Count leading zero nibbles starting from byte 4
    leading_zero_bits = 32  # First 32 bits are zero

    # Combine bytes 4-7 into a 32-bit word
    word1 = (
        (uint32(hash[16]) << 24)
        | (uint32(hash[17]) << 16)
        | (uint32(hash[18]) << 8)
        | uint32(hash[19])
    )

    # Use CUDA clz to count leading zero bits in word1
    leading_zero_bits += clz(uint32(word1))
    leading_zero_nibbles = leading_zero_bits >> 2  # Divide by 4

    score = leading_zero_nibbles * 10
    print(f"{leading_zero_nibbles=} {score=}")

    nibble_idx = 24 + leading_zero_nibbles

    for i in range(nibble_idx >> 1, 32):
        byte = hash[i]
        if byte >> 4 == 0x4:
            nibble_idx = i * 2
            break
        if byte & 0xF == 0x4:
            nibble_idx = i * 2 + 1
            break

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

    # Check for four 4s
    four_fours = (shifted & 0xFFFF) == 0x4444
    score += four_fours * 40
    if four_fours:
        print(f"four 4s, {score=}, {hex(overextended)=}, {hex(shifted)=}")
    else:
        print(f"no four 4s, {hex(overextended)=}, {hex(shifted)=}")

    # Check next nibble
    next_nibble = (overextended & 0xFF) >> (shift_amount - 4)
    score += four_fours * (next_nibble != 0x4) * 20
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
        print(f"last 4 nibbles are 4s, {score=}")

    score += num_fours
    print(f"num_fours={num_fours} {score=}")

    return score


def score_addr(addr_hexstr: str) -> int:
    left_pad = b"\x44" * 12
    addr_hexstr = addr_hexstr[2:] if addr_hexstr.startswith("0x") else addr_hexstr
    hash = np.frombuffer(left_pad + bytes.fromhex(addr_hexstr), dtype=np.uint8)
    return score_func_uniswap_v4(hash)


@pytest.mark.parametrize(
    "addr_hexstr, expected_score, desc",
    [
        ("0x0000000044449d1061679743a49F04B817fFde6c", 147, "basic"),
        ("0x000000004444e44Ba6FA1c49573F9c64E3AcAdb1", 148, "basic"),
        ("0x000000000444406D3bBA81Cd60aecDd06166f136", 154, "odd_leading_zero_nibbles"),
        ("0x00000000004444d3cB22EA006470e100Eb014F2D", 166, "lots_of_zeros"),
        ("0x0000000000d34444cB22EA006470e100Eb014F2D", 166, "zeros_and_fours_not_contiguous"),
        ("0x000000000fCb2919EbCC5148761023400D1907DC", 92, "no four 4s"),
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
