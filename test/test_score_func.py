import numpy as np
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
    word0 = (uint32(hash[12]) << 24) | (uint32(hash[13]) << 16) | \
            (uint32(hash[14]) << 8) | uint32(hash[15])
    if word0 != 0:
        return 0

    # Count leading zero nibbles starting from byte 4
    leading_zero_bits = 32  # First 32 bits are zero

    # Combine bytes 4-7 into a 32-bit word
    word1 = (uint32(hash[16]) << 24) | (uint32(hash[17]) << 16) | \
            (uint32(hash[18]) << 8) | uint32(hash[19])

    # Use CUDA clz to count leading zero bits in word1
    leading_zero_bits += clz(word1)
    leading_zero_nibbles = leading_zero_bits >> 2  # Divide by 4

    score = leading_zero_nibbles * 10
    print(f"{leading_zero_nibbles=} {score=}")

    nibble_idx = 24 + leading_zero_nibbles

    # XXX need to look for the first 4 nibble, it could be after the leading zero nibbles
    for i in range(nibble_idx >> 1, 32):
        byte = hash[i]
        if byte >> 4 == 0x4:
            nibble_idx = i * 2
            break
        if byte & 0xF == 0x4:
            nibble_idx = i * 2 + 1
            break

    # Check for four consecutive 4s starting at nibble_idx
    if nibble_idx & 1 == 0:
        # nibble_idx is even, so we can just check the next 2 bytes
        byte_idx = nibble_idx >> 1
        if hash[byte_idx] == 0x44 and hash[byte_idx + 1] == 0x44:
            score += 40

        # Check if the next nibble is not a 4
        next_nibble = hash[byte_idx + 2] >> 4
        if next_nibble != 0x4:
            score += 20

    else:
        # nibble_idx is odd, so we got so shifting to do
        byte_idx = nibble_idx >> 1
        overextended = (uint32(hash[byte_idx]) << 16) | (uint32(hash[byte_idx + 1]) << 8) | uint32(hash[byte_idx + 2])
        shifted = overextended >> 4
        if shifted == 0x4444:
            score += 40
            print(f"first 4 nibble followed by 3 more 4s, {score=}")

        next_nibble = overextended & 0xF
        if next_nibble != 0x4:
            score += 20
            print(f"first nibble after 4 4s is not 4, {score=}")

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


def test_uniswap_v4_score_func():
    # should not contribute to score
    left_pad = b"\x44" * 12

    for addr_hexstr, expected_score in [
        ("0x0000000044449d1061679743a49F04B817fFde6c", 147),
        ("0x000000004444e44Ba6FA1c49573F9c64E3AcAdb1", 148),
        ("0x000000000444406D3bBA81Cd60aecDd06166f136", 154), # odd leading zero nibbles
        # ("0x00000004444Dc6335C3721F0dc7cF4340d344444", 161), # last 4 nibbles are 4s (but fails our fast exit criteria)
        ("0x00000000004444d3cB22EA006470e100Eb014F2D", 166), # lots of zeros
        ("0x0000000000d34444cB22EA006470e100Eb014F2D", 166), # 0s and 4s not contiguous
    ]:

        hash = np.frombuffer(left_pad + bytes.fromhex(addr_hexstr[2:]), dtype=np.uint8)
        assert score_func_uniswap_v4(hash) == expected_score


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(score_addr(sys.argv[1]))
    else:
        print("usage: test_score_func.py <address>")
        sys.exit(1)
