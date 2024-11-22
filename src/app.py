import os
import sys
import time

from numba import cuda, uint32
import numpy as np
from rich import print
from rich.console import Console

from create2_cuda import create2_search
from keccak256_numba import keccak256


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


def get_salt_prefix() -> bytes:
    prefix_str = os.environ.get("SALT_PREFIX", "00" * 20)
    prefix_str = prefix_str[2:] if prefix_str.startswith("0x") else prefix_str
    prefix = bytes.fromhex(prefix_str)
    if len(prefix) > 20:
        raise ValueError("SALT_PREFIX must be 20 bytes or less")

    if len(prefix) < 20:
        # pad right
        padding = 20 - len(prefix)
        print(f"padding SALT_PREFIX={prefix.hex()} with {padding} null bytes")
        prefix = prefix + b"\x00" * padding

    return prefix


def get_deployer_addr() -> bytes:
    deployer_addr_str = os.getenv("DEPLOYER_ADDR", None)
    if not deployer_addr_str:
        raise ValueError("DEPLOYER_ADDR environment variable is not set")

    deployer_addr_str = (
        deployer_addr_str[2:]
        if deployer_addr_str.startswith("0x")
        else deployer_addr_str
    )
    deployer_addr = bytes.fromhex(deployer_addr_str)
    if len(deployer_addr) != 20:
        raise ValueError(
            f"expected DEPLOYER_ADDR to be 20 bytes, got {len(deployer_addr)}"
        )

    return deployer_addr


def get_initcode_hash() -> bytes:
    initcode_hash_str = os.getenv("INITCODE_HASH", None)
    if not initcode_hash_str:
        raise ValueError("INITCODE_HASH environment variable is not set")

    initcode_hash_str = (
        initcode_hash_str[2:]
        if initcode_hash_str.startswith("0x")
        else initcode_hash_str
    )
    initcode_hash = bytes.fromhex(initcode_hash_str)
    if len(initcode_hash) != 32:
        raise ValueError(
            f"expected INITCODE_HASH to be 32 bytes, got {len(initcode_hash)}"
        )

    return initcode_hash


def create2_hash(deployer_addr: bytes, salt: bytes, initcode_hash: bytes) -> bytes:
    """
    Compute a single CREATE2 address on the CPU
    """

    if len(deployer_addr) != 20:
        raise ValueError(
            f"expected deployer_addr to be 20 bytes, got {len(deployer_addr)}"
        )
    if len(initcode_hash) != 32:
        raise ValueError(
            f"expected initcode_hash to be 32 bytes, got {len(initcode_hash)}"
        )
    if len(salt) != 32:
        raise ValueError(f"expected salt to be 32 bytes, got {len(salt)}")

    return keccak256(b"\xff" + deployer_addr + salt + initcode_hash)


def add_to_log(logfile: str, msg: str):
    with open(logfile, "a") as f:
        f.write(msg + "\n")


def clz(x: np.uint32) -> np.int32:
    """Count leading zeros in a 32-bit integer"""
    if x == 0:
        return 32
    # Built-in bit_length() gives position of highest set bit
    return 32 - int(x).bit_length()


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

    # Fast exit if the first 4 bytes are not zero
    word0 = (
        (uint32(hash[12]) << 24)
        | (uint32(hash[13]) << 16)
        | (uint32(hash[14]) << 8)
        | uint32(hash[15])
    )
    clz_word0 = clz(uint32(word0))

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
        clz_word1 = clz(uint32(word1))
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

    # Check for four 4s
    if (shifted & 0xFFFF) != 0x4444:
        # print(f"no four 4s, worth 0 points {hex(overextended)=}, {hex(shifted)=}")
        return 0

    score += 40
    # print(f"four 4s, {score=}, {hex(overextended)=}, {hex(shifted)=}")

    # Check next nibble
    next_nibble = (overextended & 0xFF) >> (shift_amount - 4)
    score += (next_nibble != 0x4) * 20
    # print(f"next_nibble={hex(next_nibble)} {score=}")

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

    return int(score)


def format_elapsed(elapsed: float) -> str:
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    return f"{hours:02d}h{minutes:02d}m{seconds:05.2f}s"


def main():
    mp_count = cuda.current_context().device.MULTIPROCESSOR_COUNT
    threads_per_block = 256
    hashes_per_thread = 2**14
    num_hashes = threads_per_block * hashes_per_thread * mp_count * 8

    best_score = 0
    absolute_start_time = time.perf_counter()

    try:
        deployer_addr = get_deployer_addr()
        initcode_hash = get_initcode_hash()
        salt_prefix = get_salt_prefix()
    except ValueError as e:
        print(f"error: {e}")
        sys.exit(1)

    console = Console()

    logfile = f"log-{time.strftime('%Y%m%d%H%M%S')}.txt"
    with open(logfile, "w") as f:
        f.write("score,addr,salt\n")
    print(f"writing interesting results to {logfile}")

    try:
        with console.status("initializing...") as status:
            while True:
                start_time = time.perf_counter()
                score_approx, salt = create2_search(
                    deployer_addr,
                    initcode_hash,
                    salt_prefix=salt_prefix,
                    num_hashes=num_hashes,
                    threads_per_block=threads_per_block,
                    hashes_per_thread=hashes_per_thread,
                )

                elapsed_batch = time.perf_counter() - start_time
                throughput = num_hashes / elapsed_batch

                elapsed_total = time.perf_counter() - absolute_start_time

                hash = create2_hash(deployer_addr, salt, initcode_hash)
                score_actual = score_func_uniswap_v4_host(hash)

                address = hash[12:]
                msg_base = f"{score_approx=} {score_actual=} addr={address.hex()} salt={salt.hex()} [{format_elapsed(elapsed_total)}]"
                status.update(status=f"{msg_base} [light gray]({throughput:,.0f} hashes/s)")


                if score_actual > 0 and score_actual >= best_score:
                    best_score = score_actual
                    console.print(f"🏆 {msg_base}")
                    add_to_log(logfile, f"{score_actual},{address.hex()},{salt.hex()}")

                # uncomment when profiling with ncu
                # break

    except KeyboardInterrupt:
        print(f"interrupted after {format_elapsed(time.perf_counter() - absolute_start_time)}")


if __name__ == "__main__":
    main()
