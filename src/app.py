import os
import time

from numba import cuda

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
    prefix = bytes.fromhex(os.environ.get("SALT_PREFIX", "00" * 20))
    if len(prefix) > 20:
        raise ValueError("SALT_PREFIX must be 20 bytes or less")

    if len(prefix) < 20:
        # pad right
        print(
            f"padding SALT_PREFIX={prefix.hex()} with {20 - len(prefix)} bytes of \\x00"
        )
        prefix = prefix + b"\x00" * (20 - len(prefix))

    return prefix


def get_deployer_addr() -> bytes:
    deployer_addr_str = os.getenv("DEPLOYER_ADDR", None)
    if not deployer_addr_str:
        raise ValueError("DEPLOYER_ADDR environment variable is not set")

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

    initcode_hash = bytes.fromhex(initcode_hash_str)
    if len(initcode_hash) != 32:
        raise ValueError(
            f"expected INITCODE_HASH to be 32 bytes, got {len(initcode_hash)}"
        )

    return initcode_hash


def create2_addr(deployer_addr: bytes, salt: bytes, initcode_hash: bytes) -> bytes:
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

    return keccak256(b"\xff" + deployer_addr + salt + initcode_hash)[12:]


def main():
    mp_count = cuda.current_context().device.MULTIPROCESSOR_COUNT
    threads_per_block = 256
    hashes_per_thread = 256
    num_hashes = threads_per_block * hashes_per_thread * mp_count * 8

    best_score = 0
    absolute_start_time = time.perf_counter()

    deployer_addr = get_deployer_addr()
    initcode_hash = get_initcode_hash()
    salt_prefix = get_salt_prefix()

    try:
        while True:
            start_time = time.perf_counter()
            score, salt = create2_search(
                deployer_addr,
                initcode_hash,
                salt_prefix=salt_prefix,
                num_hashes=num_hashes,
                threads_per_block=threads_per_block,
                hashes_per_thread=hashes_per_thread,
            )

            elapsed = time.perf_counter() - start_time
            throughput = num_hashes / elapsed
            print(
                f"{num_hashes:,.0f} hashes computed in {elapsed:.3f} seconds ({throughput:,.0f} hashes/sec)"
            )

            address = create2_addr(deployer_addr, salt, initcode_hash)
            print(f"address: 0x{address.hex()}")

            if score > best_score:
                best_score = score
                elapsed = time.perf_counter() - absolute_start_time
                print(f"🏆 {score=} (salt={salt.hex()}) [{elapsed:.2f}s]")

    except KeyboardInterrupt:
        print(f"interrupted after {time.perf_counter() - absolute_start_time:.2f}s")


if __name__ == "__main__":
    main()
