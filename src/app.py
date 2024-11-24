import os
import sys
import time
from multiprocessing import Queue
from threading import Thread

import numpy as np
from numba import cuda, uint32
from rich import print
from rich.console import Console
from rich.live import Live
from rich.table import Table

from keccak256_numba import keccak256
from create2_cuda import BatchResult, CudaParams, SearchParams, create2_search

console = Console()


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


class HostSideResult:
    def __init__(self, batch_result: BatchResult):
        self.batch_result = batch_result

        # generate the hash on the host from the search params and salt in the result
        self.hash = create2_hash(
            batch_result.search_params.deployer_addr,
            batch_result.salt,
            batch_result.search_params.initcode_hash,
        )

        # extract the address from the hash
        self.address = self.hash[12:]

        # compute the actual score on the host
        self.actual_score = score_func_uniswap_v4_host(self.hash)

    @property
    def approx_score(self) -> int:
        return self.batch_result.approx_score

    @property
    def throughput(self) -> float:
        return self.batch_result.throughput

    @property
    def salt(self) -> bytes:
        return self.batch_result.salt

    @property
    def elapsed(self) -> float:
        return self.batch_result.elapsed

    @property
    def device_id(self) -> int:
        return self.batch_result.device_id


class Leaderboard:
    def __init__(self, num_devices: int):
        self.best_score = 0
        self.best_address = b"\x00" * 20
        self.best_salt = b"\x00" * 32

        self.latest_results: list[HostSideResult | None] = [None] * num_devices

        self.absolute_start_time = time.perf_counter()
        self.logfile = f"log-{time.strftime('%Y%m%d%H%M%S')}.txt"
        self._add_to_log("score,address,salt")

    def _add_to_log(self, msg: str):
        with open(self.logfile, "a") as f:
            f.write(msg + "\n")

    def generate_table(self) -> Table:
        table = Table()
        table.add_column("device")
        table.add_column("score")
        table.add_column("address", min_width=42)
        table.add_column("salt", min_width=64)
        table.add_column("throughput", min_width=10)
        table.add_column("elapsed")

        for i, result in enumerate(self.latest_results):
            if result is None:
                table.add_row(
                    f"{i}",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                )
                continue

            throughput = result.throughput
            table.add_row(
                f"{result.device_id}",
                f"{result.actual_score}",
                f"0x{result.address.hex()}",
                f"{result.salt.hex()}",
                f"{throughput / 1e6:,.0f} Mh/s",
                f"{result.elapsed:.2f}s",
            )
        return table

    def update(self, batch_result: BatchResult, live: Live):
        result = HostSideResult(batch_result)
        self.latest_results[result.device_id] = result
        live.update(self.generate_table())

        if result.elapsed < 1:
            # this is bad, because it means we can end up running with the same run id
            # (results in duplicate/wasted work)
            console.print(f"warn: {result.device_id} is returning <1s! (wasting work)")

        actual_score = result.actual_score
        if actual_score >= self.best_score and actual_score > 0:
            self.best_score = actual_score
            self._add_to_log(
                f"{actual_score},0x{result.address.hex()},{result.salt.hex()}"
            )

            elapsed_total = time.perf_counter() - self.absolute_start_time
            msg = f"score={result.approx_score}/{actual_score}"
            msg += f" addr=0x{result.address.hex()}"
            msg += f" salt={result.salt.hex()}"
            msg += f" device={result.device_id}"
            msg += f" [{format_elapsed(elapsed_total)}]"
            console.print(f"  {msg}")


def gpu_worker(
    cuda_params: CudaParams,
    search_params: SearchParams,
    results_queue: Queue,
):
    with cuda.gpus[cuda_params.device_id]:
        while True:
            result = create2_search(cuda_params, search_params)
            results_queue.put(result)


def main():
    # Discover available GPUs
    num_devices = len(cuda.gpus)
    if num_devices == 0:
        console.print("error: no CUDA devices found")
        sys.exit(1)

    print(f"starting {num_devices} GPU workers")

    # Shared queue for results
    results_queue = Queue()

    try:
        search_params = SearchParams(
            deployer_addr=get_deployer_addr(),
            initcode_hash=get_initcode_hash(),
            salt_prefix=get_salt_prefix(),
        )
    except ValueError as e:
        console.print(f"error: {e}")
        sys.exit(1)

    # Launch a thread for each GPU
    threads = []
    for device_id in range(num_devices):
        cuda_params = CudaParams(
            device_id=device_id,
            threads_per_block=256,
            hashes_per_thread=int(os.getenv("HASHES_PER_THREAD", 2**16)),
        )

        t = Thread(
            target=gpu_worker,
            args=(
                cuda_params,
                search_params,
                results_queue,
            ),
            daemon=True,  # don't block exit
        )
        t.start()
        threads.append(t)

    leaderboard = Leaderboard(num_devices)
    console.print(f"writing interesting results to [yellow]{leaderboard.logfile}")

    # Monitor status updates
    try:
        with Live(
            leaderboard.generate_table(), auto_refresh=4, console=console
        ) as live:
            while any(t.is_alive() for t in threads):
                while not results_queue.empty():
                    result: BatchResult = results_queue.get()
                    leaderboard.update(result, live)
    except KeyboardInterrupt:
        console.print("[red]Interrupted by user.[/red]")

    # actually, don't wait for threads to finish
    # for t in threads:
    #     t.join()


if __name__ == "__main__":
    main()
