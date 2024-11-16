# cuccak256

## Development

```sh
$ uv --version
uv 0.5.1

# initialized with
$ uv init

# add dependencies
$ uv add ruff numba

# generate lockfile
$ uv sync

# run ruff
$ uv run ruff check .

# run tests
$ uv run pytest
```

## Single Hash Latency Benchmarks

Baseline:

```sh
$ ETH_HASH_BACKEND=pycryptodome python -m timeit "from eth_hash.auto import keccak; keccak(b'')"
50000 loops, best of 5: 5.69 usec per loop

$ ETH_HASH_BACKEND=pysha3 python -m timeit "from eth_hash.auto import keccak; keccak(b'')"
200000 loops, best of 5: 1.02 usec per loop
```

```sh
$ ./bench.sh

## numpy
200 loops, best of 5: 1.42 msec per loop

## numba, excluding compile time
100000 loops, best of 5: 1.99 usec per loop
```

## Hash Throughput Benchmarks


## Acknowledgements

Based on https://github.com/dupontcyborg/sha3-numba
