from keccak256_numba import keccak256

lorem_ipsum = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque at vehicula ligula, nec ullamcorper quam. Aliquam interdum euismod porta. Aliquam rhoncus erat ligula, vitae vulputate felis varius non. Donec congue sapien sed lorem lacinia euismod. Suspendisse ut elit felis. In at fermentum turpis. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Suspendisse dignissim risus sapien, eget scelerisque sapien vestibulum et. Proin venenatis erat in hendrerit dictum. Etiam blandit dapibus sodales. Morbi eu bibendum nunc."
)


def test_keccak256_numba():
    assert (
        keccak256(b"").hex()
        == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    )

    assert (
        keccak256(b"hello").hex()
        == "1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8"
    )

    assert (
        keccak256(lorem_ipsum.encode()).hex()
        == "da11398f60a3f2d5d64432c75d4817928afd00a1df37246ffea52f0c1622e894"
    )

