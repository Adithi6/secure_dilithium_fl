# crypto/dilithium_utils.py
# Thin wrappers around CRYSTALS-Dilithium2.
# All three operations are timed and return (result, elapsed_ms).
#
# Dilithium2 specs:
#   Public key : 1312 bytes
#   Secret key : 2528 bytes
#   Signature  : 2420 bytes
#   Security   : NIST Level 2 (equivalent to AES-128)

import time
from dilithium_py.dilithium import Dilithium2


def keygen() -> tuple[bytes, bytes, float]:
    """
    Generate a Dilithium2 keypair.

    Returns:
        pk        : public key  (1312 bytes)
        sk        : secret key  (2528 bytes)
        elapsed_ms: time taken in milliseconds
    """
    t0 = time.perf_counter()
    pk, sk = Dilithium2.keygen()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return pk, sk, elapsed_ms


def sign(sk: bytes, message: bytes) -> tuple[bytes, float]:
    """
    Sign a message with a Dilithium2 secret key.

    Args:
        sk     : secret key from keygen()
        message: the bytes to sign (we pass SHA-256 hash of weights)

    Returns:
        signature : bytes (2420 bytes)
        elapsed_ms: time taken in milliseconds
    """
    t0 = time.perf_counter()
    signature = Dilithium2.sign(sk, message)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return signature, elapsed_ms


def verify(pk: bytes, message: bytes, signature: bytes) -> tuple[bool, float]:
    """
    Verify a Dilithium2 signature.

    Args:
        pk        : public key from keygen()
        message   : the original message (same bytes that were signed)
        signature : the signature to check

    Returns:
        is_valid  : True if signature is genuine, False otherwise
        elapsed_ms: time taken in milliseconds
    """
    t0 = time.perf_counter()
    is_valid = Dilithium2.verify(pk, message, signature)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return is_valid, elapsed_ms
