# utils/weights.py
# Converts PyTorch model parameters <-> raw bytes.
# This is needed so we can hash and sign the model update.

import hashlib
import numpy as np
import torch.nn as nn


def weights_to_bytes(model: nn.Module) -> bytes:
    """
    Flatten all model parameters into a single bytes object.
    Order follows model.parameters() — consistent across calls.
    """
    arrays = [p.data.cpu().numpy().flatten() for p in model.parameters()]
    flat   = np.concatenate(arrays).astype(np.float32)
    return flat.tobytes()


def bytes_to_weight_arrays(data: bytes, template_model: nn.Module) -> list[np.ndarray]:
    """
    Reconstruct per-layer numpy arrays from a flat bytes object.
    Uses template_model only to recover the original shapes.
    """
    flat   = np.frombuffer(data, dtype=np.float32).copy()
    shapes = [tuple(p.shape) for p in template_model.parameters()]

    arrays, idx = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        arrays.append(flat[idx : idx + n].reshape(shape))
        idx += n
    return arrays


def apply_weight_arrays(model: nn.Module, arrays: list[np.ndarray]):
    """
    Write numpy arrays back into a model's parameters in-place.
    """
    import torch
    for param, arr in zip(model.parameters(), arrays):
        param.data = torch.tensor(arr)


def hash_weights(model: nn.Module) -> bytes:
    """
    SHA-256 hash of the model's serialised parameters.
    We sign this hash (not the raw bytes) to keep the signed
    payload a fixed 32 bytes regardless of model size.
    """
    return hashlib.sha256(weights_to_bytes(model)).digest()
