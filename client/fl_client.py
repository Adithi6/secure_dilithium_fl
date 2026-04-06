# client/fl_client.py
# Each FL client:
#   1. Generates a Dilithium keypair on init
#   2. Receives global weights from the server each round
#   3. Trains locally on its private data
#   4. Signs its updated weights and sends the package to the server

import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.cnn import SmallCNN
from crypto import dilithium_utils
from utils.weights import apply_weight_arrays, weights_to_bytes


class FederatedClient:
    def __init__(self, client_id: str, dataloader: DataLoader, device: str):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.model = SmallCNN().to(device)

        # ── Dilithium keygen ──────────────────────────────────
        self.pk, self.sk, keygen_ms = dilithium_utils.keygen()
        print(f"  [{client_id}] keygen : {keygen_ms:.2f} ms  "
              f"(pk={len(self.pk)}B  sk={len(self.sk)}B)")

    # ------------------------------------------------------------------ #
    def local_train(self, global_weight_arrays: list, epochs: int = 1):
        """
        Load the latest global model weights and fine-tune on local data.

        Args:
            global_weight_arrays : list of numpy arrays from the server
            epochs               : how many passes over local data
        """
        apply_weight_arrays(self.model, global_weight_arrays)

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        print(f"  [{self.client_id}] trained  | loss: {total_loss/len(self.dataloader):.4f}")

    # ------------------------------------------------------------------ #
    def sign_update(self) -> dict:
        update_bytes = weights_to_bytes(self.model)
        msg_hash = hashlib.sha256(update_bytes).digest()

        signature, sign_ms = dilithium_utils.sign(self.sk, msg_hash)

        # One invalid client for Dilithium verification demo
        if self.client_id == "client_2":
            print(f"  [{self.client_id}] MALICIOUS | corrupting signature")
            bad_sig = bytearray(signature)
            bad_sig[0] ^= 255
            signature = bytes(bad_sig)

        print(f"  [{self.client_id}] signed   | {sign_ms:.3f} ms  "
              f"update={len(update_bytes)/1024:.1f} KB  "
              f"sig={len(signature)} B")

        return {
            "client_id": self.client_id,
            "update_bytes": update_bytes,
            "msg_hash": msg_hash,
            "signature": signature,
            "sign_ms": sign_ms,
        }