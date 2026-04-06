# Dilithium + Federated Learning + Gossip Protocol

## Project Structure

```
dilithium_fl/
│
├── main.py                      ← Entry point. Runs the full FL loop
│
├── model/
│   └── cnn.py                   ← SmallCNN architecture (PyTorch)
│
├── data/
│   └── loader.py                ← Downloads MNIST, splits per client
│
├── crypto/
│   └── dilithium_utils.py       ← Keygen, sign, verify (Dilithium2)
│
├── gossip/
│   ├── node.py                  ← GossipNode wraps FederatedClient + inbox
│   └── protocol.py              ← Peer-to-peer gossip spreading logic
│
├── client/
│   └── fl_client.py             ← Local training + signing (unchanged)
│
├── server/
│   └── fl_server.py             ← Verify signatures + FedAvg aggregation
│
└── utils/
    └── weights.py               ← Convert model weights <-> bytes
```

## How it flows

```
main.py
  │
  ├─ data/loader.py              → split MNIST among N nodes
  ├─ crypto/dilithium_utils.py   → each node generates Dilithium keypair
  │
  └─ for each round:
       ├─ gossip/node.py         → local SGD training on node's data
       ├─ gossip/node.py         → sign(SHA256(weights)) with Dilithium sk
       ├─ gossip/protocol.py     → spread signed updates peer-to-peer
       │     each receiver verifies Dilithium sig before forwarding
       │     stops at max_hops or when all peers have the message
       └─ server/fl_server.py    → collect inboxes, verify again, FedAvg
```

## Gossip protocol

Without gossip: every client sends directly to the server (star topology).

With gossip: each node forwards its signed update to `fanout` random peers.
Peers verify the Dilithium signature before forwarding further. The server
collects from node inboxes — it does not talk to every client directly.

```
  node_0 ---> node_1, node_2        (hop 1)
  node_1 ---> node_3, node_0        (hop 2, node_0 already seen -> skip)
  node_2 ---> node_3                (hop 2, node_3 already has it -> skip)
  all node inboxes are full
  server <--- collect from all inboxes (dedup by msg_hash)
```

Tune in main.py:
  GOSSIP_FANOUT   — peers each node forwards to (default 2)
  GOSSIP_MAX_HOPS — max propagation depth (default 3)

## Install

```bash
pip install torch torchvision dilithium-py numpy
```

##  Hash vs Direct Signing Comparison

This project implements and compares two approaches for signing model updates using Dilithium:

### 1. Hash-then-Sign (SHA-256 + Dilithium)
- Model weights are first hashed using SHA-256 (32 bytes)
- The hash is signed using Dilithium
- Verification checks both:
  - Hash integrity
  - Signature validity

### 2. Direct Signing (Without Hash)
- Full model update (≈800 KB) is directly signed
- Signature verification is performed on raw data

---

##  Observations

| Feature              | With Hash        | Without Hash      |
|---------------------|-----------------|------------------|
| Input size to sign  | 32 bytes        | ~800 KB          |
| Speed               | Faster          | Slower           |
| Security            | Strong integrity| Signature only   |
| Scalability         | High            | Low              |

---

##  Key Insight

Hashing reduces large model updates into a fixed-size representation, making signing efficient while preserving integrity.

This is especially important in Federated Learning where model updates are large.

## Run

```bash
python main.py
```
