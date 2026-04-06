# main.py
#
# Full flow per round:
#
#   1. Server broadcasts global weights to all nodes
#   2. Each node trains locally  
#   3. Each node signs its update  
#   4. GossipProtocol spreads every signed update across peers
#        - Each receiving peer verifies the Dilithium sig before forwarding
#        - Stops at max_hops or when all peers have seen the message
#   5. Server collects each node's full inbox, verifies again, runs FedAvg
#   6. Server evaluates global model accuracy

import numpy as np
import torch

from data.loader import make_client_loaders
from gossip.node import GossipNode
from gossip.protocol import GossipProtocol
from server.fl_server import FederatedServer


# ── Config ──────────────────────────────────────────────────
N_CLIENTS          = 4
N_ROUNDS           = 3
LOCAL_EPOCHS       = 1
SAMPLES_PER_CLIENT = 500
GOSSIP_FANOUT      = 2   # each node forwards to this many random peers
GOSSIP_MAX_HOPS    = 3   # propagation stops after this many hops
# ────────────────────────────────────────────────────────────


def print_timing_table(all_timings: list[dict]):
    print(f"\n{'='*64}")
    print("  Crypto Timing Summary (sign + server-verify)")
    print(f"{'='*64}")
    print(f"  {'Round':<6} {'Client':<12} {'Sign (ms)':<14} {'Verify (ms)':<14} Valid")
    print(f"  {'-'*58}")
    for t in all_timings:
        print(
            f"  {t['round']:<6} {t['client_id']:<12}"
            f" {t['sign_ms']:<14} {t['verify_ms']:<14} {t['valid']}"
        )
    sign_ms   = [t["sign_ms"]   for t in all_timings]
    verify_ms = [t["verify_ms"] for t in all_timings]
    print(f"\n  Avg sign   : {np.mean(sign_ms):.3f} ms"
          f"  (min {np.min(sign_ms):.3f}  max {np.max(sign_ms):.3f})")
    print(f"  Avg verify : {np.mean(verify_ms):.3f} ms"
          f"  (min {np.min(verify_ms):.3f}  max {np.max(verify_ms):.3f})")
    print(f"{'='*64}\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")
    print(f"Config : {N_CLIENTS} clients | {N_ROUNDS} rounds | "
          f"{LOCAL_EPOCHS} local epoch(s) | "
          f"gossip fanout={GOSSIP_FANOUT} max_hops={GOSSIP_MAX_HOPS}\n")

    # ── Data ──────────────────────────────────────────────────
    client_loaders, test_loader = make_client_loaders(
        n_clients=N_CLIENTS,
        samples_per_client=SAMPLES_PER_CLIENT,
    )

    # ── Server ────────────────────────────────────────────────
    server = FederatedServer(device)

    # ── Nodes (GossipNode wraps FederatedClient) ─────────────
    print("── Key generation ──────────────────────────────────")
    nodes: list[GossipNode] = []
    for i in range(N_CLIENTS):
        node = GossipNode(f"client_{i}", client_loaders[i], device)
        server.register_client(node.client_id, node.pk)
        nodes.append(node)

    # ── Gossip protocol (shared across rounds) ────────────────
    all_pub_keys = {n.client_id: n.pk for n in nodes}
    gossip = GossipProtocol(
        fanout=GOSSIP_FANOUT,
        max_hops=GOSSIP_MAX_HOPS,
        all_pub_keys=all_pub_keys,
    )

    # ── Federated rounds ──────────────────────────────────────
    for round_num in range(1, N_ROUNDS + 1):
        print(f"\n── Round {round_num}/{N_ROUNDS} ──────────────────────────────────")

        global_weights = server.global_weight_arrays()

        # Step 1: local training
        print("\n  [training]")
        for node in nodes:
            node.local_train(global_weights, epochs=LOCAL_EPOCHS)

        # Step 2: each node signs its own update
        print("\n  [signing]")
        for node in nodes:
            node.sign_update()

        # Step 3: gossip — spread signed updates peer-to-peer
        #   Each peer verifies the Dilithium signature before forwarding.
        #   After this, every node's inbox holds all verified updates.
        print("\n  [gossip propagation]")
        gossip.run_round(nodes)
        gossip.print_gossip_summary()

        # Step 4: server collects from node inboxes, verifies again, FedAvg
        #   De-duplicate by msg_hash so the server sees each update exactly once.
        print("\n  [server verification & aggregation]")
        seen_hashes: set[bytes] = set()
        all_submissions: list[dict] = []
        for node in nodes:
            for sub in node.get_all_submissions():
                if sub["msg_hash"] not in seen_hashes:
                    seen_hashes.add(sub["msg_hash"])
                    all_submissions.append(sub)

        server.verify_and_aggregate(all_submissions, round_num=round_num)

        # Step 5: evaluate
        accuracy = server.evaluate(test_loader)
        print(f"\n  Global model accuracy : {accuracy:.2f}%")

    # ── Final timing summary ──────────────────────────────────
    print_timing_table(server.all_timings)


if __name__ == "__main__":
    main()
