import random
import time
import hashlib
from gossip.node import GossipNode
from crypto import dilithium_utils


class GossipProtocol:
    """
    Manages one gossip round across all nodes.

    Args:
        fanout      : how many peers each node randomly forwards to
        max_hops    : maximum propagation depth before stopping
        all_pub_keys: dict mapping client_id → Dilithium public key
    """

    def __init__(
        self,
        fanout: int = 2,
        max_hops: int = 3,
        all_pub_keys: dict[str, bytes] = None,
    ):
        self.fanout = fanout
        self.max_hops = max_hops
        self.all_pub_keys = all_pub_keys or {}

        self._seen: set[bytes] = set()
        self.gossip_timings: list[dict] = []

    def reset_round(self):
        """Clear seen-set at the start of each FL round."""
        self._seen.clear()
        self.gossip_timings.clear()

    def _verify_before_forward(
        self, sender_id: str, message: dict
    ) -> tuple[bool, float]:
        """
        Verify integrity + Dilithium signature of a received gossip message.
        """
        pk = self.all_pub_keys.get(message["client_id"])
        if pk is None:
            return False, 0.0

        # Step 1: recompute hash from received bytes
        recomputed_hash = hashlib.sha256(message["update_bytes"]).digest()

        # Step 2: check integrity
        if recomputed_hash != message["msg_hash"]:
            return False, 0.0

        # Step 3: verify signature
        is_valid, verify_ms = dilithium_utils.verify(
            pk, message["msg_hash"], message["signature"]
        )
        return is_valid, verify_ms

    def spread(
        self,
        origin_node: "GossipNode",
        all_nodes: list["GossipNode"],
        message: dict,
        hop: int = 0,
    ):
        """
        Recursively spread `message` from `origin_node` to `fanout` random peers.
        """
        msg_id = message["msg_hash"]

        if msg_id in self._seen or hop >= self.max_hops:
            return
        self._seen.add(msg_id)

        peers = [n for n in all_nodes if n.client_id != origin_node.client_id]
        targets = random.sample(peers, min(self.fanout, len(peers)))

        for target in targets:
            is_valid, verify_ms = self._verify_before_forward(origin_node.client_id, message)

            self.gossip_timings.append({
                "from": origin_node.client_id,
                "to": target.client_id,
                "hop": hop + 1,
                "verify_ms": round(verify_ms, 3),
                "accepted": is_valid,
            })

            print(
                f"  [gossip] {origin_node.client_id} → {target.client_id}"
                f"  hop={hop+1}  verify={verify_ms:.3f} ms"
                f"  [{'OK' if is_valid else 'REJECTED'}]"
            )

            if is_valid:
                target.receive_gossip(message)
                self.spread(target, all_nodes, message, hop=hop + 1)

    def run_round(self, nodes: list["GossipNode"]):
        """
        Each node gossips its own signed update to the network.
        """
        self.reset_round()

        for node in nodes:
            if node.own_submission is None:
                raise RuntimeError(
                    f"{node.client_id} has no submission — call sign_update() first"
                )
            print(f"\n  [gossip] spreading update from {node.client_id} ...")
            self.spread(
                origin_node=node,
                all_nodes=nodes,
                message=node.own_submission,
            )

    def print_gossip_summary(self):
        if not self.gossip_timings:
            return

        print(f"\n  {'─'*54}")
        print(f"  Gossip log  (fanout={self.fanout}  max_hops={self.max_hops})")
        print(f"  {'─'*54}")
        print(f"  {'From':<12} {'To':<12} {'Hop':<5} {'Verify (ms)':<14} Accepted")
        print(f"  {'─'*54}")

        for t in self.gossip_timings:
            print(
                f"  {t['from']:<12} {t['to']:<12} {t['hop']:<5}"
                f" {t['verify_ms']:<14} {t['accepted']}"
            )

        accepted = [t for t in self.gossip_timings if t["accepted"]]
        if accepted:
            avg_v = sum(t["verify_ms"] for t in accepted) / len(accepted)
            print(f"\n  Total gossip hops : {len(self.gossip_timings)}")
            print(f"  Avg gossip verify : {avg_v:.3f} ms")