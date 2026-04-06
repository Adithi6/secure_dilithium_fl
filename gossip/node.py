# gossip/node.py
#
# GossipNode wraps a FederatedClient and adds:
#   - own_submission : this node's signed update (set after sign_update())
#   - inbox          : all verified updates received via gossip from peers
#   - receive_gossip : called by GossipProtocol when a peer forwards a message
#
# The server collects submissions from node.get_all_submissions() instead of
# directly from clients — it gets everything the gossip network propagated.
#
# Nothing in FederatedClient changes.  GossipNode is a pure wrapper.

from torch.utils.data import DataLoader
from client.fl_client import FederatedClient


class GossipNode:
    """
    A GossipNode = FederatedClient + gossip inbox.

    Attributes:
        client         : the underlying FederatedClient
        own_submission : dict returned by client.sign_update()
        inbox          : list of verified submissions received from peers
    """

    def __init__(self, client_id: str, dataloader: DataLoader, device: str):
        # Delegate all FL work to FederatedClient
        self.client         = FederatedClient(client_id, dataloader, device)
        self.own_submission: dict | None = None
        self.inbox:          list[dict]  = []

        # Convenience pass-throughs so main.py can treat GossipNode like a client
        self.client_id = client_id
        self.pk        = self.client.pk          # public key (for server registration)

    # ------------------------------------------------------------------ #
    def local_train(self, global_weight_arrays: list, epochs: int = 1):
        """Delegate local training to the underlying FederatedClient."""
        self.client.local_train(global_weight_arrays, epochs)

    # ------------------------------------------------------------------ #
    def sign_update(self) -> dict:
        """
        Sign this node's local update and store it as own_submission.
        Also seeds the inbox with own update so get_all_submissions() is complete.
        """
        self.own_submission = self.client.sign_update()
        self.inbox = []   # clear stale inbox from previous round
        return self.own_submission

    # ------------------------------------------------------------------ #
    def receive_gossip(self, message: dict):
        """
        Accept a gossip message from a peer.
        Deduplication is handled by GossipProtocol — we just store it here.
        """
        already_have = any(
            m["msg_hash"] == message["msg_hash"] for m in self.inbox
        )
        if not already_have:
            self.inbox.append(message)

    # ------------------------------------------------------------------ #
    def get_all_submissions(self) -> list[dict]:
        """
        Return own update + everything received via gossip.
        The server calls this to collect all updates for a round.
        """
        all_subs = []
        if self.own_submission:
            all_subs.append(self.own_submission)
        all_subs.extend(self.inbox)
        return all_subs
