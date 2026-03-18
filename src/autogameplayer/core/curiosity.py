import torch


class EpisodicMemory:
    """
    Agent57-style short-term memory (k-NN buffer).
    Rewards the agent for finding novel states *within the current episode*.
    """

    def __init__(
        self, capacity: int = 10000, k: int = 10, cluster_distance: float = 0.008
    ):
        self.capacity = capacity
        self.k = k
        self.cluster_distance = cluster_distance
        self.buffer = []

    def add(self, embedding: torch.Tensor):
        if len(self.buffer) < self.capacity:
            self.buffer.append(embedding.detach().cpu().flatten())

    def reset(self):
        self.buffer.clear()

    def compute_reward(self, embedding: torch.Tensor) -> float:
        if len(self.buffer) < self.k:
            return 1.0  # Max reward when memory is empty/warming up

        emb = embedding.detach().cpu().flatten()
        memories = torch.stack(self.buffer)

        # Calculate L2 distances to all memories
        distances = torch.norm(memories - emb, p=2, dim=1)

        # Get k-nearest neighbors
        topk_dists, _ = torch.topk(distances, self.k, largest=False)

        # Agent57 pseudo-count formula based on distance
        # Count = sum( max(0, dist - cluster_dist) + epsilon )
        # Using a simplified inverse-distance for episodic reward:
        avg_dist = topk_dists.mean().item()

        # Normalize/Scale episodic reward to a reasonable range
        # If avg_dist is 0, we've seen this exact state k times.
        r_e = 1.0 / (avg_dist + 1e-3)  # Simplified count proxy

        # Invert it so high distance = high reward, capped at 1.0
        r_e = min(avg_dist * 10.0, 1.0)

        return float(r_e)


class CuriosityEngine:
    """
    Agent57 Curiosity module.
    Combines Episodic Memory (k-NN) and Lifelong Novelty (RND)
    """

    def __init__(self):
        self.episodic_memory = EpisodicMemory()

    def add_to_episode(self, current_embedding: torch.Tensor):
        self.episodic_memory.add(current_embedding)

    def end_episode(self):
        """Flushes the short-term episodic memory buffer."""
        self.episodic_memory.reset()

    def compute_intrinsic_reward(
        self, current_embedding: torch.Tensor, r_lifelong: float = 1.0
    ) -> float:
        """
        Calculates r_i (Intrinsic Reward) using Agent57's combination formula:
        r_i = r_episodic * min(max(r_lifelong, 1.0), 5.0)
        """
        # 1. Calculate episodic reward
        r_episodic = self.episodic_memory.compute_reward(current_embedding)

        # 2. Combine with lifelong RND reward
        # As per Agent57: Scale the episodic reward by the lifelong reward (clamped)
        # This means an old state (r_l ~ 0) in a new episode still gives r_e * 1.0
        # But a truly novel state (r_l > 1) gives a multiplicative bonus.
        r_lifelong_clamped = min(max(r_lifelong, 1.0), 5.0)

        r_intrinsic = r_episodic * r_lifelong_clamped

        return float(r_intrinsic)
