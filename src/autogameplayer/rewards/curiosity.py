import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from autogameplayer.core.interfaces import RewardFunction
from autogameplayer.core.models import Observation
from autogameplayer.core.registry import Registry
from autogameplayer.core.config import settings
from autogameplayer.vision.encoder import VisionEncoder


class EpisodicMemory:
    """Lightweight episodic memory for k-NN curiosity (Agent57 style)."""

    def __init__(self, k: int = 10, max_capacity: int = 3000):
        self.k = k
        self.memory = []
        self.max_capacity = max_capacity

    def reset(self):
        self.memory = []

    def compute_bonus(self, embedding: np.ndarray) -> float:
        if len(self.memory) < self.k:
            self.memory.append(embedding)
            return 1.0  # Max reward until we have k memories

        # Fast dot-product distance (assuming normalized embeddings)
        memories_matrix = np.array(self.memory)
        distances = np.linalg.norm(memories_matrix - embedding, axis=1)

        # Get k nearest neighbors
        k_nearest = np.sort(distances)[: self.k]

        # Inverse distance kernel (Agent57 style)
        kernel = 0.001 / (k_nearest + 0.001)
        similarity = np.sum(kernel)

        # Add to memory and compute episodic bonus
        if len(self.memory) < self.max_capacity:
            self.memory.append(embedding)

        return 1.0 / np.sqrt(similarity + 1e-6)


class TargetNetwork(nn.Module):
    """A fixed, randomized network that defines the 'feature target' for curiosity."""

    def __init__(self, input_dim=None, output_dim=128):
        super().__init__()
        input_dim = input_dim or VisionEncoder.get_dim(settings.vision_model)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

        # Initialize weights randomly and freeze them
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PredictorNetwork(nn.Module):
    """A trainable network that tries to predict the TargetNetwork's output."""

    def __init__(self, input_dim=None, output_dim=128):
        super().__init__()
        input_dim = input_dim or VisionEncoder.get_dim(settings.vision_model)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@Registry.register_reward("rnd_curiosity")
@Registry.register_reward("curiosity")  # Overriding existing curiosity with RND
class RNDCuriosityReward(RewardFunction):
    """
    Random Network Distillation (RND) Curiosity Reward.
    Game-agnostic: Rewards the agent for visiting states that the
    predictor network hasn't learned to map yet.
    """

    def __init__(
        self,
        input_dim=None,
        output_dim=128,
        lr=1e-4,
        reward_scale=1.0,
        category: str = "intrinsic",
        **kwargs,
    ):
        super().__init__(category=category)
        input_dim = input_dim or VisionEncoder.get_dim(settings.vision_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target = TargetNetwork(input_dim, output_dim).to(self.device)
        self.predictor = PredictorNetwork(input_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.reward_scale = reward_scale
        self.episodic_memory = EpisodicMemory()

        # Optional: persistence for the predictor
        self.save_path = settings.models_dir / "rnd_predictor.pth"
        if self.save_path.exists():
            try:
                self.predictor.load_state_dict(
                    torch.load(self.save_path, map_location=self.device)
                )
            except Exception:
                pass

    async def compute(self, prev_obs: Observation, obs: Observation) -> float:
        """
        Calculates the intrinsic curiosity reward based on prediction error and episodic bonus.
        Agent57 Formula: r_i = episodic_novelty * min(lifelong_rnd, L)
        """
        if not obs.state.vision_vector:
            return 0.0

        # Convert vision vector to tensor
        x = (
            torch.tensor(obs.state.vision_vector, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            target_out = self.target(x)
            pred_out = self.predictor(x)

            # 1. RND Curiosity (Lifelong Error)
            rnd_error = F.mse_loss(pred_out, target_out).item()
            # Lifelong clipping (Agent57 L constant)
            L = 5.0
            lifelong_bonus = min(rnd_error, L)

            # 2. Episodic Bonus (k-NN similarity in latent space)
            embedding = pred_out.cpu().numpy().flatten()
            episodic_bonus = self.episodic_memory.compute_bonus(embedding)

        # Combined Intrinsic Reward (Agent57 Formula)
        alpha = 1.0 # scaling constant
        intrinsic_reward = alpha * episodic_bonus * lifelong_bonus * self.reward_scale

        # Improvement #9: Goal-Conditioned Intrinsic Curiosity (GCIC)
        # Scale reward by relevance to current curriculum goal if provided
        goal_latent = obs.state.context.get("goal_latent")
        if goal_latent is not None:
            # We assume embedding is the current state's latent
            from autogameplayer.utils.vector import cosine_similarity
            relevance = (1.0 + cosine_similarity(embedding, np.array(goal_latent))) / 2.0
            intrinsic_reward *= relevance

        # Store for telemetry/replay
        obs.state.context["rnd_error"] = rnd_error
        obs.state.context["intrinsic_reward"] = intrinsic_reward

        # Online Learning: Update the predictor to 'learn' this state
        self._update_predictor(x)

        return float(intrinsic_reward)

    def reset(self):
        """Resets the episodic curiosity for a new run."""
        self.episodic_memory.reset()

    def _update_predictor(self, x: torch.Tensor):
        """Train the predictor to match the target for the current observation."""
        self.optimizer.zero_grad()
        target_out = self.target(x).detach()  # Target is fixed
        pred_out = self.predictor(x)

        loss = F.mse_loss(pred_out, target_out)
        loss.backward()
        self.optimizer.step()

    def save(self):
        """Persists the predictor network."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.predictor.state_dict(), self.save_path)
