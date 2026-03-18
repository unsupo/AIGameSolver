import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from autogameplayer.core.config import settings
from autogameplayer.vision.encoder import VisionEncoder


class RNDNetwork(nn.Module):
    def __init__(
        self, input_dim: int = None, hidden_dim: int = 256, output_dim: int = 128
    ):
        super().__init__()
        if input_dim is None:
            input_dim = VisionEncoder.get_dim(settings.vision_model)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)


class RunningMeanStd:
    """Tracks running mean and standard deviation for input normalization."""

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RandomNetworkDistillation:
    """
    RND for lifelong novelty detection (Never Give Up).
    """

    def __init__(
        self,
        input_dim: int = None,
        lr: float = 1e-4,
        device: str = "cpu",
        burn_in_steps: int = 100,
    ):
        self.device = torch.device(device)
        
        if input_dim is None:
            input_dim = VisionEncoder.get_dim(settings.vision_model)
            
        self.target = RNDNetwork(input_dim=input_dim).to(self.device)
        self.predictor = RNDNetwork(input_dim=input_dim).to(self.device)

        self.input_dim = input_dim
        self.obs_rms = RunningMeanStd(shape=(input_dim,))
        self.burn_in_steps = burn_in_steps
        self.steps_seen = 0

        # Target network is frozen
        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def compute_intrinsic_reward(
        self, observation_vector: torch.Tensor, train: bool = True
    ) -> float:
        """
        Returns the RND error as the lifelong novelty reward.
        If train is True, also updates the predictor network.
        """
        obs = observation_vector.to(self.device).detach()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # 1. Update running statistics and normalize input
        obs_np = obs.cpu().numpy()
        if train:
            self.obs_rms.update(obs_np)
            self.steps_seen += obs_np.shape[0]

        obs_mean = torch.tensor(
            self.obs_rms.mean, dtype=torch.float32, device=self.device
        )
        obs_var = torch.tensor(
            self.obs_rms.var, dtype=torch.float32, device=self.device
        )
        obs_normalized = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
        obs_normalized = torch.clamp(obs_normalized, -5.0, 5.0)

        # 2. Compute Target & Prediction
        with torch.no_grad():
            target_feat = self.target(obs_normalized)

        if train:
            self.optimizer.zero_grad()
            pred_feat = self.predictor(obs_normalized)
            # Predictor isolation / loss scaling
            loss = F.mse_loss(pred_feat, target_feat, reduction="none").mean(dim=-1)
            mean_loss = loss.mean()
            mean_loss.backward()
            self.optimizer.step()

            # 3. RND Burn-in: Return 0 reward until predictor has burned in
            if self.steps_seen < self.burn_in_steps:
                return 0.0

            return loss[0].item()
        else:
            if self.steps_seen < self.burn_in_steps:
                return 0.0

            with torch.no_grad():
                pred_feat = self.predictor(obs_normalized)
                loss = F.mse_loss(pred_feat, target_feat, reduction="none").mean(dim=-1)
                return loss[0].item()
