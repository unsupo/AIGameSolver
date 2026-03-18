import torch
from typing import List
from autogameplayer.core.interfaces import RewardFunction
from autogameplayer.core.models import Observation
from autogameplayer.core.registry import Registry
from autogameplayer.utils.optimal_transport import sinkhorn_knopp, compute_cost_matrix


class ExpertDemonstration:
    """Holds latents, coords, and keyframes from an expert demonstration."""
    def __init__(self, latents: torch.Tensor, coords: torch.Tensor = None, keyframes: List[int] = None):
        self.latents = latents  # [M, D]
        self.coords = coords    # [M, 3] (map_id, x, y)
        self.keyframes = keyframes or []
        self.m = latents.size(0)

@Registry.register_reward("otr")
class OTRReward(RewardFunction):
    """
    Optimal Transport Reward: Aligns agent behavior with an expert demonstration.
    R_OTR = exp(-sinkhorn_distance / temperature)
    Supports Hindsight OTR via SLAM coordinate matching.
    """
    def __init__(
        self, 
        expert_path: str,
        temperature: float = 0.5,
        window_size: int = 50,
        category: str = "intrinsic",
        **kwargs
    ):
        super().__init__(category=category)
        self.temperature = temperature
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        # Load expert demonstration
        self.expert = self._load_expert(expert_path)
        
        # Tracking
        self.agent_latents = [] 
        self.current_window_start = 0

    def _load_expert(self, path: str) -> ExpertDemonstration:
        try:
            data = torch.load(path, map_location="cpu")
            if isinstance(data, dict):
                return ExpertDemonstration(
                    data["latents"].to(self.device), 
                    data.get("coords").to(self.device) if "coords" in data else None,
                    data.get("keyframes")
                )
            return ExpertDemonstration(data.to(self.device))
        except Exception as e:
            print(f"⚠️ Failed to load expert demo from {path}: {e}")
            return ExpertDemonstration(torch.zeros((10, 384), device=self.device))

    def reset(self):
        self.agent_latents = []
        self.current_window_start = 0

    async def compute(self, prev_obs: Observation, obs: Observation) -> float:
        """
        Computes the OTR bonus by comparing the current agent window to the expert demo.
        """
        if obs.state.vision_vector is None:
            return 0.0
            
        # 1. Update agent trajectory window
        v = torch.tensor(obs.state.vision_vector, dtype=torch.float32, device=self.device)
        self.agent_latents.append(v)
        if len(self.agent_latents) > self.window_size:
            self.agent_latents.pop(0)
            
        # 2. Hindsight OTR: Check if we are at a future expert coordinate
        if self.expert.coords is not None and obs.state.context.get("coords"):
            curr_coords = torch.tensor([
                obs.state.context.get("map_id", -1),
                obs.state.context.get("x", -1),
                obs.state.context.get("y", -1)
            ], device=self.device)
            
            # Look ahead in expert demo for this coordinate
            future_expert = self.expert.coords[self.current_window_start:]
            matches = (future_expert == curr_coords).all(dim=1)
            if matches.any():
                future_idx = matches.nonzero()[0].item() + self.current_window_start
                if future_idx > self.current_window_start:
                    print(f"🚀 Hindsight OTR: Shortcut detected! Jumping expert window to {future_idx}")
                    self.current_window_start = future_idx

        # 3. Extract current expert window
        m = self.expert.m
        end_idx = min(self.current_window_start + self.window_size, m)
        expert_window = self.expert.latents[self.current_window_start:end_idx]
        
        if expert_window.size(0) < 5 or len(self.agent_latents) < 5:
            return 0.0

        # 4. Compute Sinkhorn Distance
        agent_window = torch.stack(self.agent_latents)
        C = compute_cost_matrix(agent_window, expert_window)
        dist = sinkhorn_knopp(C, epsilon=0.1)
        
        # 5. Convert to Reward
        reward = float(torch.exp(-dist / self.temperature))
        
        # 6. Automatic Progression
        if dist < 0.1: 
             self.current_window_start = min(self.current_window_start + 1, m - 5)

        return reward
