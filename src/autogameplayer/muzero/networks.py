import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Any
from autogameplayer.core.config import settings
from autogameplayer.vision.encoder import VisionEncoder
from autogameplayer.utils.math import symlog, symexp, RMSNorm, from_two_hot


class BlockGRUCell(nn.Module):
    """
    Splits the hidden state into parallel blocks for better hardware utilization.
    Contributor to DreamerV3 and PaMoRL stability.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.block_dim = hidden_dim // num_blocks
        
        # We use a single linear layer and split manually for speed
        self.w_ih = nn.Linear(input_dim, 3 * hidden_dim)
        self.w_hh = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_dim], h: [batch, hidden_dim]
        gates_i = self.w_ih(x)
        gates_h = self.w_hh(h)
        
        i_r, i_z, i_n = gates_i.chunk(3, dim=1)
        h_r, h_z, h_n = gates_h.chunk(3, dim=1)
        
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        
        next_h = (1 - update_gate) * new_gate + update_gate * h
        return self.norm(next_h)


class RepresentationNetwork(nn.Module):
    """f: observation (vision vector) + conditioning (beta, gamma) -> hidden state"""

    def __init__(self, input_dim=None, hidden_dim=256, num_personalities=32):
        super().__init__()
        input_dim = input_dim or VisionEncoder.get_dim(settings.vision_model)
        self.fc1 = nn.Linear(input_dim + num_personalities + 2, hidden_dim)
        self.norm1 = RMSNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        # Improvement #2: Symlog inputs
        x = symlog(x)
        
        x = torch.cat([x, conditioning], dim=-1)
        x = F.silu(self.norm1(self.fc1(x))) # SiLU activation (DreamerV3)
        x = self.norm2(self.fc2(x))
        return torch.tanh(x)


class DynamicsNetwork(nn.Module):
    """g: hidden state + action -> next hidden state + reward + auxiliary signals"""

    def __init__(self, hidden_dim=256, action_dim=8, map_id_dim=256, ocr_hash_dim=256, num_bins=255):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        
        # Improvement #3: Block GRU + RMSNorm
        self.gru = BlockGRUCell(action_dim, hidden_dim)

        # Improvement #1: Two-hot reward head
        self.reward_out = nn.Linear(hidden_dim, num_bins)
        
        # Auxiliary signals
        self.map_id_out = nn.Linear(hidden_dim, map_id_dim)
        self.ocr_hash_out = nn.Linear(hidden_dim, ocr_hash_dim)

    def forward(
        self, h: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # a: action one-hot
        if a.dim() == 3:
            a = a.squeeze(1) if a.size(1) == 1 else a[:, 0, :]

        if h.dim() == 3:
            h = h.squeeze(0) if h.size(0) == 1 else (h.squeeze(1) if h.size(1) == 1 else h[:, 0, :])

        next_h = self.gru(a, h)

        # Predict auxiliary signals (Categorical/Two-hot)
        reward_logits = self.reward_out(next_h)
        map_id_logits = self.map_id_out(next_h)
        ocr_hash_logits = self.ocr_hash_out(next_h)

        return next_h, reward_logits, map_id_logits, ocr_hash_logits


class PredictionNetwork(nn.Module):
    """phi: hidden state -> policy logits + value (two-hot)"""

    def __init__(self, hidden_dim=256, action_dim=8, num_personalities=32, num_bins=255):
        super().__init__()
        self.num_bins = num_bins
        self.fc1 = nn.Linear(hidden_dim + num_personalities + 2, hidden_dim)
        self.norm1 = RMSNorm(hidden_dim)
        self.policy_out = nn.Linear(hidden_dim, action_dim)
        
        # Improvement #1: Two-hot value heads
        self.value_ext = nn.Linear(hidden_dim, num_bins)
        self.value_int = nn.Linear(hidden_dim, num_bins)

    def forward(
        self, s: torch.Tensor, conditioning: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([s, conditioning], dim=-1)
        x = F.silu(self.norm1(self.fc1(x)))
        policy_logits = self.policy_out(x)

        v_ext_logits = self.value_ext(x)
        v_int_logits = self.value_int(x)
        return policy_logits, v_ext_logits, v_int_logits


class MuZeroModel(nn.Module):
    """Upgraded DreamerV3-style Recurrent Model."""

    def __init__(
        self,
        input_dim=None,
        hidden_dim=256,
        action_dim=8,
        map_id_dim=256,
        ocr_hash_dim=256,
        num_personalities=32,
        num_bins=255,
    ):
        super().__init__()
        input_dim = input_dim or VisionEncoder.get_dim(settings.vision_model)
        self.num_personalities = num_personalities
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        
        self.representation = RepresentationNetwork(
            input_dim, hidden_dim, num_personalities
        )
        self.dynamics = DynamicsNetwork(
            hidden_dim, action_dim, map_id_dim, ocr_hash_dim, num_bins
        )
        self.prediction = PredictionNetwork(hidden_dim, action_dim, num_personalities, num_bins)
        self.action_dim = action_dim
        self.map_id_dim = map_id_dim

    def _get_conditioning(self, batch_size, device, personality_id, beta=0.1, gamma=0.99):
        if isinstance(personality_id, int):
            p_ids = torch.full((batch_size,), personality_id, dtype=torch.long, device=device)
        else:
            p_ids = personality_id.to(device)
            
        p_onehot = F.one_hot(p_ids, num_classes=self.num_personalities).float()
        betas = torch.full((batch_size, 1), float(beta), device=device) if isinstance(beta, (float, int)) else beta.view(batch_size, 1).to(device)
        gammas = torch.full((batch_size, 1), float(gamma), device=device) if isinstance(gamma, (float, int)) else gamma.view(batch_size, 1).to(device)
        return torch.cat([p_onehot, symlog(betas), symlog(gammas)], dim=-1)

    def initial_inference(
        self, observation: torch.Tensor, personality_id: int = 0, beta=0.1, gamma=0.99
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = observation.device
        batch_size = observation.size(0) if observation.dim() > 1 else 1
        conditioning = self._get_conditioning(batch_size, device, personality_id, beta, gamma)
        
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            
        h = self.representation(observation, conditioning)
        policy_logits, v_ext_l, v_int_l = self.prediction(h, conditioning)
        
        # Decode two-hot to scalars for inference
        v_ext = symexp(from_two_hot(F.softmax(v_ext_l, dim=-1)))
        v_int = symexp(from_two_hot(F.softmax(v_int_l, dim=-1)))
        
        if batch_size == 1 and observation.dim() == 1:
             return h.squeeze(0), policy_logits.squeeze(0), v_ext.squeeze(0), v_int.squeeze(0)
             
        return h, policy_logits, v_ext, v_int

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action_idx: Any, 
        personality_id: int = 0, beta=0.1, gamma=0.99
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = hidden_state.device
        
        if isinstance(action_idx, int):
            action_onehot = F.one_hot(torch.tensor([action_idx], device=device), num_classes=self.action_dim).float()
        elif isinstance(action_idx, torch.Tensor):
            action_onehot = F.one_hot(action_idx, num_classes=self.action_dim).float() if action_idx.dtype == torch.long else action_idx
        else:
            action_onehot = action_idx
            
        next_h, reward_l, map_id_logits, ocr_hash_logits = self.dynamics(
            hidden_state, action_onehot
        )

        batch_size = next_h.size(0)
        conditioning = self._get_conditioning(batch_size, device, personality_id, beta, gamma)

        policy_logits, v_ext_l, v_int_l = self.prediction(next_h, conditioning)
        
        # Decode
        reward = symexp(from_two_hot(F.softmax(reward_l, dim=-1)))
        v_ext = symexp(from_two_hot(F.softmax(v_ext_l, dim=-1)))
        v_int = symexp(from_two_hot(F.softmax(v_int_l, dim=-1)))
        
        return (
            next_h,
            reward,
            map_id_logits,
            ocr_hash_logits,
            policy_logits,
            v_ext,
            v_int,
        )

    def load_genome(self, genome: np.ndarray):
        state_dict = self.state_dict()
        start = 0
        for name, param in state_dict.items():
            size = param.numel()
            param_data = genome[start : start + size].reshape(param.shape)
            state_dict[name].copy_(torch.from_numpy(param_data))
            start += size

    def get_genome_size(self) -> int:
        return sum(p.numel() for p in self.parameters())
