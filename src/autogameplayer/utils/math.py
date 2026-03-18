import torch
import torch.nn as nn

def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Compresses the magnitudes of both large positive and negative values.
    symlog(x) = sign(x) * ln(|x| + 1)
    """
    return x.sign() * torch.log(x.abs() + 1.0)

def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of symlog.
    symexp(x) = sign(x) * (exp(|x|) - 1)
    """
    return x.sign() * (torch.exp(x.abs()) - 1.0)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Used in DreamerV3 and PaMoRL for stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.scale

def two_hot(x: torch.Tensor, min_val: float = -20.0, max_val: float = 20.0, num_bins: int = 255) -> torch.Tensor:
    """
    Encodes a scalar into a two-hot categorical distribution.
    Useful for predicting rewards and values with high variance.
    """
    # x: [batch] or [batch, 1]
    if x.dim() == 2:
        x = x.squeeze(-1)
        
    # Clip to range
    x = torch.clamp(x, min_val, max_val)
    
    # Map to [0, num_bins - 1]
    pos = (x - min_val) / (max_val - min_val) * (num_bins - 1)
    
    low = pos.long()
    high = torch.clamp(low + 1, max=num_bins - 1)
    
    p_high = pos - low.float()
    p_low = 1.0 - p_high
    
    # Create distribution
    batch_size = x.size(0)
    dist = torch.zeros((batch_size, num_bins), device=x.device)
    dist.scatter_(1, low.unsqueeze(1), p_low.unsqueeze(1))
    dist.scatter_(1, high.unsqueeze(1), p_high.unsqueeze(1))
    
    return dist

def from_two_hot(probs: torch.Tensor, min_val: float = -20.0, max_val: float = 20.0, num_bins: int = 255) -> torch.Tensor:
    """
    Decodes a two-hot categorical distribution back into a scalar.
    Uses expectation over bin values.
    """
    device = probs.device
    bin_values = torch.linspace(min_val, max_val, num_bins, device=device)
    # expectation: sum(p_i * v_i)
    return torch.sum(probs * bin_values, dim=-1, keepdim=True)
