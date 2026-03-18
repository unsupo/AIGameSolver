import torch
import torch.nn.functional as F

def sinkhorn_knopp(C: torch.Tensor, epsilon: float = 0.1, n_iter: int = 100) -> torch.Tensor:
    """
    Computes the Sinkhorn distance between two distributions with cost matrix C.
    C: [N, M] cost matrix (e.g. Euclidean distances between latents)
    epsilon: regularization strength
    n_iter: number of iterations
    """
    device = C.device
    n, m = C.shape
    
    # Target distributions (uniform)
    mu = torch.full((n,), 1.0 / n, device=device)
    nu = torch.full((m,), 1.0 / m, device=device)
    
    # Kernel K
    K = torch.exp(-C / epsilon)
    
    u = torch.ones(n, device=device) / n
    v = torch.ones(m, device=device) / m
    
    for _ in range(n_iter):
        u = mu / (K @ v)
        v = nu / (K.T @ u)
        
    # Transport matrix P
    P = u.view(-1, 1) * K * v.view(1, -1)
    
    # Distance
    distance = torch.sum(P * C)
    return distance

def compute_cost_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    X: [N, D] agent latents
    Y: [M, D] expert latents
    Returns: [N, M] cosine distance matrix
    """
    # Normalize
    X_n = F.normalize(X, p=2, dim=1)
    Y_n = F.normalize(Y, p=2, dim=1)
    
    # Cosine Similarity
    sim = X_n @ Y_n.T
    
    # Cosine Distance [0, 2]
    return 1.0 - sim
