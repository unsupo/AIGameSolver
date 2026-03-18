import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two vision vectors."""
    if v1 is None or v2 is None:
        return 0.0

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates Euclidean distance between two vision vectors."""
    if v1 is None or v2 is None:
        return 0.0
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.linalg.norm(v1 - v2))
