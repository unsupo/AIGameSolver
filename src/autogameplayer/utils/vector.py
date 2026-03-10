import math
from typing import List

def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Calculates Euclidean distance between two vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 1.0 # Default max delta
    
    distance = sum((c - p) ** 2 for c, p in zip(v1, v2))
    return math.sqrt(distance)
