from typing import List, Optional
from pydantic import BaseModel

class Checkpoint(BaseModel):
    """A milestone in the game representing progress (e.g., getting a badge)."""
    name: str
    address: int
    target_value: int
    description: str
    map_id: Optional[int] = None
    reward_bonus: float = 10.0

class Curriculum:
    """Helper to manage and evaluate curriculum progress."""
    def __init__(self, checkpoints: List[Checkpoint]):
        self.checkpoints = checkpoints
        self.current_idx = 0

    def get_active_target(self) -> Optional[Checkpoint]:
        if 0 <= self.current_idx < len(self.checkpoints):
            return self.checkpoints[self.current_idx]
        return None

    def advance(self):
        self.current_idx += 1

    def is_complete(self) -> bool:
        return self.current_idx >= len(self.checkpoints)
