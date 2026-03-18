import re
import numpy as np
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, model_validator
from autogameplayer.core.models import Observation


class ConditionConfig(BaseModel):
    """Configuration for a curriculum advancement condition."""
    type: str  # "ram", "reward", "vision_similarity", "ocr"
    params: Dict[str, Any] = {}

    model_config = {"extra": "allow"}


class Checkpoint(BaseModel):
    """A milestone in the game representing progress."""
    name: str
    description: str
    condition: Optional[ConditionConfig] = None
    map_id: Optional[int] = None
    reward_bonus: float = 10.0

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def convert_legacy_format(cls, data: Any) -> Any:
        if isinstance(data, dict) and "condition" not in data:
            # Handle legacy format with address/target_value at top level
            if "address" in data and "target_value" in data:
                # We need to copy because we are removing fields from data
                data = data.copy()
                data["condition"] = {
                    "type": "ram",
                    "params": {
                        "address": data.pop("address"),
                        "target_value": data.pop("target_value")
                    }
                }
        return data


class ConditionEvaluator:
    """Evaluates various types of curriculum conditions."""
    
    @staticmethod
    async def evaluate(condition: ConditionConfig, obs: Observation, client: Any) -> bool:
        ctype = condition.type
        params = condition.params
        
        if ctype == "ram":
            addr = int(params["address"], 16) if isinstance(params["address"], str) else int(params["address"])
            target = params["target_value"]
            comparison = params.get("comparison", "==")
            
            # Read from emulator via client
            val = await client.read_byte(addr)
            
            if comparison == "==":
                return val == target
            if comparison == ">=":
                return val >= target
            if comparison == "<=":
                return val <= target
            if comparison == ">":
                return val > target
            if comparison == "<":
                return val < target
            
        elif ctype == "reward":
            threshold = params["threshold"]
            # This usually checks cumulative reward or last reward
            return obs.state.context.get("last_reward", 0.0) >= threshold
            
        elif ctype == "ocr":
            regex = params["regex"]
            if obs.state.ocr_text:
                return bool(re.search(regex, obs.state.ocr_text, re.IGNORECASE))
                
        elif ctype == "vision_similarity":
            # Compare current vision vector to a target latent
            target_vector = np.array(params["target_vector"])
            from autogameplayer.utils.vector import cosine_similarity
            sim = cosine_similarity(obs.state.vision_vector, target_vector)
            return sim >= params.get("threshold", 0.95)
            
        return False


class Curriculum:
    """Manages and evaluates curriculum progress using pluggable conditions."""

    def __init__(self, checkpoints: List[Checkpoint], client: Any):
        self.checkpoints = checkpoints
        self.client = client
        self.current_idx = 0
        self.max_reached_idx = 0
        self.history = [] # List of (index, timestamp, success)

    def get_active_target(self) -> Optional[Checkpoint]:
        if 0 <= self.current_idx < len(self.checkpoints):
            return self.checkpoints[self.current_idx]
        return None

    async def check_advancement(self, obs: Observation) -> bool:
        """Checks if the current checkpoint condition is met."""
        target = self.get_active_target()
        if not target:
            return False
            
        if await ConditionEvaluator.evaluate(target.condition, obs, self.client):
            print(f"🏆 Curriculum Milestone Achieved: {target.name}!")
            self.current_idx += 1
            self.max_reached_idx = max(self.max_reached_idx, self.current_idx)
            return True
        return False

    def rollback(self):
        """Rolls back to the previous milestone if regression is detected."""
        if self.current_idx > 0:
            print(f"⚠️ Curriculum Regression! Rolling back from stage {self.current_idx} to {self.current_idx - 1}")
            self.current_idx -= 1

    def is_complete(self) -> bool:
        return self.current_idx >= len(self.checkpoints)
