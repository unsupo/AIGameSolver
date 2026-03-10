from typing import List
from autogameplayer.core.interfaces import RewardFunction
from autogameplayer.core.models import Observation
from autogameplayer.core.registry import Registry

@Registry.register_reward("ocr")
class OcrKeywordReward(RewardFunction):
    def __init__(self, keywords: List[str] = None, reward_value: float = 10.0, **kwargs):
        self.keywords = keywords or ["WIN", "LEVEL UP", "BADGE", "GOT", "OBTAINED"]
        self.reward_value = reward_value

    async def compute(self, prev_obs: Observation, obs: Observation) -> float:
        if not obs.state.ocr_text:
            return 0.0
            
        reward = 0.0
        upper_text = obs.state.ocr_text.upper()
        for keyword in self.keywords:
            if keyword in upper_text:
                reward += self.reward_value
        return reward
