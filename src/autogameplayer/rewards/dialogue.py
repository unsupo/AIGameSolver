import hashlib
from autogameplayer.core.interfaces import RewardFunction
from autogameplayer.core.models import Observation
from autogameplayer.core.registry import Registry

@Registry.register_reward("dialogue_discovery")
class DialogueDiscoveryReward(RewardFunction):
    """
    Incentivizes the agent to discover and read new dialogue lines.
    Bridges the gap between raw exploration and semantic learning.
    """
    def __init__(self, reward_value: float = 5.0, **kwargs):
        self.seen_text_hashes = set()
        self.reward_value = reward_value

    async def compute(self, prev_obs: Observation, obs: Observation) -> float:
        ocr_text = (obs.state.ocr_text or "").strip()
        is_dialogue = obs.state.context.get("is_dialogue", False)
        
        if not ocr_text or not is_dialogue or len(ocr_text) < 4:
            return 0.0
            
        text_hash = hashlib.md5(ocr_text.encode()).hexdigest()
        if text_hash not in self.seen_text_hashes:
            self.seen_text_hashes.add(text_hash)
            print(f"📖 New Dialogue Discovered: '{ocr_text[:30]}...' (+{self.reward_value})")
            return self.reward_value
            
        return 0.0
