from autogameplayer.core.interfaces import RewardFunction
from autogameplayer.core.models import Observation
from autogameplayer.utils.vector import euclidean_distance
from autogameplayer.core.registry import Registry

@Registry.register_reward("exploration")
class ExplorationReward(RewardFunction):
    def __init__(self, threshold: float = 0.1, stagnation_penalty: float = 0.1, **kwargs):
        self.threshold = threshold
        self.stagnation_penalty = stagnation_penalty
        self.stagnation_counter = 0

    async def compute(self, prev_obs: Observation, obs: Observation) -> float:
        delta = euclidean_distance(prev_obs.state.vision_vector, obs.state.vision_vector)
        
        if delta < self.threshold:
            self.stagnation_counter += 1
            # ESCALATING PENALTY: -0.1, -0.2, -0.3... etc.
            reward = -(self.stagnation_penalty * self.stagnation_counter)
        else:
            self.stagnation_counter = 0
            reward = 1.0
            
        return reward
