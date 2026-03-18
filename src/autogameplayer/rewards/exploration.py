from autogameplayer.core.interfaces import RewardFunction
from autogameplayer.core.models import Observation
from autogameplayer.utils.vector import euclidean_distance
from autogameplayer.core.registry import Registry


@Registry.register_reward("exploration")
class ExplorationReward(RewardFunction):
    def __init__(self, threshold: float = 0.1, category: str = "intrinsic", **kwargs):
        super().__init__(category=category)
        self.threshold = threshold

    def reset(self) -> None:
        pass

    async def compute(self, prev_obs: Observation, obs: Observation) -> float:
        delta = euclidean_distance(
            prev_obs.state.vision_vector, obs.state.vision_vector
        )
        if delta >= self.threshold:
            return 1.0
        return 0.0
