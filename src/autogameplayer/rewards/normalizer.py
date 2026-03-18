from autogameplayer.core.models import Observation


class RewardNormalizer:
    """
    Centralizes the logic for suppressing negative rewards,
    capping massive curiosity bursts, and handling stagnation penalties.
    """

    def __init__(
        self,
        stagnation_threshold: float = 0.1,
        stagnation_penalty: float = 0.1,
        curiosity_burst_threshold: float = 1.5,
    ):
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_penalty = stagnation_penalty
        self.curiosity_burst_threshold = curiosity_burst_threshold
        self.stagnation_counter = 0

    def reset(self):
        self.stagnation_counter = 0

    def normalize(self, reward: float, current_obs: Observation) -> float:
        """
        Applies escaping stagnation penalties and contextual suppression.
        """
        # 1. Escaping stagnation penalty
        delta = current_obs.state.vision_delta
        if delta < self.stagnation_threshold:
            self.stagnation_counter += 1
            reward -= self.stagnation_penalty * self.stagnation_counter
        else:
            self.stagnation_counter = 0

        # 2. Contextual suppression (Game agnostic checks based on generalized flags)
        is_dialogue = current_obs.state.context.get("is_dialogue", False)
        is_exploration = current_obs.state.context.get("is_exploration_active", True)
        is_engine_locked = current_obs.state.context.get("is_engine_locked", False)
        is_intro = current_obs.state.context.get("is_intro_sequence", False)

        if is_dialogue or not is_exploration or is_engine_locked or is_intro:
            # Suppress massive negative penalties when we're trapped in UI or text
            reward = max(0.0, reward)
            # Add a tiny steady drip of reward if vision is changing (i.e., mashing A through text)
            if delta > 0.05:
                reward += 0.1

        # 3. Curiosity burst cap & detection
        if reward > self.curiosity_burst_threshold:
            current_obs.state.context["curiosity_burst"] = True
            reward = min(reward, self.curiosity_burst_threshold)
        else:
            current_obs.state.context["curiosity_burst"] = False

        return reward
