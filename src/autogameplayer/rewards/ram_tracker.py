from autogameplayer.core.interfaces import RewardFunction
from autogameplayer.core.models import Observation
from autogameplayer.core.registry import Registry
from autogameplayer.core.memory import MemoryReader
from autogameplayer.core.config_loader import GameConfig


@Registry.register_reward("ram_tracker")
class RAMTrackerReward(RewardFunction):
    """
    Generic RAM tracking reward function.
    Reads addresses and multipliers dynamically from the `variables` config.
    """

    def __init__(
        self,
        memory_reader: MemoryReader = None,
        client: MemoryReader = None,
        config: GameConfig = None,
        variables: dict = None,
        category: str = "extrinsic",
        **kwargs,
    ):
        super().__init__(category=category)
        self.memory_reader = memory_reader or client
        if self.memory_reader is None:
            raise ValueError("RAMTrackerReward requires a memory_reader or client.")

        self.layout_dict = {}
        if config and config.profile and config.profile.ram:
            self.layout_dict = config.profile.ram.model_dump()

        self.variables = variables or {}
        self.last_values = {}

    def reset(self) -> None:
        self.last_values.clear()

    async def compute(self, prev_obs: Observation, obs: Observation) -> float:
        if not self.layout_dict or not self.variables:
            return 0.0

        reward = 0.0

        try:
            for var_name, var_config in self.variables.items():
                address = self.layout_dict.get(var_name)
                if address is None:
                    continue

                raw_val = await self.memory_reader.read_byte(address)

                var_type = var_config.get("type", "value")
                if var_type == "bitcount":
                    val = bin(raw_val).count("1")
                else:
                    val = raw_val

                if var_name not in self.last_values:
                    self.last_values[var_name] = val
                    continue

                last_val = self.last_values[var_name]
                if val > last_val:
                    max_val = var_config.get("max_value")
                    if max_val is None or val <= max_val:
                        reward += (val - last_val) * var_config.get("multiplier", 1.0)
                        self.last_values[var_name] = val
        except Exception:
            pass

        return reward
