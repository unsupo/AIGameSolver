from autogameplayer.core.interfaces import RewardFunction
from autogameplayer.core.models import Observation
from autogameplayer.core.registry import Registry
from autogameplayer.core.memory import MemoryReader
from autogameplayer.core.config_loader import GameConfig

@Registry.register_reward("pokemon_red")
class PokemonRedReward(RewardFunction):
    """
    Specific reward function for Pokemon Red (GB).
    Uses abstract MemoryReader to reward game progression based on profile.
    """
    def __init__(self, memory_reader: MemoryReader = None, client: MemoryReader = None, config: GameConfig = None, **kwargs):
        self.memory_reader = memory_reader or client
        if self.memory_reader is None:
            raise ValueError("PokemonRedReward requires a memory_reader or client.")
        
        if not config or not config.profile or not config.profile.ram:
             raise ValueError("PokemonRedReward requires a GameConfig with a valid profile/ram layout.")
            
        self.layout = config.profile.ram
        self.last_level = None
        self.last_badges = None
        self.last_party_size = None

    async def compute(self, prev_obs: Observation, obs: Observation) -> float:
        reward = 0.0
        
        try:
            # 1. Level Reward
            if self.layout.level:
                level = await self.memory_reader.read_byte(self.layout.level)
                if self.last_level is None: self.last_level = level
                
                if level > self.last_level and level < 101:
                    reward += (level - self.last_level) * 10.0
                    self.last_level = level

            # 2. Badge Reward
            if self.layout.badges:
                badges_raw = await self.memory_reader.read_byte(self.layout.badges)
                badges_count = bin(badges_raw).count('1')
                if self.last_badges is None: self.last_badges = badges_count
                
                if badges_count > self.last_badges:
                    reward += (badges_count - self.last_badges) * 100.0
                    self.last_badges = badges_count

            # 3. Party Size Reward
            if self.layout.party_size:
                party_size = await self.memory_reader.read_byte(self.layout.party_size)
                if self.last_party_size is None: self.last_party_size = party_size
                
                if party_size > self.last_party_size and party_size <= 6:
                    reward += (party_size - self.last_party_size) * 5.0
                    self.last_party_size = party_size

        except Exception:
            # print(f"Reward error: {e}")
            pass
            
        return reward
