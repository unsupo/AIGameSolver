import sqlite3
from autogameplayer.core.interfaces import RewardFunction
from autogameplayer.core.models import Observation
from autogameplayer.core.registry import Registry
from autogameplayer.core.config import settings

@Registry.register_reward("curiosity")
class CuriosityReward(RewardFunction):
    """
    Discovery-Driven Reward System.
    Rewards the AI for exploring new tiles and penalizes stagnation on old ones.
    """
    def __init__(self, novelty_reward: float = 2.0, visit_penalty: float = -0.5, penalty_threshold: int = 10, **kwargs):
        self.novelty_reward = novelty_reward
        self.visit_penalty = visit_penalty
        self.penalty_threshold = penalty_threshold
        self.db_path = settings.models_dir / "long_term_memory.db"

    async def compute(self, prev_obs: Observation, obs: Observation) -> float:
        ctx = obs.state.context
        map_id = ctx.get('map_id')
        x, y = ctx.get('x'), ctx.get('y')
        
        if map_id is None or x is None or y is None:
            return 0.0
            
        reward = 0.0
        
        try:
            with sqlite3.connect(str(self.db_path), timeout=10) as conn:
                cursor = conn.execute(
                    "SELECT visit_count FROM explored_locations WHERE map_id = ? AND x = ? AND y = ?",
                    (map_id, x, y)
                )
                row = cursor.fetchone()
                
                if not row:
                    # Brand new tile! Huge win.
                    reward = self.novelty_reward
                else:
                    visit_count = row[0]
                    if visit_count > self.penalty_threshold:
                        # We've been here too much. Feel the "repulsion".
                        reward = self.visit_penalty
        except Exception:
            # Fallback if DB is locked or not ready
            pass
            
        return reward
