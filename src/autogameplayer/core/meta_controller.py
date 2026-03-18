import asyncio
from collections import deque
import numpy as np


class MetaController:
    """
    Sliding-Window UCB-based Bandit for selecting Agent57 personalities.
    Each personality corresponds to a specific (beta, gamma) schedule.
    """

    def __init__(
        self,
        num_personalities: int = 32,
        window_size: int = 1000,
        max_db_size_gb: float = 1.0,
    ):
        self.num_personalities = num_personalities
        self.window_size = window_size
        self.max_db_size_gb = max_db_size_gb

        # Warm start: 1 count per personality
        self.counts = np.ones(num_personalities)
        self.scores = np.zeros(num_personalities)

        # Sliding window elements: tuple of (personality_id, reward)
        self.window = deque(maxlen=window_size)

        self.lock = asyncio.Lock()

        # Define the beta (intrinsic reward weight) and gamma (discount factor) for each personality.
        # Following Agent57: some have high beta/low gamma (Explorers), some low beta/high gamma (Speedrunners).
        self.betas = np.linspace(0.0, 0.5, num_personalities)
        self.gammas = np.linspace(0.99, 0.999, num_personalities)

    async def select_personality(self) -> int:
        async with self.lock:
            total_counts = np.sum(self.counts)
            if total_counts == 0:
                return int(np.random.randint(self.num_personalities))

            # Standard UCB formula to balance 'Trying the best' vs 'Exploring new settings'
            ucb_values = self.scores + np.sqrt(
                2 * np.log(total_counts + 1) / (self.counts + 1e-5)
            )
            return int(np.argmax(ucb_values))

    async def update(self, personality_id: int, reward: float):
        async with self.lock:
            if len(self.window) == self.window_size:
                old_id, old_reward = self.window[0]
                self.counts[old_id] -= 1
                # Recompute score for old_id (approximate by moving average un-roll or exact recompute below)

            self.window.append((personality_id, reward))

            # Exact recompute of counts and scores from window (plus warm start)
            self.counts = np.ones(self.num_personalities)
            sum_rewards = np.zeros(self.num_personalities)
            for p_id, r in self.window:
                self.counts[p_id] += 1
                sum_rewards[p_id] += r

            # Average score (0 if warm start only)
            pulls = self.counts - 1
            self.scores = np.divide(
                sum_rewards, pulls, out=np.zeros_like(self.scores), where=pulls != 0
            )

    async def save_state(self, db_conn):
        """Persist bandit state to SQLite."""
        # SQLite connection handles commit/serialization.
        async with self.lock:
            cursor = db_conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS bandit_stats (window_json TEXT)")

            import json

            window_data = list(self.window)
            cursor.execute("DELETE FROM bandit_stats")
            cursor.execute(
                "INSERT INTO bandit_stats (window_json) VALUES (?)",
                (json.dumps(window_data),),
            )
            db_conn.commit()

    async def load_state(self, db_conn):
        """Restore bandit state from SQLite."""
        async with self.lock:
            cursor = db_conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS bandit_stats (window_json TEXT)")
            cursor.execute("SELECT window_json FROM bandit_stats LIMIT 1")
            row = cursor.fetchone()
            if row:
                import json

                window_data = json.loads(row[0])
                self.window = deque(window_data, maxlen=self.window_size)

                # Recompute exact state
                self.counts = np.ones(self.num_personalities)
                sum_rewards = np.zeros(self.num_personalities)
                for p_id, r in self.window:
                    self.counts[p_id] += 1
                    sum_rewards[p_id] += r

                pulls = self.counts - 1
                self.scores = np.divide(
                    sum_rewards, pulls, out=np.zeros_like(self.scores), where=pulls != 0
                )

    def get_beta(self, personality_id: int) -> float:
        return self.betas[personality_id]

    def get_gamma(self, personality_id: int) -> float:
        return self.gammas[personality_id]
