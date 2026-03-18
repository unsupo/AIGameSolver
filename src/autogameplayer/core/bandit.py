import numpy as np
import math
import asyncio
import sqlite3
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from autogameplayer.utils.database import get_db_connection


@dataclass
class PolicyArm:
    id: int
    gamma: float  # Discount factor
    beta: float   # Intrinsic reward weight
    epsilon: float # Exploration rate
    frame_skip: int = 1


class SlidingWindowUCB:
    """
    Sliding-window UCB for Agent57 arm selection.
    Tracks performance of different (gamma, beta) arms over a window of episodes.
    """

    def __init__(
        self,
        num_arms: int = 32,
        window_size: int = 100,
        exploration_constant: float = 0.5,
        db_path: Optional[Path] = None,
    ):
        self.num_arms = num_arms
        self.window_size = window_size
        self.xi = exploration_constant
        self.db_path = db_path
        self.lock = asyncio.Lock()

        # Generate arms following Agent57 pattern
        self.arms = self._generate_arms(num_arms)
        
        # Stats tracked per arm
        self.arm_history: List[Tuple[int, float]] = [] # list of (arm_id, reward)
        self.arm_pulls = np.zeros(num_arms)
        self.total_pulls = 0

    def _generate_arms(self, num_arms: int) -> List[PolicyArm]:
        arms = []
        for i in range(num_arms):
            # Beta (intrinsic weight) - scaled from 0 to 0.3
            beta = 0.3 * (i / (num_arms - 1)) if num_arms > 1 else 0.1
            
            # Gamma (discount) - scaled from 0.9 to 0.999
            # Usually higher beta arms use higher gammas
            gamma = 0.9 + 0.099 * (i / (num_arms - 1)) if num_arms > 1 else 0.99
            
            # Epsilon (exploration)
            # Arm 0 might be the most exploitative, last arm most explorative
            epsilon = 0.01 + 0.1 * (i / (num_arms - 1))
            
            arms.append(PolicyArm(id=i, gamma=gamma, beta=beta, epsilon=epsilon))
        return arms

    async def select_arm(self) -> PolicyArm:
        """Selects the best arm using UCB formula."""
        async with self.lock:
            # Warm-start: Ensure every arm is pulled at least once
            for i in range(self.num_arms):
                if self.arm_pulls[i] == 0:
                    self.arm_pulls[i] += 1
                    self.total_pulls += 1
                    return self.arms[i]

            # Calculate UCB scores
            scores = np.zeros(self.num_arms)
            t = min(len(self.arm_history), self.window_size)
            
            # Get windowed averages
            window = self.arm_history[-self.window_size:]
            arm_window_rewards = {i: [] for i in range(self.num_arms)}
            for aid, r in window:
                arm_window_rewards[aid].append(r)
            
            for i in range(self.num_arms):
                n_i = len(arm_window_rewards[i])
                if n_i == 0:
                    # Fallback for arms not in current window
                    mu_i = 0.0
                    n_i = 1 # avoid div by zero
                else:
                    mu_i = np.mean(arm_window_rewards[i])
                
                # UCB = mean + xi * sqrt(ln(window_size) / n_i)
                ucb = mu_i + self.xi * math.sqrt(math.log(t) / n_i)
                scores[i] = ucb

            best_arm_id = int(np.argmax(scores))
            self.arm_pulls[best_arm_id] += 1
            self.total_pulls += 1
            return self.arms[best_arm_id]

    async def report_episode_result(self, arm_id: int, reward: float):
        """Updates statistics with the result of an episode."""
        async with self.lock:
            self.arm_history.append((arm_id, reward))
            if len(self.arm_history) > self.window_size * 2:
                self.arm_history = self.arm_history[-self.window_size:]
            
            if self.db_path:
                await self._persist_stats()

    async def _persist_stats(self):
        """Saves bandit state to SQLite."""
        if not self.db_path:
            return
            
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with get_db_connection(self.db_path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS bandit_stats (arm_id INTEGER, reward REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
                # Only keep recent history in DB too
                conn.execute("INSERT INTO bandit_stats (arm_id, reward) VALUES (?, ?)", self.arm_history[-1])
        except Exception as e:
            print(f"⚠️ Failed to persist bandit stats: {e}")

    @classmethod
    async def load_from_db(cls, db_path: Path, num_arms: int = 32, window_size: int = 100) -> 'SlidingWindowUCB':
        instance = cls(num_arms=num_arms, window_size=window_size, db_path=db_path)
        try:
            with get_db_connection(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT arm_id, reward FROM bandit_stats ORDER BY timestamp DESC LIMIT ?", (window_size,))
                rows = cursor.fetchall()
                for row in reversed(rows):
                    instance.arm_history.append((row['arm_id'], row['reward']))
                    instance.arm_pulls[row['arm_id']] += 1
                    instance.total_pulls += 1
        except Exception:
            pass
        return instance
