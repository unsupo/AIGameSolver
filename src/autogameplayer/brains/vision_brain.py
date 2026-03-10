import random
import collections
import sqlite3
import time
from autogameplayer.core.interfaces import Brain, Controller
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.registry import Registry
from autogameplayer.core.config import settings

@Registry.register_brain("vision")
class VisionBrain(Brain):
    """A brain that uses vision hashing to detect if it's stuck, persisting long-term 'dead ends'."""
    def __init__(self, controller: Controller, **kwargs):
        self.controller = controller
        self.max_history = 20 # Keep track of last 20 frames
        self.history = collections.deque(maxlen=self.max_history)
        self.hash_counts = collections.Counter()
        self.db_path = settings.models_dir / "long_term_memory.db"

    def _update_persistent_stuck(self, vision_hash: str):
        """Increments the global stuck-count for this screen in the DB."""
        try:
            with sqlite3.connect(str(self.db_path), timeout=5) as conn:
                conn.execute("""
                    INSERT INTO stuck_hashes (vision_hash, count, last_seen) 
                    VALUES (?, 1, ?) 
                    ON CONFLICT(vision_hash) DO UPDATE SET count = count + 1, last_seen = ?
                """, (vision_hash, time.time(), time.time()))
                conn.commit()
        except Exception: pass

    def _get_persistent_stuck_count(self, vision_hash: str) -> int:
        """Returns how many times we have been stuck on this screen historically."""
        try:
            with sqlite3.connect(str(self.db_path), timeout=5) as conn:
                cursor = conn.execute("SELECT count FROM stuck_hashes WHERE vision_hash = ?", (vision_hash,))
                row = cursor.fetchone()
                return row[0] if row else 0
        except Exception: return 0

    async def act(self, observation: Observation, mcp_client=None) -> Action:

        current_hash = observation.state_hash
        
        # 1. Local/Short-term Stuck detection (Current session)
        if len(self.history) == self.max_history:
            oldest = self.history[0]
            self.hash_counts[oldest] -= 1
            if self.hash_counts[oldest] == 0:
                del self.hash_counts[oldest]
                
        self.history.append(current_hash)
        self.hash_counts[current_hash] += 1
            
        local_count = self.hash_counts[current_hash]
        
        # 2. Global/Long-term Stuck detection (Across sessions)
        # We only increment global count if we are locally stuck for more than 5 frames
        if local_count == 6:
            self._update_persistent_stuck(current_hash)
            
        global_stuck_count = self._get_persistent_stuck_count(current_hash)
        
        # If we've seen this screen too much locally OR globally
        # (Global threshold is higher since it accumulates over time)
        if local_count > 5 or global_stuck_count > 100:
            if global_stuck_count > 100:
                print(f"⚠️ VisionBrain: Recognizing a persistent DEAD END (Global Stuck Count: {global_stuck_count}).")
            
            # We are stuck. Mash non-directional buttons to get out of menus,
            # or try a completely random button.
            escape_buttons = [b for b in ["a", "b", "start"] if b in self.controller.buttons]
            if escape_buttons and random.random() < 0.5:
                button = random.choice(escape_buttons)
                return Action(button=button, duration=10)
            else:
                return self.fallback_action(self.controller)
        else:
            # Not stuck, just explore. Let's bias towards walking.
            walk_buttons = [b for b in ["up", "down", "left", "right"] if b in self.controller.buttons]
            if walk_buttons and random.random() < 0.8:
                button = random.choice(walk_buttons)
            else:
                button = random.choice(self.controller.buttons)
            return Action(button=button, duration=5)
