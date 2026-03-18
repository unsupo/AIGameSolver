import json
import sqlite3
from typing import Dict, Any
from autogameplayer.core.curriculum import Curriculum
from autogameplayer.core.models import Observation
from autogameplayer.utils.database import get_db_connection


class ReplayProcessor:
    """
    Handles post-episode processing of experience, including 
    Hindsight Experience Replay (HER) relabeling.
    """

    def __init__(self, db_path: str, curriculum: Curriculum):
        self.db_path = db_path
        self.curriculum = curriculum

    async def process_episode_her(self, session_id: str):
        """
        Scans an episode for reached curriculum goals and creates 
        synthetic 'success' transitions for those goals.
        """
        print(f"🔄 HER: Relabeling episode {session_id}...")
        
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # 1. Fetch all steps for this session
                cursor = conn.execute(
                    "SELECT * FROM replay_buffer WHERE session_id = ? ORDER BY step_index ASC",
                    (session_id,)
                )
                steps = [dict(row) for row in cursor]
                
                if not steps:
                    return

                # 2. Check for future goals reached in this trajectory
                # We check every checkpoint in the curriculum
                for cp_idx, checkpoint in enumerate(self.curriculum.checkpoints):
                    # Find the FIRST step that satisfies this checkpoint
                    success_step_idx = -1
                    for i, step in enumerate(steps):
                        # Mock an observation from step data for the evaluator
                        # This is a bit heavy, in a real impl we'd store minimal 'state'
                        obs = self._step_to_obs(step)
                        
                        from autogameplayer.core.curriculum import ConditionEvaluator
                        if await ConditionEvaluator.evaluate(checkpoint.condition, obs, self.curriculum.client):
                            success_step_idx = i
                            break
                    
                    if success_step_idx != -1:
                        print(f"✨ HER: Found shortcut to '{checkpoint.name}' at step {success_step_idx}. Relabeling...")
                        # 3. Create synthetic successful transitions
                        # For HER, we often just need to mark the reward as high for the goal
                        # and potentially duplicate the sequence with the new goal_id.
                        
                        # In this simplified implementation, we'll boost the priority 
                        # of the steps leading to the discovered goal.
                        conn.execute(
                            "UPDATE replay_buffer SET priority = priority + 2.0 WHERE session_id = ? AND step_index <= ?",
                            (session_id, success_step_idx)
                        )
                        conn.commit()

        except Exception as e:
            print(f"⚠️ HER Relabeling failed: {e}")

    def _step_to_obs(self, step: Dict[str, Any]) -> Observation:
        # Minimal conversion for ConditionEvaluator
        from autogameplayer.core.models import GameState, Observation
        import numpy as np
        
        state = GameState(
            ocr_text=step.get("ocr", ""),
            context=json.loads(step.get("state_metadata", "{}")),
            vision_vector=np.frombuffer(step["vision_vector"], dtype=np.float32).tolist() if step.get("vision_vector") else []
        )
        return Observation(state=state, state_hash=step.get("vision_hash", ""))
